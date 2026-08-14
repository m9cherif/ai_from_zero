"""Training loop: orchestrates the training process."""

import time
import math
from typing import Optional, Dict, Any, Iterable, Iterator, Callable, List
import torch
from ..core.logging import logger
from ..core.types import Tensor
from ..nn.module import Module
from ..nn.model import LanguageModel
from .optimizer import Optimizer, AdamW
from .scheduler import LRScheduler, WarmupCosineLR
from .gradient import GradientClipper, GradientAccumulator
from .precision import MixedPrecisionManager
from ..data.streaming import StreamingDataset
from ..data.batching import BatchBuilder
from ..data.validation import DatasetValidator


class _null_context:
    """No-op context manager used when mixed precision is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def iter_batches(
    dataset: Iterable,
    batch_size: int,
    length_bucketing: bool = True,
    bucket_multiplier: int = 8,
    drop_last: bool = False,
) -> Iterator[List[List[int]]]:
    """Group a stream of token sequences into batches.

    With ``length_bucketing`` the loader buffers ``batch_size * multiplier``
    sequences, sorts them by length and emits contiguous chunks, so each batch
    contains similar-length sequences. Padding is charged against the longest
    member of a batch, so mixing a 10-token and a 500-token sequence wastes most
    of the compute on padding; bucketing typically recovers a large fraction of
    it.
    """
    buffer: List[List[int]] = []
    window = batch_size * max(bucket_multiplier, 1) if length_bucketing else batch_size

    def flush(items: List[List[int]]) -> Iterator[List[List[int]]]:
        if length_bucketing:
            items = sorted(items, key=len)
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            if len(chunk) < batch_size and drop_last:
                continue
            if chunk:
                yield chunk

    for sequence in dataset:
        if not sequence:
            continue
        buffer.append(sequence)
        if len(buffer) >= window:
            yield from flush(buffer)
            buffer = []

    if buffer:
        yield from flush(buffer)


class TrainingLoop:
    """High-level training loop orchestrating the entire training process."""

    def __init__(
        self,
        model: LanguageModel,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        gradient_clipper: Optional[GradientClipper] = None,
        gradient_accumulator: Optional[GradientAccumulator] = None,
        precision_manager: Optional[MixedPrecisionManager] = None,
        batch_size: int = 8,
        pad_token_id: int = 0,
        length_bucketing: bool = True,
        log_every: int = 10,
        eval_every: int = 500,
        eval_steps: int = 100,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        on_step_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        on_eval: Optional[Callable[[int, float, bool], None]] = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._gradient_clipper = gradient_clipper
        self._gradient_accumulator = gradient_accumulator
        self._precision_manager = precision_manager
        self._batch_size = batch_size
        self._length_bucketing = length_bucketing
        self._log_every = log_every
        self._eval_every = eval_every
        self._eval_steps = eval_steps
        self._max_steps = max_steps
        self._max_epochs = max_epochs
        self._on_step_end = on_step_end
        self._on_eval = on_eval

        # Built once and reused: constructing it per batch was pure overhead.
        self._batch_builder = BatchBuilder(
            pad_token_id=pad_token_id,
            max_length=model.config.max_seq_len,
        )

        self._global_step = 0
        self._epoch = 0
        self._best_loss = float("inf")
        self._running_loss = 0.0
        self._running_tokens = 0
        self._last_log_time = time.time()
        self._last_log_tokens = 0
        self._start_time = time.time()

    def _prepare_batch(self, sequences: List[List[int]]) -> Dict[str, Tensor]:
        batch = self._batch_builder.build_batch(sequences)
        device = self._model.device
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def train_epoch(
        self,
        train_dataset: StreamingDataset,
        val_dataset: Optional[StreamingDataset] = None,
    ) -> Dict[str, Any]:
        """Train for one epoch over the dataset."""
        self._model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        batch_count = 0
        grad_norm = 0.0

        accumulator = self._gradient_accumulator

        for sequences in iter_batches(
            train_dataset,
            batch_size=self._batch_size,
            length_bucketing=self._length_bucketing,
        ):
            batch = self._prepare_batch(sequences)

            # autocast only wraps the forward pass; the backward pass inherits
            # the dtypes recorded on the graph.
            forward_ctx = (
                self._precision_manager.get_forward_context()
                if self._precision_manager is not None
                else _null_context()
            )
            with forward_ctx:
                outputs = self._model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )
                loss = outputs["loss"]

            # Report the true loss, but backpropagate the scaled one.
            batch_loss = loss.detach()

            if accumulator is not None:
                loss = accumulator.scale_loss(loss)
            if self._precision_manager is not None:
                loss = self._precision_manager.scale_loss(loss)

            loss.backward()

            should_step = accumulator is None or accumulator.should_step()

            if should_step:
                if self._precision_manager is not None:
                    self._precision_manager.unscale_gradients(self._optimizer)

                if self._gradient_clipper is not None:
                    grad_norm = self._gradient_clipper.clip(self._model.parameters())

                if self._precision_manager is not None:
                    self._precision_manager.step_optimizer(self._optimizer)
                elif self._optimizer is not None:
                    self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

                if self._optimizer is not None:
                    self._optimizer.zero_grad()

                self._global_step += 1

            batch_tokens = int((batch["labels"] != -100).sum().item())
            loss_value = float(batch_loss.item())
            epoch_loss += loss_value
            epoch_tokens += batch_tokens
            batch_count += 1
            self._running_loss += loss_value
            self._running_tokens += batch_tokens

            if should_step and self._global_step % self._log_every == 0:
                self._log_progress(loss_value, grad_norm)

            if should_step and self._on_step_end is not None:
                self._on_step_end(self._global_step, {"loss": loss_value, "grad_norm": grad_norm})

            if (
                val_dataset is not None
                and should_step
                and self._global_step % self._eval_every == 0
            ):
                self.evaluate_and_track(val_dataset)

            if self._max_steps and self._global_step >= self._max_steps:
                break

        # A partial accumulation window would otherwise leak stale gradients
        # into the next epoch's first step.
        if accumulator is not None and accumulator.is_pending:
            if self._optimizer is not None:
                self._optimizer.zero_grad()
            accumulator.reset()

        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = max(time.time() - self._start_time, 1e-6)

        self._epoch += 1

        return {
            "epoch": self._epoch - 1,
            "global_step": self._global_step,
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20)),
            "tokens": epoch_tokens,
            "tokens_per_second": epoch_tokens / elapsed,
        }

    @torch.no_grad()
    def evaluate_and_track(self, dataset, num_steps: Optional[int] = None) -> float:
        """Evaluate, update the best loss, and notify ``on_eval``.

        Returns the validation loss. Restores training mode, so this is safe to
        call from inside the training loop.
        """
        val_loss = self.evaluate(dataset, num_steps or self._eval_steps)
        self._model.train()

        is_best = val_loss < self._best_loss
        if is_best:
            self._best_loss = val_loss
        if self._on_eval is not None:
            self._on_eval(self._global_step, val_loss, is_best)
        return val_loss

    def evaluate(self, dataset: StreamingDataset, num_steps: int = 100) -> float:
        """Evaluate the model on a dataset. Returns token-weighted mean loss."""
        self._model.eval()
        total_loss = 0.0
        total_tokens = 0
        steps = 0

        for sequences in iter_batches(
            dataset, batch_size=self._batch_size, length_bucketing=self._length_bucketing
        ):
            if steps >= num_steps:
                break

            batch = self._prepare_batch(sequences)
            outputs = self._model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch.get("attention_mask"),
            )

            # Loss is a mean over predicted positions; the last token of each
            # sequence has no target, hence the -1 per row.
            predicted = int((batch["labels"][:, 1:] != -100).sum().item())
            total_loss += outputs["loss"].item() * predicted
            total_tokens += predicted
            steps += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))

        logger.info(
            f"Evaluation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}, "
            f"batches={steps}, tokens={total_tokens}"
        )

        return avg_loss

    def _current_lr(self) -> float:
        if self._optimizer is not None and self._optimizer.param_groups:
            return float(self._optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    def _log_progress(self, batch_loss: float, grad_norm: float) -> None:
        """Log training progress with interval-local throughput."""
        now = time.time()
        window = max(now - self._last_log_time, 1e-6)
        window_tokens = self._running_tokens - self._last_log_tokens
        tokens_per_sec = window_tokens / window

        self._last_log_time = now
        self._last_log_tokens = self._running_tokens

        logger.info(
            f"Step {self._global_step}: loss={batch_loss:.4f}, "
            f"ppl={math.exp(min(batch_loss, 20)):.2f}, "
            f"lr={self._current_lr():.2e}, grad_norm={grad_norm:.4f}, "
            f"tokens/s={tokens_per_sec:.0f}, elapsed={now - self._start_time:.0f}s"
        )

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self._global_step,
            "epoch": self._epoch,
            "best_loss": self._best_loss,
            "running_loss": self._running_loss,
            "running_tokens": self._running_tokens,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._global_step = state.get("global_step", 0)
        self._epoch = state.get("epoch", 0)
        self._best_loss = state.get("best_loss", float("inf"))
        self._running_loss = state.get("running_loss", 0.0)
        self._running_tokens = state.get("running_tokens", 0)
