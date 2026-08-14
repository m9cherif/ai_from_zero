"""Complete training engine with full orchestration."""

import os
import time
import math
from typing import Optional, Dict, Any, Callable, List
import torch
from ..core.logging import logger
from ..core.random import RandomStateManager, get_default_rng
from ..config.presets import TrainConfig
from ..nn.model import LanguageModel, LMConfig
from .optimizer import Optimizer, Adam, AdamW, SGD, build_param_groups
from .scheduler import LRScheduler, WarmupCosineLR
from .gradient import GradientClipper, GradientAccumulator
from .precision import MixedPrecisionManager
from .loop import TrainingLoop
from ..data.streaming import StreamingDataset
from ..checkpoint import CheckpointManager
from ..diagnostics import TrainingMonitor


def build_model_config(config: TrainConfig) -> LMConfig:
    """Translate the user-facing TrainConfig into the model's internal config."""
    model = config.model
    return LMConfig(
        vocab_size=model.vocab_size or config.tokenizer.vocab_size,
        d_model=model.d_model,
        n_heads=model.n_heads,
        d_ff=model.d_ff,
        n_layers=model.n_layers,
        max_seq_len=model.max_seq_len,
        dropout=model.dropout,
        activation=model.activation,
        norm_type=model.norm_type,
        pre_norm=model.pre_norm,
        tie_embeddings=model.weight_tying,
        bias=model.bias,
        n_kv_heads=model.n_kv_heads,
        position_encoding=model.position_encoding,
        rope_base=model.rope_base,
        rope_scaling=model.rope_scaling,
        use_flash=model.use_flash,
        gradient_checkpointing=model.gradient_checkpointing,
        padding_idx=0,
    )


def _build_optimizer(model: LanguageModel, config: TrainConfig) -> Optimizer:
    """Build optimizer from configuration."""
    opt_config = config.optimizer
    opt_type = opt_config.optimizer

    # Weight decay applies to matrices only; biases and norm gains are excluded.
    groups = build_param_groups(model, weight_decay=opt_config.weight_decay)

    if opt_type == "adamw":
        return AdamW(
            groups,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
        )
    if opt_type == "adam":
        return Adam(
            groups,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
        )
    if opt_type == "sgd":
        return SGD(
            groups,
            lr=opt_config.learning_rate,
            momentum=0.9,
            weight_decay=opt_config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {opt_type}")


def _build_scheduler(optimizer: Optimizer, config: TrainConfig) -> LRScheduler:
    """Build learning rate scheduler from configuration."""
    from .scheduler import ConstantLR, CosineLR, LinearLR, WarmupLinearLR

    sched_config = config.scheduler
    total_steps = config.max_steps or 100000
    sched_type = sched_config.scheduler

    if sched_type == "constant":
        return ConstantLR(optimizer)
    if sched_type == "linear":
        return LinearLR(optimizer, total_steps=total_steps)
    if sched_type == "cosine":
        return CosineLR(optimizer, total_steps=total_steps, min_lr_ratio=sched_config.min_lr_ratio)
    if sched_type == "warmup_cosine":
        return WarmupCosineLR(
            optimizer,
            warmup_steps=sched_config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=sched_config.min_lr_ratio,
        )
    if sched_type == "warmup_linear":
        return WarmupLinearLR(
            optimizer,
            warmup_steps=sched_config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=sched_config.min_lr_ratio,
        )
    return ConstantLR(optimizer)


class Trainer:
    """Complete training engine with full lifecycle management."""

    def __init__(self, config: TrainConfig, tokenizer: Any = None):
        self._config = config
        self._tokenizer = tokenizer

        device_str = config.resolve_device()
        self._device = torch.device(device_str)
        logger.info(f"Using device: {device_str}")

        rng = get_default_rng()
        rng.seed_all(config.seed)

        model_config = build_model_config(config)
        self._model = LanguageModel(model_config).to(self._device)

        if config.model.gradient_checkpointing:
            self._model.enable_gradient_checkpointing(True)

        logger.info(f"Model created: {self._model}")
        logger.info(
            f"  parameters: {self._model.num_parameters():,} "
            f"({self._model.num_parameters_excluding_embeddings():,} non-embedding)"
        )

        # torch.compile fuses pointwise chains and cuts Python dispatch overhead.
        # It is opt-in: the first call pays a substantial compilation cost.
        if config.hardware.compile_model:
            logger.info("Compiling model with torch.compile (first step will be slow)...")
            self._model.forward = torch.compile(self._model.forward)

        self._optimizer = _build_optimizer(self._model, config)
        self._scheduler = _build_scheduler(self._optimizer, config)

        max_grad_norm = config.optimizer.max_grad_norm
        self._gradient_clipper = GradientClipper(max_norm=max_grad_norm) if max_grad_norm else None
        self._gradient_accumulator = (
            GradientAccumulator(config.optimizer.gradient_accumulation_steps)
            if config.optimizer.gradient_accumulation_steps > 1
            else None
        )

        self._precision_manager = MixedPrecisionManager(
            enabled=config.hardware.use_mixed_precision,
            dtype=config.hardware.dtype,
            device_type=device_str,
        )

        self._checkpoint_manager = CheckpointManager(
            save_dir=config.checkpoint.save_dir,
            save_every_steps=config.checkpoint.save_every_steps,
            keep_last_n=config.checkpoint.keep_last_n,
        )

        self._monitor = TrainingMonitor()

        self._loop = TrainingLoop(
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            gradient_clipper=self._gradient_clipper,
            gradient_accumulator=self._gradient_accumulator,
            precision_manager=self._precision_manager,
            batch_size=config.data.batch_size,
            pad_token_id=0,
            length_bucketing=config.data.bucketing,
            log_every=config.logging.log_every_steps,
            eval_every=config.eval_every_steps,
            eval_steps=config.eval_steps,
            max_steps=config.max_steps,
            max_epochs=config.num_epochs,
            on_step_end=self._on_step_end,
            on_eval=self._on_eval,
        )

        if config.checkpoint.resume_from:
            self._load_checkpoint(config.checkpoint.resume_from)

    def _on_step_end(self, step: int, metrics: Dict[str, Any]) -> None:
        """Step-level hook: mid-epoch checkpointing on the configured interval."""
        if self._checkpoint_manager.should_save(step):
            self.save_checkpoint(step=step)

    def _on_eval(self, step: int, val_loss: float, is_best: bool) -> None:
        """Eval-level hook: persist the best-validation-loss model."""
        if is_best and self._config.checkpoint.save_best:
            self.save_checkpoint(step=step, is_best=True)

    def save_checkpoint(self, step: Optional[int] = None, is_best: bool = False) -> str:
        """Persist model, optimizer, scheduler, loop and RNG state."""
        tokenizer_state = None
        if self._tokenizer is not None and hasattr(self._tokenizer, "to_dict"):
            tokenizer_state = self._tokenizer.to_dict()

        return self._checkpoint_manager.save(
            model=self._model,
            optimizer=self._optimizer if self._config.checkpoint.save_optimizer else None,
            scheduler=self._scheduler if self._config.checkpoint.save_scheduler else None,
            loop_state=self._loop.state_dict(),
            config=self._config,
            rng_state=get_default_rng().capture_state(),
            tokenizer_state=tokenizer_state,
            step=step,
            is_best=is_best,
        )

    def train(
        self,
        dataset: StreamingDataset,
        val_dataset: Optional[StreamingDataset] = None,
    ) -> Dict[str, Any]:
        """Run the full training process."""
        stats: Dict[str, Any] = {}

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info(f"  Model parameters: {self._model.num_parameters():,}")
        logger.info(f"  Batch size: {self._config.data.batch_size}")
        logger.info(
            f"  Gradient accumulation: {self._config.optimizer.gradient_accumulation_steps} "
            f"(effective batch "
            f"{self._config.data.batch_size * self._config.optimizer.gradient_accumulation_steps})"
        )
        logger.info(f"  Max sequence length: {self._config.data.max_seq_len}")
        logger.info(f"  Max steps: {self._config.max_steps or 'unlimited'}")
        logger.info(f"  Epochs: {self._config.num_epochs}")
        logger.info("=" * 60)

        try:
            for epoch in range(self._config.num_epochs):
                if self._config.max_steps and self._loop.global_step >= self._config.max_steps:
                    break

                logger.info(f"Starting epoch {epoch + 1}/{self._config.num_epochs}")
                stats = self._loop.train_epoch(dataset, val_dataset)
                logger.info(
                    f"Epoch {epoch + 1} done: loss={stats['loss']:.4f}, "
                    f"ppl={stats['perplexity']:.2f}, "
                    f"tokens/s={stats['tokens_per_second']:.0f}"
                )

                self.save_checkpoint(step=self._loop.global_step)

            # Training almost never ends on an eval boundary, so without this the
            # final stretch is never scored and checkpoint_best.pt can be stale -
            # measured worse on held-out text than the final weights.
            if val_dataset is not None:
                self._loop.evaluate_and_track(val_dataset)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            path = self.save_checkpoint(step=self._loop.global_step)
            logger.info(f"Interrupted checkpoint saved to {path}")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Total steps: {self._loop.global_step}")
        logger.info("=" * 60)

        return stats

    def evaluate(self, dataset: StreamingDataset, num_steps: int = 100) -> float:
        """Evaluate the model."""
        return self._loop.evaluate(dataset, num_steps)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training."""
        state = self._checkpoint_manager.load(checkpoint_path)
        if not state:
            return

        self._model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state and self._scheduler is not None:
            self._scheduler.load_state_dict(state["scheduler_state_dict"])
        if "loop_state" in state:
            self._loop.load_state_dict(state["loop_state"])
        if "rng_state" in state:
            get_default_rng().restore_state(state["rng_state"])
        logger.info(
            f"Resumed from checkpoint: {checkpoint_path} (step {self._loop.global_step})"
        )

    @property
    def model(self) -> LanguageModel:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def config(self) -> TrainConfig:
        return self._config
