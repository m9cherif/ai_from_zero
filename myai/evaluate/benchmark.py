"""Benchmarking framework for model evaluation."""

import time
import math
from typing import List, Dict, Optional, Callable, Any
import torch
from ..core.logging import logger
from ..core.types import Tensor
from ..nn.model import LanguageModel
from ..data.batching import BatchBuilder
from .metrics import PerplexityMetric, AccuracyMetric, TextGenerationMetrics


class Benchmark:
    """Comprehensive model evaluation benchmark."""

    def __init__(self, model: LanguageModel, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer

    def evaluate_perplexity(self, dataset, num_steps: int = 100) -> float:
        """Evaluate perplexity on a dataset."""
        self._model.eval()
        total_loss = 0.0
        total_tokens = 0
        steps = 0

        with torch.no_grad():
            for token_ids in dataset:
                if steps >= num_steps:
                    break

                batch = BatchBuilder(
                    pad_token_id=0,
                    max_length=self._model.config.max_seq_len,
                ).build_batch([token_ids])

                batch = {k: v.to(self._model.device) for k, v in batch.items()}

                outputs = self._model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )

                tokens = (batch["labels"] != -100).sum().item()
                total_loss += outputs["loss"].item() * tokens
                total_tokens += tokens
                steps += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))

        logger.info(f"Perplexity: {perplexity:.2f} (loss={avg_loss:.4f}, tokens={total_tokens})")
        return perplexity

    def measure_throughput(self, batch_size: int = 1, seq_len: int = 128, num_batches: int = 100) -> Dict[str, float]:
        """Measure inference throughput."""
        self._model.eval()
        device = self._model.device

        # Create dummy input
        input_ids = torch.randint(
            0, self._model.config.vocab_size,
            (batch_size, seq_len),
            device=device,
        )

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self._model(input_ids)

        # Benchmark
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.perf_counter()
        total_tokens = 0

        with torch.no_grad():
            for _ in range(num_batches):
                output = self._model(input_ids)
                total_tokens += batch_size * seq_len

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start_time

        throughput = total_tokens / elapsed
        batches_per_sec = num_batches / elapsed

        return {
            "tokens_per_second": throughput,
            "batches_per_second": batches_per_sec,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "elapsed_seconds": elapsed,
        }

    def generate_and_evaluate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate text and evaluate quality metrics."""
        if self._tokenizer is None:
            logger.error("Tokenizer required for text generation evaluation")
            return []

        from ..inference import AutoregressiveGenerator
        generator = AutoregressiveGenerator(
            model=self._model,
            tokenizer=self._tokenizer,
        )

        results = []
        for prompt in prompts:
            start_time = time.time()
            generated = generator.generate(prompt, max_new_tokens=max_new_tokens)
            generation_time = time.time() - start_time

            metrics = {
                "prompt": prompt,
                "generated": generated,
                "generation_time": generation_time,
                "tokens_per_second": len(generated.split()) / max(generation_time, 0.001),
            }
            metrics.update(TextGenerationMetrics.compute_all(generated))
            results.append(metrics)

        return results

    def full_evaluation(
        self,
        eval_dataset=None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a full evaluation suite."""
        results = {}

        # Perplexity
        if eval_dataset is not None:
            results["perplexity"] = self.evaluate_perplexity(eval_dataset)

        # Throughput
        throughput = self.measure_throughput()
        results["throughput"] = throughput

        # Text generation
        if prompts:
            gen_results = self.generate_and_evaluate(prompts)
            results["generation"] = gen_results

        return results
