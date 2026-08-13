"""Dataset validation: integrity checks before training."""

from typing import List, Optional, Tuple, Dict, Any
from ...core.logging import logger


class DatasetValidator:
    """Validates dataset integrity before training begins."""

    @staticmethod
    def check_empty_samples(sequences: List[List[int]], max_empty_ratio: float = 0.01) -> Tuple[bool, int, int]:
        """Check ratio of empty sequences."""
        if not sequences:
            return False, 0, 0
        empty = sum(1 for s in sequences if len(s) == 0)
        ratio = empty / len(sequences)
        return ratio <= max_empty_ratio, empty, len(sequences)

    @staticmethod
    def check_extreme_lengths(sequences: List[List[int]], max_length: int) -> Tuple[bool, int, int]:
        """Check for sequences exceeding max_length."""
        if not sequences:
            return True, 0, 0
        extreme = sum(1 for s in sequences if len(s) > max_length)
        return extreme == 0, extreme, len(sequences)

    @staticmethod
    def check_invalid_tokens(sequences: List[List[int]], vocab_size: int) -> Tuple[bool, int]:
        """Check for token IDs outside valid range."""
        invalid = 0
        for seq in sequences:
            for tid in seq:
                if tid < 0 or tid >= vocab_size:
                    invalid += 1
        return invalid == 0, invalid

    @staticmethod
    def check_encoding_errors(texts: List[str]) -> Tuple[bool, int]:
        """Check for encoding errors in raw texts."""
        errors = 0
        for text in texts:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                errors += 1
        return errors == 0, errors

    @staticmethod
    def validate_batch(batch: Dict[str, Any]) -> List[str]:
        """Validate a single batch for structural integrity."""
        issues = []
        if "input_ids" not in batch:
            issues.append("Missing 'input_ids'")
        if "labels" not in batch:
            issues.append("Missing 'labels'")

        if "input_ids" in batch and "labels" in batch:
            import torch
            if batch["input_ids"].shape != batch["labels"].shape:
                issues.append(f"Shape mismatch: input_ids {batch['input_ids'].shape} vs labels {batch['labels'].shape}")

        return issues

    def full_validation(
        self,
        sequences: Optional[List[List[int]]] = None,
        texts: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        """Run full validation suite."""
        results = {}

        if sequences is not None:
            ok, count, total = self.check_empty_samples(sequences)
            results["empty_samples"] = {"pass": ok, "count": count, "total": total}

            if max_length:
                ok, count, total = self.check_extreme_lengths(sequences, max_length)
                results["extreme_lengths"] = {"pass": ok, "count": count, "total": total}

            if vocab_size:
                ok, count = self.check_invalid_tokens(sequences, vocab_size)
                results["invalid_tokens"] = {"pass": ok, "count": count}

        if texts is not None:
            ok, count = self.check_encoding_errors(texts)
            results["encoding_errors"] = {"pass": ok, "count": count}

        all_pass = all(v.get("pass", True) for v in results.values())
        results["all_pass"] = all_pass

        return results
