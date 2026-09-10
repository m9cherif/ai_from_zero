"""Checkpoint serialization with versioning and compatibility."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ..core.logging import logger
from ..core.errors import CheckpointError
from ..nn.model import LanguageModel
from .version import CheckpointVersion


class CheckpointSerializer:
    """Handles checkpoint serialization and deserialization."""

    @staticmethod
    def save(
        path: str,
        model_state_dict: Dict[str, Any],
        config: Any,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        loop_state: Optional[Dict[str, Any]] = None,
        rng_state: Optional[Dict[str, Any]] = None,
        tokenizer_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a complete checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build checkpoint dictionary
        checkpoint = {
            "version": CheckpointVersion.make_version_info(),
            "timestamp": time.time(),
            "model_state_dict": model_state_dict,
            "config": config.to_dict() if hasattr(config, "to_dict") else config,
        }

        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict

        if scheduler_state_dict is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state_dict

        if loop_state is not None:
            checkpoint["loop_state"] = loop_state

        if rng_state is not None:
            checkpoint["rng_state"] = rng_state

        if tokenizer_state is not None:
            checkpoint["tokenizer_state"] = tokenizer_state

        if extra is not None:
            checkpoint.update(extra)

        # Save with atomic write
        temp_path = str(path) + ".tmp"
        try:
            torch.save(checkpoint, temp_path)
        except Exception:
            # A save that fails partway (typically disk-full on a checkpoint
            # this size) leaves a truncated, unusable .tmp file behind. Left
            # in place it does nothing but hold the space that made the save
            # fail in the first place, hostile to exactly the situation it
            # occurs in - so remove it rather than let it linger.
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        os.replace(temp_path, str(path))

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved to {path} ({size_mb:.1f} MB)")

        return str(path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint from disk with validation."""
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}")

        try:
            # weights_only=False is required because a full checkpoint carries
            # more than tensors: the config dict, loop state, and the RNG state
            # (which includes numpy's generator state). These files are produced
            # by this trainer, so only load checkpoints you produced or trust.
            checkpoint = torch.load(
                str(path), map_location=map_location or "cpu", weights_only=False
            )
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint {path}: {e}") from e

        # Validate
        if not CheckpointVersion.validate_checkpoint(checkpoint):
            raise CheckpointError(f"Invalid or corrupted checkpoint: {path}")

        # Check version compatibility
        if "version" in checkpoint:
            CheckpointVersion.check_compatibility(checkpoint["version"])

        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
