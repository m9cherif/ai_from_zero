"""Checkpoint version tracking for compatibility checks."""

from typing import Dict, Any
from ..core.logging import logger


class CheckpointVersion:
    """Tracks checkpoint format version for compatibility."""

    CURRENT_VERSION = (1, 0, 0)

    REQUIRED_KEYS = {
        "version",
        "model_state_dict",
        "config",
        "timestamp",
    }

    @staticmethod
    def make_version_info() -> Dict[str, Any]:
        return {
            "major": CheckpointVersion.CURRENT_VERSION[0],
            "minor": CheckpointVersion.CURRENT_VERSION[1],
            "patch": CheckpointVersion.CURRENT_VERSION[2],
            "version_str": ".".join(str(v) for v in CheckpointVersion.CURRENT_VERSION),
        }

    @staticmethod
    def check_compatibility(version_info: Dict[str, Any]) -> bool:
        """Check if a checkpoint version is compatible with this code."""
        major = version_info.get("major", 0)
        current_major = CheckpointVersion.CURRENT_VERSION[0]

        if major != current_major:
            logger.warning(
                f"Checkpoint major version {major} != current {current_major}. "
                f"Checkpoint may be incompatible."
            )
            return False

        return True

    @staticmethod
    def validate_checkpoint(checkpoint: Dict[str, Any]) -> bool:
        """Validate that a checkpoint has all required keys."""
        for key in CheckpointVersion.REQUIRED_KEYS:
            if key not in checkpoint:
                logger.error(f"Checkpoint missing required key: '{key}'")
                return False
        return True
