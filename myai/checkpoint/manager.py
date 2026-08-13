"""Checkpoint management: saving, loading, and maintaining checkpoints."""

import os
import re
import glob
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from ..core.logging import logger
from ..core.errors import CheckpointError
from .serializer import CheckpointSerializer


class CheckpointManager:
    """Manages the full checkpoint lifecycle.

    Handles automatic saving, naming, rotation, and resumption.
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_every_steps: int = 1000,
        keep_last_n: int = 5,
    ):
        self._save_dir = Path(save_dir)
        self._save_every_steps = save_every_steps
        self._keep_last_n = keep_last_n
        self._last_save_step = -1

        self._save_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, step: int) -> Path:
        return self._save_dir / f"checkpoint_step_{step}.pt"

    def best_checkpoint_path(self) -> Path:
        return self._save_dir / "checkpoint_best.pt"

    def _latest_checkpoint_path(self) -> Path:
        return self._save_dir / "checkpoint_latest.pt"

    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        loop_state=None,
        config=None,
        rng_state=None,
        tokenizer_state=None,
        step: Optional[int] = None,
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: The model to save
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            loop_state: Optional training loop state
            config: Training configuration
            rng_state: Random state for reproducibility
            tokenizer_state: Optional tokenizer state
            step: Current training step (overrides auto-detection)
            is_best: Also mirror this checkpoint to ``checkpoint_best.pt``

        Returns:
            Path to saved checkpoint
        """
        if step is None and loop_state is not None:
            step = loop_state.get("global_step", 0)
        elif step is None:
            step = 0

        # Serialize
        path = self._checkpoint_path(step)
        saved_path = CheckpointSerializer.save(
            path=str(path),
            model_state_dict=model.state_dict(),
            config=config,
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            loop_state=loop_state,
            rng_state=rng_state,
            tokenizer_state=tokenizer_state,
        )

        # Mirror to "latest". Copying the bytes avoids the deserialize +
        # reserialize round-trip the previous implementation paid on every save.
        latest_path = self._latest_checkpoint_path()
        shutil.copyfile(str(path), str(latest_path))

        # "best" is a copy rather than a pointer so it survives rotation: the
        # step file it came from is eventually deleted by _cleanup_old_checkpoints,
        # and the best model is usually not among the last N.
        if is_best:
            shutil.copyfile(str(path), str(self.best_checkpoint_path()))
            logger.info(f"New best checkpoint at step {step}")

        self._cleanup_old_checkpoints()

        self._last_save_step = step
        return saved_path

    def should_save(self, step: int) -> bool:
        """True when ``step`` lands on the configured save interval."""
        if self._save_every_steps <= 0:
            return False
        return step > 0 and step % self._save_every_steps == 0 and step != self._last_save_step

    def load(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint."""
        return CheckpointSerializer.load(path)

    def resume_from_latest(self) -> Optional[Dict[str, Any]]:
        """Resume from the latest checkpoint if available."""
        latest = self._latest_checkpoint_path()
        if latest.exists():
            return self.load(str(latest))
        return None

    def resume_from_best(self) -> Optional[Dict[str, Any]]:
        """Load the best-validation-loss checkpoint if one was saved."""
        best = self.best_checkpoint_path()
        if best.exists():
            return self.load(str(best))
        return None

    @staticmethod
    def _step_of(path: str) -> int:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.replace("\\", "/"))
        return int(match.group(1)) if match else -1

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()

        while len(checkpoints) > self._keep_last_n:
            oldest = checkpoints.pop(0)
            os.remove(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints, oldest first.

        Sorted by step number, not lexicographically - a plain string sort puts
        step_10 before step_9 and would delete the newest checkpoints first.
        """
        pattern = str(self._save_dir / "checkpoint_step_*.pt")
        return sorted(glob.glob(pattern), key=self._step_of)
