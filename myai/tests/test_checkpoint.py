"""Tests for checkpoint system."""

import tempfile
import os
from pathlib import Path
import torch
from ..checkpoint.serializer import CheckpointSerializer
from ..checkpoint.manager import CheckpointManager
from ..checkpoint.version import CheckpointVersion
from ..nn.model import LanguageModel, LMConfig
from ..config.presets import TrainConfig


class TestCheckpointVersion:
    def test_version_info(self):
        version = CheckpointVersion.make_version_info()
        assert "major" in version
        assert "minor" in version
        assert "patch" in version
        assert version["major"] == 1

    def test_check_compatibility(self):
        compatible = CheckpointVersion.check_compatibility({"major": 1, "minor": 0, "patch": 0})
        assert compatible


class TestCheckpointSerializer:
    def test_save_and_load(self):
        config = TrainConfig()
        model = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            CheckpointSerializer.save(
                path=path,
                model_state_dict=model.state_dict(),
                config=config,
                extra={"test_key": "test_value"},
            )
            loaded = CheckpointSerializer.load(path)
            assert "model_state_dict" in loaded
            assert "config" in loaded
            assert loaded.get("test_key") == "test_value"
            assert "version" in loaded
        finally:
            os.unlink(path)


class TestCheckpointManager:
    def test_save_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, save_every_steps=100, keep_last_n=3)
            config = TrainConfig()
            model = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))

            for step in range(5):
                loop_state = {"global_step": step}
                manager.save(model=model, config=config, loop_state=loop_state)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) <= 3  # keep_last_n=3


class TestBestCheckpoint:
    @staticmethod
    def _model():
        return LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))

    def test_is_best_writes_best_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, save_every_steps=100, keep_last_n=3)
            model = self._model()

            assert not manager.best_checkpoint_path().exists()
            manager.save(model=model, config=TrainConfig(), step=1, is_best=True)
            assert manager.best_checkpoint_path().exists()

    def test_ordinary_save_leaves_best_untouched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, save_every_steps=100, keep_last_n=3)
            manager.save(model=self._model(), config=TrainConfig(), step=1)
            assert not manager.best_checkpoint_path().exists()
            assert manager.resume_from_best() is None

    def test_best_survives_rotation(self):
        """The whole point of copying: the best model is usually not in the last N."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, save_every_steps=100, keep_last_n=2)
            model = self._model()

            # Step 0 is the best; ten more saves push its step file out of rotation.
            manager.save(model=model, config=TrainConfig(), step=0, is_best=True)
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            for param in model.parameters():
                with torch.no_grad():
                    param.data.add_(1.0)
            for step in range(1, 11):
                manager.save(model=model, config=TrainConfig(), step=step)

            assert len(manager.list_checkpoints()) == 2
            assert not (Path(tmpdir) / "checkpoint_step_0.pt").exists()

            restored = manager.resume_from_best()
            assert restored is not None
            for key, value in restored["model_state_dict"].items():
                assert torch.allclose(value, best_weights[key]), key

    def test_best_and_latest_diverge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, save_every_steps=100, keep_last_n=5)
            model = self._model()

            manager.save(model=model, config=TrainConfig(), step=1, is_best=True)
            with torch.no_grad():
                for param in model.parameters():
                    param.data.add_(1.0)
            manager.save(model=model, config=TrainConfig(), step=2)

            best = manager.resume_from_best()["model_state_dict"]
            latest = manager.resume_from_latest()["model_state_dict"]
            assert not torch.allclose(best["embedding.weight"], latest["embedding.weight"])
