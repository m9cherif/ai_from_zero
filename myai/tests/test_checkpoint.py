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
