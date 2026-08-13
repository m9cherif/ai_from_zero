"""Tests for the configuration system."""

import json
import tempfile
import os
from ..config.base import Config, ConfigError
from ..config.presets import ModelConfig, TrainConfig


class TestModelConfig:
    def test_default_creation(self):
        config = ModelConfig()
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.dropout == 0.1

    def test_custom_values(self):
        config = ModelConfig(d_model=512, n_heads=16, n_layers=12)
        assert config.d_model == 512
        assert config.n_heads == 16
        assert config.n_layers == 12

    def test_validation(self):
        try:
            ModelConfig(d_model=-1)
            assert False, "Should have raised ConfigError"
        except ConfigError:
            pass

    def test_serialization_roundtrip(self):
        config = ModelConfig(d_model=512, n_heads=8, n_layers=12)
        data = config.to_dict()
        restored = ModelConfig.from_dict(data)
        assert restored.d_model == config.d_model
        assert restored.n_heads == config.n_heads
        assert restored.n_layers == config.n_layers

    def test_save_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            config = ModelConfig(d_model=512, n_heads=8)
            config.save(path)
            loaded = ModelConfig.load(path)
            assert loaded.d_model == 512
            assert loaded.n_heads == 8
        finally:
            os.unlink(path)

    def test_merge(self):
        base = ModelConfig(d_model=256, n_heads=8)
        override = ModelConfig(d_model=512)
        base.merge(override)
        assert base.d_model == 512
        assert base.n_heads == 8


class TestTrainConfig:
    def test_full_config_creation(self):
        config = TrainConfig(
            model={"d_model": 512, "n_heads": 8, "n_layers": 6},
            seed=123,
            num_epochs=10,
        )
        assert config.model.d_model == 512
        assert config.seed == 123
        assert config.num_epochs == 10

    def test_config_to_dict(self):
        config = TrainConfig()
        data = config.to_dict()
        assert "model" in data
        assert "optimizer" in data
        assert "scheduler" in data

    def test_resolve_device(self):
        config = TrainConfig(hardware={"device": "cpu"})
        device = config.resolve_device()
        assert device == "cpu"
