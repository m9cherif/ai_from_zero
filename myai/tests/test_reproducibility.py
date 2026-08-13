"""Tests for reproducibility across runs."""

import torch
from ..core.random import RandomStateManager, set_seed
from ..nn.model import LanguageModel, LMConfig


class TestReproducibility:
    def test_deterministic_initialization(self):
        set_seed(42)
        model1 = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))
        params1 = [p.data.clone() for p in model1.parameters()]

        set_seed(42)
        model2 = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))
        params2 = [p.data.clone() for p in model2.parameters()]

        for p1, p2 in zip(params1, params2):
            assert torch.equal(p1, p2), "Model parameters differ between runs with same seed"

    def test_different_seeds_different_params(self):
        set_seed(42)
        model1 = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))
        params1 = [p.data.clone() for p in model1.parameters()]

        set_seed(99)
        model2 = LanguageModel(LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2))
        params2 = [p.data.clone() for p in model2.parameters()]

        any_different = False
        for p1, p2 in zip(params1, params2):
            if not torch.equal(p1, p2):
                any_different = True
                break
        assert any_different, "Different seeds should produce different parameters"

    def test_random_state_capture_restore(self):
        rng = RandomStateManager(seed=42)
        state = rng.capture_state()
        # Do some random operations
        _ = torch.randn(10)
        # Restore
        rng.restore_state(state)
        val1 = torch.randn(1).item()
        rng.restore_state(state)
        val2 = torch.randn(1).item()
        assert val1 == val2, "Random state restoration should produce same values"
