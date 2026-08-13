"""Tests for inference components."""

import torch
from ..inference.sampling import TemperatureSampler, TopKSampler, TopPSampler, GreedySampler
from ..nn.model import LanguageModel, LMConfig


class TestSamplers:
    def test_temperature_sampler(self):
        sampler = TemperatureSampler(temperature=1.0)
        logits = torch.randn(2, 100)
        tokens = sampler.sample(logits)
        assert tokens.shape == (2, 1)

    def test_top_k_sampler(self):
        sampler = TopKSampler(k=10)
        logits = torch.randn(2, 100)
        tokens = sampler.sample(logits)
        assert tokens.shape == (2, 1)

    def test_top_p_sampler(self):
        sampler = TopPSampler(p=0.9)
        logits = torch.randn(2, 100)
        tokens = sampler.sample(logits)
        assert tokens.shape == (2, 1)

    def test_greedy_sampler(self):
        sampler = GreedySampler()
        logits = torch.randn(2, 100)
        tokens = sampler.sample(logits)
        assert tokens.shape == (2, 1)
        # Greedy should pick the argmax
        expected = logits.argmax(dim=-1, keepdim=True)
        assert (tokens == expected).all()


class TestModelGeneration:
    def test_generate(self):
        config = LMConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=2,
            max_seq_len=32,
        )
        model = LanguageModel(config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 5))
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            top_k=20,
            top_p=0.9,
            eos_token_id=None,
        )
        assert generated.shape[1] == 15, f"Expected 15, got {generated.shape[1]}"

    def test_generate_greedy(self):
        config = LMConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=2,
            max_seq_len=32,
        )
        model = LanguageModel(config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 5))
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.0,
        )
        assert generated.shape == (1, 10)
