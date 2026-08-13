"""Tests for the modern architecture components.

These cover the properties that unit tests on shapes alone cannot catch:
position awareness, KV-cache correctness, and equivalence between the
hand-written and fused attention paths.
"""

import math
import pytest
import torch

from ..nn.model import LanguageModel, LMConfig
from ..nn.attention import CausalSelfAttention, KVCache, repeat_kv, _HAS_SDPA
from ..nn.positional import RoPE, ALiBi, SinusoidalPositionalEncoding, build_position_encoder
from ..nn.parameter import Parameter
from ..nn.logits import apply_top_k, apply_top_p, apply_min_p, apply_repetition_penalty
from ..train.optimizer import AdamW, SGD, build_param_groups


def make_model(**overrides) -> LanguageModel:
    config = dict(
        vocab_size=64, d_model=32, n_heads=4, d_ff=64, n_layers=2,
        max_seq_len=32, dropout=0.0,
    )
    config.update(overrides)
    torch.manual_seed(0)
    model = LanguageModel(LMConfig(**config))
    model.eval()
    return model


class TestPositionAwareness:
    def test_model_is_not_permutation_invariant(self):
        """Reordering the prefix must change the prediction.

        Without a position encoding, attention is permutation-equivariant and a
        one-layer model returns identical logits for any prefix ordering - the
        model would be a bag of words.
        """
        model = make_model(n_layers=1, position_encoding="rope")
        with torch.no_grad():
            a = model(torch.tensor([[5, 9, 2]]))["logits"][0, -1]
            b = model(torch.tensor([[9, 5, 2]]))["logits"][0, -1]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_no_position_encoding_is_permutation_invariant(self):
        """The control case: position_encoding="none" reproduces the old behaviour."""
        model = make_model(n_layers=1, position_encoding="none")
        with torch.no_grad():
            a = model(torch.tensor([[5, 9, 2]]))["logits"][0, -1]
            b = model(torch.tensor([[9, 5, 2]]))["logits"][0, -1]
        assert torch.allclose(a, b, atol=1e-5)

    @pytest.mark.parametrize("kind", ["rope", "alibi", "sinusoidal", "learned"])
    def test_every_encoding_is_position_aware(self, kind):
        model = make_model(n_layers=1, position_encoding=kind)
        with torch.no_grad():
            a = model(torch.tensor([[5, 9, 2]]))["logits"][0, -1]
            b = model(torch.tensor([[9, 5, 2]]))["logits"][0, -1]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_unknown_encoding_rejected(self):
        with pytest.raises(ValueError):
            build_position_encoder("banana", 32, 8, 4, 16)


class TestRoPE:
    def test_preserves_shape_and_norm(self):
        rope = RoPE(dim=16, max_seq_len=32)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_out, k_out = rope(q, k)

        assert q_out.shape == q.shape and k_out.shape == k.shape
        # Rotation is norm-preserving.
        assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)

    def test_attention_score_depends_only_on_relative_distance(self):
        """The defining property of RoPE: <q_m, k_n> is a function of (m - n)."""
        rope = RoPE(dim=16, max_seq_len=64)
        torch.manual_seed(0)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)

        def score(pos_q: int, pos_k: int) -> float:
            qr, _ = rope(q, q, offset=pos_q)
            _, kr = rope(k, k, offset=pos_k)
            return float((qr * kr).sum())

        assert score(5, 3) == pytest.approx(score(12, 10), abs=1e-4)
        assert score(9, 1) == pytest.approx(score(20, 12), abs=1e-4)

    def test_odd_dimension_rejected(self):
        with pytest.raises(ValueError):
            RoPE(dim=15)


class TestGroupedQueryAttention:
    def test_repeat_kv_expands_groups(self):
        x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float).view(2, 2, 3, 4)
        out = repeat_kv(x, 3)
        assert out.shape == (2, 6, 3, 4)
        # Each KV head is repeated contiguously across its query group.
        assert torch.equal(out[:, 0], x[:, 0])
        assert torch.equal(out[:, 1], x[:, 0])
        assert torch.equal(out[:, 3], x[:, 1])

    def test_gqa_shrinks_parameters(self):
        mha = make_model(n_heads=4, n_kv_heads=4)
        gqa = make_model(n_heads=4, n_kv_heads=1)
        assert gqa.num_parameters() < mha.num_parameters()

    def test_gqa_forward_shape(self):
        model = make_model(n_heads=4, n_kv_heads=2)
        out = model(torch.randint(0, 64, (2, 7)))
        assert out["logits"].shape == (2, 7, 64)

    def test_indivisible_kv_heads_rejected(self):
        from ..core.errors import NNError
        with pytest.raises(NNError):
            CausalSelfAttention(d_model=32, n_heads=4, n_kv_heads=3)


class TestKVCache:
    def test_cache_matches_full_recompute(self):
        """Incremental decoding must equal running the whole prefix each step."""
        model = make_model(n_layers=2)
        prompt = torch.randint(0, 64, (2, 5))

        greedy_cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
        greedy_full = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)

        assert torch.equal(greedy_cached, greedy_full)

    def test_cache_logits_match_step_by_step(self):
        model = make_model(n_layers=2)
        ids = torch.randint(0, 64, (1, 6))

        with torch.no_grad():
            reference = model(ids)["logits"]

            cache = model.build_cache(batch_size=1)
            collected = []
            for i in range(ids.shape[1]):
                out = model(ids[:, i:i + 1], cache=cache, offset=i)
                collected.append(out["logits"])
            incremental = torch.cat(collected, dim=1)

        assert torch.allclose(reference, incremental, atol=1e-4)

    def test_cache_overflow_raises(self):
        from ..core.errors import NNError
        cache = KVCache(1, 1, 2, 4, max_seq_len=4, device=torch.device("cpu"))
        cache.update(0, torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))
        cache.advance(4)
        with pytest.raises(NNError):
            cache.update(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))


@pytest.mark.skipif(not _HAS_SDPA, reason="fused SDPA unavailable")
class TestFlashAttentionPath:
    def test_fused_matches_manual(self):
        model = make_model(n_layers=2)
        ids = torch.randint(0, 64, (2, 9))

        with torch.no_grad():
            manual = model(ids)["logits"]
            model.set_flash_attention(True)
            fused = model(ids)["logits"]
            model.set_flash_attention(False)

        assert torch.allclose(manual, fused, atol=1e-4)


class TestWeightTying:
    def test_tied_weights_share_one_parameter(self):
        model = make_model(tie_embeddings=True)
        assert model._lm_head._parameters["weight"] is model._embedding._parameters["weight"]

    def test_tied_parameter_counted_once(self):
        tied = make_model(tie_embeddings=True)
        untied = make_model(tie_embeddings=False)
        vocab_matrix = 64 * 32
        assert untied.num_parameters() - tied.num_parameters() == vocab_matrix

    def test_tied_weights_receive_one_update(self):
        """A shared parameter must appear once in the optimizer's list."""
        model = make_model(tie_embeddings=True)
        params = list(model.parameters())
        ids = [id(p) for p in params]
        assert len(ids) == len(set(ids))


class TestParameterDeviceMove:
    def test_to_preserves_leaf_status(self):
        """A moved parameter must stay a leaf, or it silently never gets .grad.

        tensor.to() is an autograd op: its output is non-leaf and .grad is never
        populated. That made GPU training run without learning anything.
        """
        p = Parameter(torch.randn(3, 3))
        assert p.data.is_leaf
        p.to(dtype=torch.float64)
        assert p.data.is_leaf
        assert p.data.requires_grad

    def test_gradients_survive_a_move(self):
        p = Parameter(torch.randn(4, 4))
        p.to(dtype=torch.float64)
        (p.data.sum() * 2).backward()
        assert p.grad is not None
        assert torch.allclose(p.grad, torch.full((4, 4), 2.0, dtype=torch.float64))

    def test_model_move_keeps_tying(self):
        model = make_model(tie_embeddings=True)
        model.to(dtype=torch.float64)
        assert model._lm_head._parameters["weight"] is model._embedding._parameters["weight"]


class TestGradientCheckpointing:
    def test_produces_same_gradients(self):
        torch.manual_seed(0)
        plain = make_model(n_layers=2)
        plain.train()
        ids = torch.randint(0, 64, (2, 8))

        plain(ids, labels=ids)["loss"].backward()
        baseline = [p.grad.clone() for p in plain.parameters()]

        plain.zero_grad()
        plain.enable_gradient_checkpointing(True)
        plain(ids, labels=ids)["loss"].backward()
        checkpointed = [p.grad.clone() for p in plain.parameters()]

        for a, b in zip(baseline, checkpointed):
            assert torch.allclose(a, b, atol=1e-5)


class TestLogitProcessors:
    def test_top_k_keeps_exactly_k(self):
        logits = torch.randn(3, 50)
        filtered = apply_top_k(logits, 5)
        assert (filtered > float("-inf")).sum(dim=-1).tolist() == [5, 5, 5]

    def test_top_p_always_keeps_best_token(self):
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.1]])
        filtered = apply_top_p(logits, 0.01)
        assert (filtered > float("-inf")).sum().item() == 1
        assert filtered.argmax().item() == 0

    def test_min_p_scales_with_confidence(self):
        confident = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        flat = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
        kept_confident = (apply_min_p(confident, 0.1) > float("-inf")).sum().item()
        kept_flat = (apply_min_p(flat, 0.1) > float("-inf")).sum().item()
        assert kept_confident < kept_flat

    def test_repetition_penalty_moves_scores_toward_zero(self):
        logits = torch.tensor([[2.0, -2.0, 1.0]])
        generated = torch.tensor([[0, 1]])
        out = apply_repetition_penalty(logits, generated, 2.0)
        assert out[0, 0].item() == pytest.approx(1.0)   # positive divided
        assert out[0, 1].item() == pytest.approx(-4.0)  # negative multiplied
        assert out[0, 2].item() == pytest.approx(1.0)   # untouched


class TestParamGroups:
    def test_one_dimensional_params_excluded_from_decay(self):
        model = make_model(bias=True, norm_type="layernorm")
        groups = build_param_groups(model, weight_decay=0.1)

        assert len(groups) == 2
        decayed, undecayed = groups
        assert decayed["weight_decay"] == 0.1
        assert undecayed["weight_decay"] == 0.0
        assert all(p.data.dim() >= 2 for p in decayed["params"])
        assert all(p.data.dim() < 2 for p in undecayed["params"])

    def test_optimizer_accepts_groups(self):
        model = make_model()
        opt = AdamW(build_param_groups(model, 0.1), lr=1e-3)
        model.train()
        model(torch.randint(0, 64, (2, 6)), labels=torch.randint(0, 64, (2, 6)))["loss"].backward()
        before = model._embedding._parameters["weight"].data.clone()
        opt.step()
        assert not torch.equal(before, model._embedding._parameters["weight"].data)


class TestOptimizerStateDict:
    def test_state_dict_is_callable(self):
        """It is a method, not a property - checkpointing calls it as one."""
        model = make_model()
        opt = AdamW(model.parameters(), lr=1e-3)
        state = opt.state_dict()
        assert isinstance(state, dict)
        assert "step" in state

    def test_roundtrip_restores_moments(self):
        model = make_model()
        opt = AdamW(model.parameters(), lr=1e-2)
        model.train()
        model(torch.randint(0, 64, (2, 6)), labels=torch.randint(0, 64, (2, 6)))["loss"].backward()
        opt.step()

        saved = opt.state_dict()

        fresh_model = make_model()
        fresh = AdamW(fresh_model.parameters(), lr=1e-2)
        fresh.load_state_dict(saved)

        assert fresh._step_count == opt._step_count
        assert len(fresh._state) == len(opt._state)
