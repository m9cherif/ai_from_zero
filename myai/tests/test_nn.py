"""Tests for neural network components."""

import torch
from ..nn.linear import Linear
from ..nn.embedding import Embedding
from ..nn.activation import ReLU, GELU, SiLU
from ..nn.normalization import LayerNorm, RMSNorm
from ..nn.dropout import Dropout
from ..nn.attention import MultiHeadAttention, CausalSelfAttention
from ..nn.feedforward import FeedForward
from ..nn.positional import SinusoidalPositionalEncoding
from ..nn.transformer import TransformerBlock, TransformerDecoder
from ..nn.model import LanguageModel, LMConfig
from ..nn.loss import CrossEntropyLoss
from ..nn.module import Module


class TestLinear:
    def test_forward_shape(self):
        linear = Linear(128, 256)
        x = torch.randn(4, 128)
        y = linear(x)
        assert y.shape == (4, 256)

    def test_forward_no_bias(self):
        linear = Linear(128, 256, bias=False)
        x = torch.randn(4, 128)
        y = linear(x)
        assert y.shape == (4, 256)

    def test_parameters(self):
        linear = Linear(128, 256)
        params = list(linear.parameters())
        assert len(params) == 2  # weight + bias


class TestEmbedding:
    def test_forward_shape(self):
        emb = Embedding(100, 64)
        x = torch.randint(0, 100, (4, 10))
        y = emb(x)
        assert y.shape == (4, 10, 64)

    def test_padding(self):
        emb = Embedding(100, 64, padding_idx=0)
        x = torch.tensor([[1, 2, 0, 3]])
        y = emb(x)
        assert (y[:, 2, :] == 0).all()


class TestActivation:
    def test_relu(self):
        relu = ReLU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = relu(x)
        assert (y == torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])).all()

    def test_gelu(self):
        gelu = GELU()
        x = torch.randn(4, 16)
        y = gelu(x)
        assert y.shape == x.shape

    def test_silu(self):
        silu = SiLU()
        x = torch.randn(4, 16)
        y = silu(x)
        assert y.shape == x.shape


class TestNormalization:
    def test_layernorm_shape(self):
        ln = LayerNorm(64)
        x = torch.randn(4, 16, 64)
        y = ln(x)
        assert y.shape == x.shape

    def test_rmsnorm_shape(self):
        rn = RMSNorm(64)
        x = torch.randn(4, 16, 64)
        y = rn(x)
        assert y.shape == x.shape


class TestDropout:
    def test_training_mode(self):
        dropout = Dropout(0.5)
        dropout.train()
        x = torch.ones(4, 16)
        y = dropout(x)
        assert (y != 0).sum() > 0  # Some elements kept

    def test_eval_mode(self):
        dropout = Dropout(0.5)
        dropout.eval()
        x = torch.ones(4, 16)
        y = dropout(x)
        assert (y == x).all()


class TestAttention:
    def test_multi_head_shape(self):
        attn = MultiHeadAttention(64, 8)
        x = torch.randn(2, 10, 64)
        y = attn(x, x, x)
        assert y.shape == (2, 10, 64)

    def test_causal_self_attention_shape(self):
        attn = CausalSelfAttention(64, 8)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)


class TestFeedForward:
    def test_forward_shape(self):
        ff = FeedForward(256, 1024)
        x = torch.randn(4, 16, 256)
        y = ff(x)
        assert y.shape == (4, 16, 256)


class TestTransformerBlock:
    def test_forward_shape(self):
        block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)


class TestTransformerDecoder:
    def test_forward_shape(self):
        decoder = TransformerDecoder(d_model=64, n_heads=8, d_ff=256, n_layers=3)
        x = torch.randn(2, 10, 64)
        y = decoder(x)
        assert y.shape == (2, 10, 64)


class TestLanguageModel:
    def test_forward_shape(self):
        config = LMConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=2,
            max_seq_len=32,
        )
        model = LanguageModel(config)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, labels=labels)
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (2, 16, 100)

    def test_generate_shape(self):
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
        generated = model.generate(input_ids, max_new_tokens=10)
        assert generated.shape[1] == 15  # 5 prompt + 10 new

    def test_parameter_count(self):
        config = LMConfig(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=2)
        model = LanguageModel(config)
        assert model.num_parameters() > 0


class TestCrossEntropyLoss:
    def test_loss_computation(self):
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_ignore_index(self):
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.full((2, 4), -100, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0
