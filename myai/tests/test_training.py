"""Tests for training components."""

import torch
from ..train.optimizer import SGD, Adam, AdamW
from ..train.scheduler import ConstantLR, CosineLR, WarmupCosineLR
from ..train.gradient import GradientClipper
from ..nn.linear import Linear
from ..nn.module import Module
from ..nn.parameter import Parameter


class TestOptimizer:
    def test_sgd_step(self):
        linear = Linear(10, 5)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        initial_weight = linear._parameters["weight"].data.clone()

        optimizer = SGD(linear.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            y = linear(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Weight should have changed
        assert not torch.equal(initial_weight, linear._parameters["weight"].data)

    def test_adam_step(self):
        linear = Linear(10, 5)
        x = torch.randn(4, 10)

        optimizer = Adam(linear.parameters(), lr=0.001)
        initial_weight = linear._parameters["weight"].data.clone()

        for _ in range(3):
            optimizer.zero_grad()
            y = linear(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        assert not torch.equal(initial_weight, linear._parameters["weight"].data)

    def test_adamw_step(self):
        linear = Linear(10, 5)
        x = torch.randn(4, 10)

        optimizer = AdamW(linear.parameters(), lr=0.001, weight_decay=0.01)
        initial_weight = linear._parameters["weight"].data.clone()

        for _ in range(3):
            optimizer.zero_grad()
            y = linear(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        assert not torch.equal(initial_weight, linear._parameters["weight"].data)


class TestScheduler:
    def test_constant_lr(self):
        linear = Linear(10, 5)
        optimizer = SGD(linear.parameters(), lr=0.01)
        scheduler = ConstantLR(optimizer)
        scheduler.step()
        assert optimizer._param_groups[0]["lr"] == 0.01

    def test_cosine_lr(self):
        linear = Linear(10, 5)
        optimizer = SGD(linear.parameters(), lr=0.01)
        scheduler = CosineLR(optimizer, total_steps=100)
        scheduler.step()
        initial_lr = scheduler._base_lrs[0]
        current_lr = optimizer._param_groups[0]["lr"]
        assert current_lr <= initial_lr

    def test_warmup_cosine(self):
        linear = Linear(10, 5)
        optimizer = SGD(linear.parameters(), lr=0.01)
        scheduler = WarmupCosineLR(optimizer, warmup_steps=10, total_steps=100)

        # During warmup, LR should increase
        scheduler.step()  # step 1
        lr_step1 = optimizer._param_groups[0]["lr"]
        scheduler.step()  # step 2
        lr_step2 = optimizer._param_groups[0]["lr"]
        assert lr_step2 > lr_step1


class TestGradientClipper:
    def test_clip_norm(self):
        linear = Linear(10, 5)
        x = torch.randn(4, 10)

        y = linear(x)
        loss = y.sum()
        loss.backward()

        clipper = GradientClipper(max_norm=0.1)
        total_norm = clipper.clip(linear.parameters())
        assert total_norm >= 0.0
