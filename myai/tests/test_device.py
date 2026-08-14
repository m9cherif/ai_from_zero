"""Tests for hardware selection.

The interesting cases - a cgroup quota below the host core count, several GPUs
with different amounts of free memory - do not exist on the machine running
these tests, so they are simulated.
"""

import torch

from ..core import device as dev


class TestCgroupLimit:
    def test_v2_quota_is_read(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("150000 100000\n")
        monkeypatch.setattr(dev, "Path", lambda p: cpu_max if p.endswith("cpu.max") else tmp_path / "absent")
        assert dev._cgroup_cpu_limit() == 1.5

    def test_v2_max_means_unlimited(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("max 100000\n")
        monkeypatch.setattr(dev, "Path", lambda p: cpu_max if p.endswith("cpu.max") else tmp_path / "absent")
        assert dev._cgroup_cpu_limit() is None

    def test_malformed_quota_does_not_raise(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("garbage\n")
        monkeypatch.setattr(dev, "Path", lambda p: cpu_max if p.endswith("cpu.max") else tmp_path / "absent")
        assert dev._cgroup_cpu_limit() is None


class TestUsableCpuCount:
    def test_quota_below_affinity_wins(self, monkeypatch):
        """A 2-CPU container on a 64-core host must not start 64 threads."""
        monkeypatch.setattr(dev.os, "sched_getaffinity", lambda _: set(range(64)))
        monkeypatch.setattr(dev, "_cgroup_cpu_limit", lambda: 2.0)
        assert dev.usable_cpu_count() == 2

    def test_affinity_below_quota_wins(self, monkeypatch):
        monkeypatch.setattr(dev.os, "sched_getaffinity", lambda _: {0, 1})
        monkeypatch.setattr(dev, "_cgroup_cpu_limit", lambda: 32.0)
        assert dev.usable_cpu_count() == 2

    def test_fractional_quota_still_gives_one_thread(self, monkeypatch):
        monkeypatch.setattr(dev.os, "sched_getaffinity", lambda _: set(range(8)))
        monkeypatch.setattr(dev, "_cgroup_cpu_limit", lambda: 0.5)
        assert dev.usable_cpu_count() == 1

    def test_unlimited_uses_affinity(self, monkeypatch):
        monkeypatch.setattr(dev.os, "sched_getaffinity", lambda _: set(range(4)))
        monkeypatch.setattr(dev, "_cgroup_cpu_limit", lambda: None)
        assert dev.usable_cpu_count() == 4

    def test_real_machine_is_sane(self):
        count = dev.usable_cpu_count()
        assert count >= 1
        assert count <= (dev.os.cpu_count() or 1)


class TestSelectDevice:
    def test_picks_gpu_with_most_free_memory(self, monkeypatch):
        # cuda:0 nearly full, cuda:1 mostly free - the emptiest must win.
        monkeypatch.setattr(dev, "list_cuda_devices", lambda: [
            (0, "A100", 2_000_000_000, 40_000_000_000),
            (1, "A100", 38_000_000_000, 40_000_000_000),
        ])
        assert dev.select_device("auto") == torch.device("cuda:1")

    def test_ties_break_to_lower_index(self, monkeypatch):
        monkeypatch.setattr(dev, "list_cuda_devices", lambda: [
            (0, "A100", 40_000_000_000, 40_000_000_000),
            (1, "A100", 40_000_000_000, 40_000_000_000),
        ])
        assert dev.select_device("auto") == torch.device("cuda:0")

    def test_explicit_device_is_honoured(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(dev, "list_cuda_devices", lambda: [
            (0, "A100", 1, 40_000_000_000),
            (1, "A100", 40_000_000_000, 40_000_000_000),
        ])
        # An explicit request must not be silently redirected to the emptier card.
        assert dev.select_device("cuda:0") == torch.device("cuda:0")

    def test_cuda_request_without_cuda_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert dev.select_device("cuda") == torch.device("cpu")

    def test_no_accelerator_gives_cpu(self, monkeypatch):
        monkeypatch.setattr(dev, "list_cuda_devices", lambda: [])
        monkeypatch.setattr(torch.backends, "mps", None, raising=False)
        assert dev.select_device("auto") == torch.device("cpu")

    def test_cpu_is_selectable_on_this_machine(self):
        assert dev.select_device("cpu") == torch.device("cpu")


class TestConfigureThreads:
    def test_explicit_override_is_applied(self):
        before = torch.get_num_threads()
        try:
            assert dev.configure_threads(2) == 2
            assert torch.get_num_threads() == 2
        finally:
            torch.set_num_threads(before)

    def test_detection_used_when_unset(self):
        before = torch.get_num_threads()
        try:
            assert dev.configure_threads() == dev.usable_cpu_count()
        finally:
            torch.set_num_threads(before)

    def test_zero_and_negative_fall_back_to_detection(self):
        before = torch.get_num_threads()
        try:
            assert dev.configure_threads(0) == dev.usable_cpu_count()
            assert dev.configure_threads(-4) == dev.usable_cpu_count()
        finally:
            torch.set_num_threads(before)


class TestDescribe:
    def test_cpu_description_mentions_threads(self):
        text = dev.describe(torch.device("cpu"))
        assert "cpu" in text
        assert "threads" in text
