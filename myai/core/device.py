"""Hardware selection: pick the best accelerator, and size CPU threads correctly.

Two things go wrong on servers that do not go wrong on a laptop:

* **The wrong GPU.** ``torch.cuda.is_available()`` says yes and everything
  lands on ``cuda:0``, which on a shared box is often the one someone else is
  already filling. Choosing by *free* memory avoids running out on a machine
  that had plenty of room on another card.

* **The wrong thread count.** ``os.cpu_count()`` reports the host's cores, not
  the container's share. A pod limited to 2 CPUs on a 64-core host still sees
  64, so PyTorch starts 64 threads that spend their time fighting over 2 cores
  worth of runtime. Reading the cgroup quota and the CPU affinity mask gives
  the number actually usable.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .logging import logger


def _cgroup_cpu_limit() -> Optional[float]:
    """CPUs allowed by the cgroup quota, or None when unlimited."""
    # cgroup v2: "<quota> <period>", or "max <period>" when unrestricted.
    v2 = Path("/sys/fs/cgroup/cpu.max")
    if v2.is_file():
        try:
            quota, period = v2.read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                return int(quota) / int(period)
        except (ValueError, OSError):
            pass

    # cgroup v1: quota and period in separate files, quota -1 when unrestricted.
    v1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    v1_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if v1_quota.is_file() and v1_period.is_file():
        try:
            quota = int(v1_quota.read_text().strip())
            period = int(v1_period.read_text().strip())
            if quota > 0 and period > 0:
                return quota / period
        except (ValueError, OSError):
            pass

    return None


def usable_cpu_count() -> int:
    """Number of CPUs this process can actually use.

    The minimum of the scheduler affinity mask and the cgroup quota, which is
    what a container is really allowed, rather than what the host reports.
    """
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        affinity = os.cpu_count() or 1

    limit = _cgroup_cpu_limit()
    if limit is not None:
        # Round down, but never below one - a 0.5-CPU quota still needs a thread.
        return max(1, min(affinity, int(limit)))
    return max(1, affinity)


def list_cuda_devices() -> List[Tuple[int, str, int, int]]:
    """Every visible CUDA device as ``(index, name, free_bytes, total_bytes)``."""
    if not torch.cuda.is_available():
        return []

    devices = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        try:
            free, total = torch.cuda.mem_get_info(index)
        except (RuntimeError, AssertionError):
            # Older drivers, or a device we cannot query - fall back to capacity.
            total = torch.cuda.get_device_properties(index).total_memory
            free = total
        devices.append((index, name, int(free), int(total)))
    return devices


def select_device(prefer: str = "auto") -> torch.device:
    """Resolve a device string, choosing the emptiest GPU when several exist.

    ``prefer`` may be "auto", "cuda", "cpu", "mps", or an explicit device such
    as "cuda:1", which is honoured as given.
    """
    if prefer and prefer != "auto":
        # An explicit choice is the caller's to make, including a bad one.
        if prefer.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(prefer)

    devices = list_cuda_devices()
    if devices:
        # Most free memory wins; ties break toward the lower index.
        index, name, free, total = max(devices, key=lambda d: (d[2], -d[0]))
        if len(devices) > 1:
            logger.info(
                f"Selected cuda:{index} ({name}, {free/1e9:.1f} GB free of "
                f"{total/1e9:.1f} GB) from {len(devices)} GPUs"
            )
        return torch.device(f"cuda:{index}")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def configure_threads(threads: Optional[int] = None) -> int:
    """Point PyTorch at the CPUs this process may actually use.

    Returns the thread count applied. Passing ``threads`` overrides detection.
    """
    resolved = threads if threads and threads > 0 else usable_cpu_count()

    previous = torch.get_num_threads()
    torch.set_num_threads(resolved)
    try:
        # One inter-op thread: this project runs a single sequential graph, so
        # extra coordination threads only add contention.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable before the parallel region starts; ignore afterwards.
        pass

    if previous != resolved:
        logger.info(f"CPU threads: {previous} -> {resolved} (usable CPUs: {usable_cpu_count()})")
    return resolved


def describe(device: torch.device) -> str:
    """One-line summary of what compute was selected."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return (
            f"cuda:{index} {props.name}, {props.total_memory/1e9:.1f} GB, "
            f"capability {props.major}.{props.minor}, {props.multi_processor_count} SMs"
        )
    if device.type == "mps":
        return "mps (Apple Silicon), unified memory"

    limit = _cgroup_cpu_limit()
    detail = f"{usable_cpu_count()} usable CPUs"
    if limit is not None:
        detail += f" (cgroup quota {limit:.2f}, host reports {os.cpu_count()})"
    return f"cpu, {detail}, {torch.get_num_threads()} threads"


def setup(prefer: str = "auto", threads: Optional[int] = None) -> torch.device:
    """Select a device and size CPU threads for it, then log the result."""
    device = select_device(prefer)
    # GPU runs still use CPU threads for the data pipeline, just far fewer.
    configure_threads(threads if threads else (4 if device.type != "cpu" else None))
    logger.info(f"Compute: {describe(device)}")
    return device
