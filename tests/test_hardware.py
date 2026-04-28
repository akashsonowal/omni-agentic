"""Tests for the hardware abstraction layer."""

from __future__ import annotations

import pytest

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.hardware.pool import DevicePool
from omni_agentic.hardware.scheduler import HardwareScheduler, SchedulingHint


# ------------------------------------------------------------------ #
# Device                                                              #
# ------------------------------------------------------------------ #


class TestDevice:
    def test_cpu_device(self):
        dev = Device.cpu()
        assert dev.type == DeviceType.CPU
        assert dev.torch_id == "cpu"
        assert not dev.is_accelerator

    def test_cuda_device_torch_id(self):
        dev = Device(type=DeviceType.CUDA, index=0, name="Test GPU")
        assert dev.torch_id == "cuda:0"
        assert dev.is_accelerator

    def test_cuda_device_index(self):
        dev = Device(type=DeviceType.CUDA, index=1)
        assert dev.torch_id == "cuda:1"

    def test_mps_device(self):
        dev = Device.mps()
        assert dev.type == DeviceType.MPS
        assert dev.torch_id == "mps"
        assert dev.is_accelerator

    def test_rocm_uses_cuda_string(self):
        dev = Device(type=DeviceType.ROCM, index=0)
        assert dev.torch_id == "cuda:0"

    def test_total_memory_gb(self):
        dev = Device(type=DeviceType.CUDA, index=0, total_memory_bytes=8 * 1024**3)
        assert pytest.approx(dev.total_memory_gb, 0.01) == 8.0

    def test_no_memory(self):
        dev = Device.cpu()
        assert dev.total_memory_gb is None

    def test_str(self):
        dev = Device(type=DeviceType.CUDA, index=0, name="A100", total_memory_bytes=40 * 1024**3)
        s = str(dev)
        assert "cuda:0" in s
        assert "A100" in s

    def test_frozen(self):
        dev = Device.cpu()
        with pytest.raises((AttributeError, TypeError)):
            dev.index = 99  # type: ignore[misc]


# ------------------------------------------------------------------ #
# DevicePool                                                          #
# ------------------------------------------------------------------ #


class TestDevicePool:
    def _make_pool(self):
        devices = [
            Device(type=DeviceType.CUDA, index=0, name="GPU-0", total_memory_bytes=16 * 1024**3),
            Device(type=DeviceType.CUDA, index=1, name="GPU-1", total_memory_bytes=8 * 1024**3),
            Device.cpu(),
        ]
        return DevicePool(devices)

    def test_len(self):
        pool = self._make_pool()
        assert len(pool) == 3

    def test_iter(self):
        pool = self._make_pool()
        types = [d.type for d in pool]
        assert DeviceType.CUDA in types
        assert DeviceType.CPU in types

    def test_by_type(self):
        pool = self._make_pool()
        gpus = pool.by_type(DeviceType.CUDA)
        assert len(gpus) == 2

    def test_accelerators_sorted_by_memory(self):
        pool = self._make_pool()
        accs = pool.accelerators()
        assert accs[0].total_memory_bytes > accs[1].total_memory_bytes

    def test_best_accelerator(self):
        pool = self._make_pool()
        best = pool.best_accelerator()
        assert best.type == DeviceType.CUDA
        assert best.total_memory_gb == pytest.approx(16.0, 0.1)

    def test_best_accelerator_min_memory(self):
        pool = self._make_pool()
        # Require 12 GiB → only the 16 GiB GPU qualifies
        best = pool.best_accelerator(min_memory_gb=12.0)
        assert best.total_memory_gb == pytest.approx(16.0, 0.1)

    def test_best_accelerator_fallback_to_cpu(self):
        pool = DevicePool([Device.cpu()])
        dev = pool.best_accelerator()
        assert dev.type == DeviceType.CPU

    def test_from_device_strings(self):
        pool = DevicePool.from_device_strings(["cuda:0", "cpu"])
        assert len(pool) == 2
        assert pool.get("cuda:0") is not None
        assert pool.get("cpu") is not None

    def test_get_unknown(self):
        pool = self._make_pool()
        assert pool.get("xpu:0") is None

    def test_auto_always_has_cpu(self):
        pool = DevicePool.auto()
        cpu = pool.cpu_device()
        assert cpu.type == DeviceType.CPU


# ------------------------------------------------------------------ #
# HardwareScheduler                                                   #
# ------------------------------------------------------------------ #


class TestHardwareScheduler:
    def _make_scheduler(self):
        devices = [
            Device(type=DeviceType.CUDA, index=0, name="GPU-0", total_memory_bytes=16 * 1024**3),
            Device(type=DeviceType.CUDA, index=1, name="GPU-1", total_memory_bytes=8 * 1024**3),
            Device.cpu(),
        ]
        pool = DevicePool(devices)
        return HardwareScheduler(pool)

    def test_schedule_returns_device(self):
        sched = self._make_scheduler()
        dev = sched.schedule()
        assert dev is not None
        sched.release(dev)

    def test_schedule_prefers_accelerator(self):
        sched = self._make_scheduler()
        dev = sched.schedule(SchedulingHint())
        assert dev.is_accelerator
        sched.release(dev)

    def test_schedule_pinned_device(self):
        sched = self._make_scheduler()
        dev = sched.schedule(SchedulingHint(pinned_device="cpu"))
        assert dev.torch_id == "cpu"
        sched.release(dev)

    def test_schedule_min_memory(self):
        sched = self._make_scheduler()
        dev = sched.schedule(SchedulingHint(min_gpu_memory_gb=12.0))
        assert dev.total_memory_bytes == 16 * 1024**3
        sched.release(dev)

    def test_schedule_fallback_to_cpu(self):
        pool = DevicePool([Device.cpu()])
        sched = HardwareScheduler(pool)
        dev = sched.schedule(SchedulingHint(allow_cpu_fallback=True))
        assert dev.type == DeviceType.CPU
        sched.release(dev)

    def test_schedule_no_fallback_raises(self):
        pool = DevicePool([Device.cpu()])
        sched = HardwareScheduler(pool)
        with pytest.raises(RuntimeError):
            sched.schedule(SchedulingHint(allow_cpu_fallback=False))

    def test_least_loaded_scheduling(self):
        sched = self._make_scheduler()
        # Acquire GPU-0 twice → GPU-1 should be less loaded
        g0 = sched.schedule(SchedulingHint(pinned_device="cuda:0"))
        g0_again = sched.schedule(SchedulingHint(pinned_device="cuda:0"))
        # Now free-schedule should pick GPU-1
        dev = sched.schedule(SchedulingHint())
        assert dev.torch_id == "cuda:1"
        sched.release(g0)
        sched.release(g0_again)
        sched.release(dev)

    def test_stats_tracking(self):
        sched = self._make_scheduler()
        dev = sched.schedule()
        stats = sched.stats()
        assert stats[dev.torch_id].active_tasks == 1
        sched.release(dev)
        stats = sched.stats()
        assert stats[dev.torch_id].active_tasks == 0

    def test_pinned_unknown_device_raises(self):
        sched = self._make_scheduler()
        with pytest.raises(ValueError):
            sched.schedule(SchedulingHint(pinned_device="xpu:99"))
