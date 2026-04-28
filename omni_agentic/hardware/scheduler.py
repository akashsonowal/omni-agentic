"""HardwareScheduler: route inference tasks to the most suitable device."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.hardware.pool import DevicePool

logger = logging.getLogger(__name__)


@dataclass
class SchedulingHint:
    """Hints that influence device selection for a single inference task."""

    # Minimum GPU memory (GiB) required; 0 means no preference.
    min_gpu_memory_gb: float = 0.0
    # Preferred device type; None means no preference.
    preferred_type: Optional[DeviceType] = None
    # Whether the task can fall back to CPU if no GPU is available.
    allow_cpu_fallback: bool = True
    # Explicit torch device string to pin execution (overrides everything else).
    pinned_device: Optional[str] = None


@dataclass
class _DeviceStats:
    """Lightweight runtime statistics tracked per device."""

    active_tasks: int = 0
    total_tasks: int = 0


class HardwareScheduler:
    """Schedules inference tasks across the devices in a :class:`DevicePool`.

    The scheduler implements a simple policy:

    1. If a task has a *pinned_device*, use it directly.
    2. Otherwise prefer the device type indicated by *preferred_type*.
    3. Among qualifying devices, choose the one with the fewest active tasks
       (least-loaded).
    4. Fall back to CPU if no accelerator is available and *allow_cpu_fallback*
       is True.

    Usage::

        pool = DevicePool.auto()
        scheduler = HardwareScheduler(pool)

        device = scheduler.schedule(SchedulingHint(min_gpu_memory_gb=8))
        try:
            run_inference(device)
        finally:
            scheduler.release(device)
    """

    def __init__(self, pool: DevicePool) -> None:
        self._pool = pool
        self._stats: Dict[str, _DeviceStats] = {
            dev.torch_id: _DeviceStats() for dev in pool
        }

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def schedule(self, hint: Optional[SchedulingHint] = None) -> Device:
        """Select the best device for a task described by *hint*.

        Increments the active-task counter for the chosen device.
        Always call :meth:`release` when the task finishes.
        """
        hint = hint or SchedulingHint()

        # Pinned device – honour exactly.
        if hint.pinned_device:
            dev = self._pool.get(hint.pinned_device)
            if dev is None:
                raise ValueError(
                    f"Pinned device {hint.pinned_device!r} is not in the device pool"
                )
            self._acquire(dev)
            return dev

        candidates = self._build_candidates(hint)
        if not candidates:
            if hint.allow_cpu_fallback:
                dev = self._pool.cpu_device()
                logger.warning(
                    "No suitable accelerator found – falling back to %s", dev
                )
                self._acquire(dev)
                return dev
            raise RuntimeError(
                "No suitable device found for the given scheduling hint and CPU "
                "fallback is disabled."
            )

        # Least-loaded first, then largest memory (stable sort preserves order).
        best = min(
            candidates,
            key=lambda d: (
                self._stats[d.torch_id].active_tasks,
                -(d.total_memory_bytes or 0),
            ),
        )
        self._acquire(best)
        logger.debug("Scheduled task on %s", best)
        return best

    def release(self, device: Device) -> None:
        """Signal that a previously scheduled task has completed."""
        stats = self._stats.get(device.torch_id)
        if stats is None:
            return
        stats.active_tasks = max(0, stats.active_tasks - 1)

    def stats(self) -> Dict[str, _DeviceStats]:
        """Return a snapshot of per-device statistics."""
        return {k: _DeviceStats(v.active_tasks, v.total_tasks) for k, v in self._stats.items()}

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _acquire(self, dev: Device) -> None:
        stats = self._stats.setdefault(dev.torch_id, _DeviceStats())
        stats.active_tasks += 1
        stats.total_tasks += 1

    def _build_candidates(self, hint: SchedulingHint) -> List[Device]:
        """Return devices that satisfy memory and type constraints."""
        candidates: List[Device] = []
        for dev in self._pool:
            if not dev.is_accelerator:
                continue
            # Filter by preferred type
            if hint.preferred_type and dev.type != hint.preferred_type:
                continue
            # Filter by minimum GPU memory
            mem = dev.total_memory_gb or float("inf")
            if mem < hint.min_gpu_memory_gb:
                continue
            candidates.append(dev)
        return candidates
