"""PipelineStage: a single, potentially device-pinned unit of work."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.hardware.scheduler import SchedulingHint

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Hardware placement preferences for a pipeline stage."""

    # Minimum GPU memory required to run on an accelerator.
    min_gpu_memory_gb: float = 0.0
    # Force a particular device type (None = any).
    preferred_device_type: Optional[DeviceType] = None
    # Explicit torch device string, overrides all other hints.
    pinned_device: Optional[str] = None
    # Whether this stage may run on CPU when no GPU is available.
    allow_cpu_fallback: bool = True

    def as_scheduling_hint(self) -> SchedulingHint:
        return SchedulingHint(
            min_gpu_memory_gb=self.min_gpu_memory_gb,
            preferred_type=self.preferred_device_type,
            allow_cpu_fallback=self.allow_cpu_fallback,
            pinned_device=self.pinned_device,
        )


class PipelineStage:
    """A named, callable stage in a heterogeneous inference pipeline.

    Each stage receives the output of the previous stage (or the initial
    *input* for the first stage) and returns a value that is forwarded to
    the next stage.

    Usage::

        def tokenize(text: str, device: Device) -> dict:
            ...

        stage = PipelineStage(
            name="tokenizer",
            fn=tokenize,
            config=StageConfig(preferred_device_type=DeviceType.CPU),
        )
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        config: Optional[StageConfig] = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.config = config or StageConfig()
        self._assigned_device: Optional[Device] = None

    # ------------------------------------------------------------------ #
    # Device assignment                                                   #
    # ------------------------------------------------------------------ #

    @property
    def assigned_device(self) -> Optional[Device]:
        return self._assigned_device

    @assigned_device.setter
    def assigned_device(self, dev: Device) -> None:
        self._assigned_device = dev

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    def __call__(self, data: Any) -> Any:
        """Execute this stage, passing the current *device* as a keyword arg."""
        if self._assigned_device is None:
            raise RuntimeError(
                f"Stage {self.name!r} has no assigned device. "
                "Add it to a Pipeline before calling."
            )
        logger.debug("Stage %r executing on %s", self.name, self._assigned_device)
        return self.fn(data, device=self._assigned_device)

    def __repr__(self) -> str:
        dev = str(self._assigned_device) if self._assigned_device else "unassigned"
        return f"PipelineStage(name={self.name!r}, device={dev})"
