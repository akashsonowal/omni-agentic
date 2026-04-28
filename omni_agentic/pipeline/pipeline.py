"""Heterogeneous inference pipeline: chain stages across multiple devices."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omni_agentic.hardware.pool import DevicePool
from omni_agentic.hardware.scheduler import HardwareScheduler
from omni_agentic.pipeline.stage import PipelineStage

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """The outcome of running a full pipeline."""

    output: Any
    stage_timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_time(self) -> float:
        return sum(self.stage_timings.values())


class Pipeline:
    """A sequential pipeline of :class:`~omni_agentic.pipeline.stage.PipelineStage`
    objects that can be spread across heterogeneous hardware.

    Device assignment happens at *run time* (lazy) so the same pipeline
    definition can be reused across machines with different hardware.

    Usage::

        pool = DevicePool.auto()
        pipeline = Pipeline(pool=pool)

        pipeline.add_stage(PipelineStage("preprocess", preprocess_fn))
        pipeline.add_stage(
            PipelineStage(
                "inference",
                inference_fn,
                config=StageConfig(min_gpu_memory_gb=8),
            )
        )
        pipeline.add_stage(PipelineStage("postprocess", postprocess_fn))

        result = pipeline.run("Hello world")
        print(result.output)
    """

    def __init__(self, pool: Optional[DevicePool] = None) -> None:
        self._pool = pool or DevicePool.auto()
        self._scheduler = HardwareScheduler(self._pool)
        self._stages: List[PipelineStage] = []

    # ------------------------------------------------------------------ #
    # Building the pipeline                                               #
    # ------------------------------------------------------------------ #

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Append *stage* to the pipeline (builder-style, returns self)."""
        self._stages.append(stage)
        return self

    def __len__(self) -> int:
        return len(self._stages)

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    def run(self, input_data: Any) -> PipelineResult:
        """Execute all stages sequentially.

        Each stage receives the output of the previous stage (or *input_data*
        for the first stage).  Device assignment is resolved just-in-time via
        the :class:`~omni_agentic.hardware.scheduler.HardwareScheduler`.
        """
        if not self._stages:
            return PipelineResult(output=input_data)

        timings: Dict[str, float] = {}
        data = input_data
        acquired_devices = []

        try:
            for stage in self._stages:
                # Assign a device if not yet assigned
                if stage.assigned_device is None:
                    hint = stage.config.as_scheduling_hint()
                    device = self._scheduler.schedule(hint)
                    stage.assigned_device = device
                    acquired_devices.append((stage, device))

                t0 = time.perf_counter()
                try:
                    data = stage(data)
                except Exception as exc:
                    logger.error("Stage %r failed: %s", stage.name, exc)
                    return PipelineResult(
                        output=None,
                        stage_timings=timings,
                        error=f"Stage {stage.name!r} raised: {exc}",
                    )
                finally:
                    timings[stage.name] = time.perf_counter() - t0

        finally:
            # Release all acquired device slots
            for stage, device in acquired_devices:
                self._scheduler.release(device)

        return PipelineResult(output=data, stage_timings=timings)

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #

    def describe(self) -> List[Dict[str, str]]:
        """Return a human-readable description of each stage."""
        result = []
        for stage in self._stages:
            dev = stage.assigned_device.torch_id if stage.assigned_device else "unassigned"
            result.append({"stage": stage.name, "device": dev})
        return result

    def __repr__(self) -> str:
        return f"Pipeline(stages={[s.name for s in self._stages]}, pool={self._pool!r})"
