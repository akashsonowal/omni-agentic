"""Tests for the heterogeneous pipeline execution."""

from __future__ import annotations

import pytest

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.hardware.pool import DevicePool
from omni_agentic.pipeline.pipeline import Pipeline, PipelineResult
from omni_agentic.pipeline.stage import PipelineStage, StageConfig


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _make_pool(*device_types: DeviceType) -> DevicePool:
    devices = []
    cuda_idx = 0
    for dt in device_types:
        if dt == DeviceType.CPU:
            devices.append(Device.cpu())
        elif dt == DeviceType.CUDA:
            devices.append(
                Device(
                    type=DeviceType.CUDA,
                    index=cuda_idx,
                    name=f"GPU-{cuda_idx}",
                    total_memory_bytes=16 * 1024**3,
                )
            )
            cuda_idx += 1
        else:
            devices.append(Device(type=dt, index=0))
    return DevicePool(devices)


# ------------------------------------------------------------------ #
# StageConfig                                                         #
# ------------------------------------------------------------------ #


class TestStageConfig:
    def test_as_scheduling_hint(self):
        cfg = StageConfig(min_gpu_memory_gb=8.0, pinned_device="cpu")
        hint = cfg.as_scheduling_hint()
        assert hint.min_gpu_memory_gb == 8.0
        assert hint.pinned_device == "cpu"


# ------------------------------------------------------------------ #
# PipelineStage                                                       #
# ------------------------------------------------------------------ #


class TestPipelineStage:
    def test_unassigned_raises(self):
        stage = PipelineStage("test", fn=lambda data, device: data)
        with pytest.raises(RuntimeError, match="no assigned device"):
            stage("hello")

    def test_execute_with_device(self):
        cpu = Device.cpu()

        def fn(data, device):
            return f"processed:{data}:{device.type.value}"

        stage = PipelineStage("test", fn=fn)
        stage.assigned_device = cpu
        result = stage("input")
        assert result == "processed:input:cpu"

    def test_repr(self):
        stage = PipelineStage("myStage", fn=lambda d, device: d)
        assert "myStage" in repr(stage)
        assert "unassigned" in repr(stage)


# ------------------------------------------------------------------ #
# Pipeline                                                            #
# ------------------------------------------------------------------ #


class TestPipeline:
    def _make_pipeline(self, *device_types: DeviceType) -> Pipeline:
        pool = _make_pool(*device_types)
        return Pipeline(pool=pool)

    def test_empty_pipeline_returns_input(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        result = pipeline.run("hello")
        assert result.ok
        assert result.output == "hello"

    def test_single_stage(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        pipeline.add_stage(
            PipelineStage("upper", fn=lambda data, device: data.upper())
        )
        result = pipeline.run("hello")
        assert result.ok
        assert result.output == "HELLO"

    def test_multi_stage_chaining(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        pipeline.add_stage(PipelineStage("step1", fn=lambda d, device: d + " world"))
        pipeline.add_stage(PipelineStage("step2", fn=lambda d, device: d.upper()))
        result = pipeline.run("hello")
        assert result.ok
        assert result.output == "HELLO WORLD"

    def test_stage_timings_recorded(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        pipeline.add_stage(PipelineStage("step1", fn=lambda d, device: d))
        pipeline.add_stage(PipelineStage("step2", fn=lambda d, device: d))
        result = pipeline.run("x")
        assert "step1" in result.stage_timings
        assert "step2" in result.stage_timings
        assert result.total_time >= 0

    def test_builder_returns_self(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        ret = pipeline.add_stage(PipelineStage("s", fn=lambda d, device: d))
        assert ret is pipeline

    def test_len(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        assert len(pipeline) == 0
        pipeline.add_stage(PipelineStage("s", fn=lambda d, device: d))
        assert len(pipeline) == 1

    def test_stage_error_captured(self):
        pipeline = self._make_pipeline(DeviceType.CPU)

        def failing(data, device):
            raise ValueError("deliberate failure")

        pipeline.add_stage(PipelineStage("boom", fn=failing))
        result = pipeline.run("x")
        assert not result.ok
        assert "deliberate failure" in result.error

    def test_stage_assigned_to_device(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        stage = PipelineStage("s", fn=lambda d, device: d)
        pipeline.add_stage(stage)
        pipeline.run("x")
        assert stage.assigned_device is not None

    def test_pinned_device_stage(self):
        pipeline = self._make_pipeline(DeviceType.CUDA, DeviceType.CPU)
        stage = PipelineStage(
            "pinned",
            fn=lambda d, device: device.torch_id,
            config=StageConfig(pinned_device="cpu"),
        )
        pipeline.add_stage(stage)
        result = pipeline.run("x")
        assert result.ok
        assert result.output == "cpu"

    def test_describe(self):
        pipeline = self._make_pipeline(DeviceType.CPU)
        pipeline.add_stage(PipelineStage("s", fn=lambda d, device: d))
        pipeline.run("x")
        desc = pipeline.describe()
        assert len(desc) == 1
        assert desc[0]["stage"] == "s"

    def test_multi_gpu_pipeline(self):
        """Two stages can be assigned to different GPUs."""
        pipeline = self._make_pipeline(DeviceType.CUDA, DeviceType.CUDA, DeviceType.CPU)

        seen_devices = []

        def record(data, device):
            seen_devices.append(device.torch_id)
            return data

        pipeline.add_stage(PipelineStage("s1", fn=record))
        pipeline.add_stage(PipelineStage("s2", fn=record))
        result = pipeline.run("x")
        assert result.ok
        # Both stages ran
        assert len(seen_devices) == 2
