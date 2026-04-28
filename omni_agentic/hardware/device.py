"""Device abstraction: enumeration of device types and a Device dataclass."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional


class DeviceType(str, Enum):
    """Supported hardware device families."""

    CPU = "cpu"
    CUDA = "cuda"      # NVIDIA GPU via CUDA
    MPS = "mps"        # Apple Silicon GPU via Metal Performance Shaders
    ROCM = "rocm"      # AMD GPU via ROCm (torch device string is still "cuda")
    XPU = "xpu"        # Intel Data-Center GPU (oneAPI)

    def torch_device_prefix(self) -> str:
        """Return the torch device string prefix for this device type."""
        if self == DeviceType.ROCM:
            return "cuda"  # PyTorch ROCm reuses the "cuda" device string
        return self.value


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """Represents a single compute device."""

    type: DeviceType
    index: int = 0
    name: str = ""
    total_memory_bytes: int = 0

    # ------------------------------------------------------------------ #
    # Convenience constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def cpu(cls) -> "Device":
        """Return the default CPU device."""
        import platform

        return cls(type=DeviceType.CPU, index=0, name=platform.processor() or "cpu")

    @classmethod
    def cuda(cls, index: int = 0) -> "Device":
        """Return a CUDA device, optionally verifying it exists."""
        try:
            import torch

            props = torch.cuda.get_device_properties(index)
            return cls(
                type=DeviceType.CUDA,
                index=index,
                name=props.name,
                total_memory_bytes=props.total_memory,
            )
        except Exception:
            return cls(type=DeviceType.CUDA, index=index, name=f"cuda:{index}")

    @classmethod
    def mps(cls) -> "Device":
        """Return the Apple MPS device."""
        return cls(type=DeviceType.MPS, index=0, name="Apple Metal Performance Shaders")

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def torch_id(self) -> str:
        """Return the torch device string (e.g. ``'cuda:0'``, ``'cpu'``)."""
        prefix = self.type.torch_device_prefix()
        if self.type == DeviceType.CPU or self.type == DeviceType.MPS:
            return prefix
        return f"{prefix}:{self.index}"

    @property
    def is_accelerator(self) -> bool:
        """True for all non-CPU devices."""
        return self.type != DeviceType.CPU

    @property
    def total_memory_gb(self) -> Optional[float]:
        """Total memory in GiB, or *None* for CPU / unknown."""
        if self.total_memory_bytes == 0:
            return None
        return self.total_memory_bytes / (1024**3)

    def __str__(self) -> str:
        mem = f" ({self.total_memory_gb:.1f} GiB)" if self.total_memory_gb else ""
        name = f" – {self.name}" if self.name else ""
        return f"Device({self.torch_id}{name}{mem})"
