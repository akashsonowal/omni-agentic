"""DevicePool: discover and manage all available compute devices."""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

from omni_agentic.hardware.device import Device, DeviceType

logger = logging.getLogger(__name__)


class DevicePool:
    """Discovers all available hardware and exposes them as :class:`Device` objects.

    Usage::

        pool = DevicePool.auto()
        gpu = pool.best_accelerator()   # first GPU, or CPU fallback
        for dev in pool:
            print(dev)
    """

    def __init__(self, devices: List[Device]) -> None:
        self._devices = list(devices)

    # ------------------------------------------------------------------ #
    # Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def auto(cls) -> "DevicePool":
        """Auto-detect all available devices on this machine."""
        devices: List[Device] = []

        # --- CUDA / ROCm ---------------------------------------------------
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    # Heuristically detect ROCm from the device name
                    dev_type = DeviceType.ROCM if "AMD" in props.name else DeviceType.CUDA
                    devices.append(
                        Device(
                            type=dev_type,
                            index=i,
                            name=props.name,
                            total_memory_bytes=props.total_memory,
                        )
                    )
                    logger.debug("Discovered %s device: %s", dev_type.value, props.name)
        except ImportError:
            logger.debug("torch not installed – skipping CUDA/ROCm detection")

        # --- MPS (Apple Silicon) -------------------------------------------
        try:
            import torch

            if torch.backends.mps.is_available():
                devices.append(Device.mps())
                logger.debug("Discovered MPS device")
        except (ImportError, AttributeError):
            pass

        # --- XPU (Intel) ---------------------------------------------------
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                xpu_count = torch.xpu.device_count()
                for i in range(xpu_count):
                    devices.append(Device(type=DeviceType.XPU, index=i, name=f"Intel XPU {i}"))
                    logger.debug("Discovered XPU device %d", i)
        except (ImportError, AttributeError):
            pass

        # --- CPU (always present) ------------------------------------------
        devices.append(Device.cpu())

        return cls(devices)

    @classmethod
    def from_device_strings(cls, device_strings: List[str]) -> "DevicePool":
        """Build a pool from explicit torch device strings such as ``['cuda:0', 'cpu']``."""
        devices: List[Device] = []
        for ds in device_strings:
            parts = ds.split(":")
            type_str = parts[0].lower()
            index = int(parts[1]) if len(parts) > 1 else 0
            try:
                dev_type = DeviceType(type_str)
            except ValueError:
                logger.warning("Unknown device type %r – skipping", type_str)
                continue
            devices.append(Device(type=dev_type, index=index))
        return cls(devices)

    # ------------------------------------------------------------------ #
    # Query helpers                                                       #
    # ------------------------------------------------------------------ #

    def all_devices(self) -> List[Device]:
        """Return all devices in this pool."""
        return list(self._devices)

    def by_type(self, device_type: DeviceType) -> List[Device]:
        """Return all devices of a given type."""
        return [d for d in self._devices if d.type == device_type]

    def accelerators(self) -> List[Device]:
        """Return all non-CPU devices, sorted largest-memory-first."""
        accs = [d for d in self._devices if d.is_accelerator]
        return sorted(accs, key=lambda d: d.total_memory_bytes, reverse=True)

    def best_accelerator(self, min_memory_gb: float = 0.0) -> Device:
        """Return the highest-memory accelerator that meets *min_memory_gb*.

        Falls back to CPU if no qualifying accelerator is found.
        """
        for dev in self.accelerators():
            mem = dev.total_memory_gb or float("inf")
            if mem >= min_memory_gb:
                return dev
        return self.cpu_device()

    def cpu_device(self) -> Device:
        """Return the CPU device."""
        cpus = self.by_type(DeviceType.CPU)
        return cpus[0] if cpus else Device.cpu()

    def get(self, torch_id: str) -> Optional[Device]:
        """Lookup a device by its torch device string (e.g. ``'cuda:0'``)."""
        for dev in self._devices:
            if dev.torch_id == torch_id:
                return dev
        return None

    # ------------------------------------------------------------------ #
    # Dunder methods                                                      #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Device]:
        return iter(self._devices)

    def __len__(self) -> int:
        return len(self._devices)

    def __repr__(self) -> str:
        return f"DevicePool([{', '.join(str(d) for d in self._devices)}])"
