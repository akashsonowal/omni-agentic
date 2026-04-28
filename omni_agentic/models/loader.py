"""ModelLoader: device-aware model loading with automatic dtype selection."""

from __future__ import annotations

import logging
from typing import Optional

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.hardware.pool import DevicePool
from omni_agentic.hardware.scheduler import HardwareScheduler, SchedulingHint
from omni_agentic.models.base import ModelBackend

logger = logging.getLogger(__name__)


class ModelLoader:
    """High-level helper that loads models onto the best available device.

    Usage::

        loader = ModelLoader.from_auto()
        backend = loader.load("Qwen/Qwen2-0.5B-Instruct", min_gpu_memory_gb=4)
        output = backend.generate("Hello, world!")
        loader.unload("Qwen/Qwen2-0.5B-Instruct")
    """

    def __init__(self, scheduler: HardwareScheduler) -> None:
        self._scheduler = scheduler
        self._backends: dict[str, ModelBackend] = {}

    # ------------------------------------------------------------------ #
    # Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_auto(cls) -> "ModelLoader":
        """Create a loader backed by an auto-detected device pool."""
        pool = DevicePool.auto()
        scheduler = HardwareScheduler(pool)
        return cls(scheduler)

    @classmethod
    def from_pool(cls, pool: DevicePool) -> "ModelLoader":
        """Create a loader from an explicit device pool."""
        return cls(HardwareScheduler(pool))

    # ------------------------------------------------------------------ #
    # Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(
        self,
        model_id: str,
        *,
        min_gpu_memory_gb: float = 0.0,
        preferred_device_type: Optional[DeviceType] = None,
        pinned_device: Optional[str] = None,
        allow_cpu_fallback: bool = True,
        **backend_kwargs: object,
    ) -> ModelBackend:
        """Load *model_id* onto the best available device and return the backend.

        If the model is already loaded, the cached backend is returned.
        """
        if model_id in self._backends:
            logger.debug("Cache hit for model %r", model_id)
            return self._backends[model_id]

        hint = SchedulingHint(
            min_gpu_memory_gb=min_gpu_memory_gb,
            preferred_type=preferred_device_type,
            allow_cpu_fallback=allow_cpu_fallback,
            pinned_device=pinned_device,
        )
        device = self._scheduler.schedule(hint)

        backend = self._create_backend(model_id, device, **backend_kwargs)
        backend.load()
        self._backends[model_id] = backend
        logger.info("Loaded %r on %s", model_id, device)
        return backend

    def unload(self, model_id: str) -> None:
        """Unload *model_id* and release device resources."""
        backend = self._backends.pop(model_id, None)
        if backend is None:
            return
        backend.unload()
        self._scheduler.release(backend.device)
        logger.info("Unloaded %r from %s", model_id, backend.device)

    def unload_all(self) -> None:
        """Unload every cached model."""
        for model_id in list(self._backends):
            self.unload(model_id)

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_backend(model_id: str, device: Device, **kwargs: object) -> ModelBackend:
        """Instantiate the appropriate backend for *model_id* and *device*."""
        try:
            from omni_agentic.models.hf_backend import HuggingFaceBackend

            return HuggingFaceBackend(model_id=model_id, device=device, **kwargs)
        except ImportError:
            raise RuntimeError(
                "No suitable model backend found. Install 'transformers' for "
                "HuggingFace model support."
            ) from None
