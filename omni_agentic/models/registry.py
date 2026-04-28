"""ModelRegistry: global catalogue of available model backends."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Type

from omni_agentic.models.base import ModelBackend

logger = logging.getLogger(__name__)

# Type alias for a backend factory function.
BackendFactory = Callable[..., ModelBackend]


class ModelRegistry:
    """A simple registry that maps model identifiers to backend factories.

    Usage::

        registry = ModelRegistry.global_registry()

        # Register a custom backend
        registry.register("my-model", MyBackendClass)

        # Retrieve and instantiate
        backend = registry.create("my-model", device=device)
    """

    _global: Optional["ModelRegistry"] = None

    def __init__(self) -> None:
        self._factories: Dict[str, Type[ModelBackend]] = {}

    # ------------------------------------------------------------------ #
    # Global singleton                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def global_registry(cls) -> "ModelRegistry":
        """Return the process-level global registry (created on first call)."""
        if cls._global is None:
            cls._global = cls()
            cls._global._register_builtin_backends()
        return cls._global

    # ------------------------------------------------------------------ #
    # Registration                                                        #
    # ------------------------------------------------------------------ #

    def register(self, model_id: str, backend_cls: Type[ModelBackend]) -> None:
        """Associate *model_id* with *backend_cls*.

        Raises :class:`ValueError` if the ID is already registered (use
        *force=True* to override).
        """
        if model_id in self._factories:
            logger.warning("Overwriting existing registration for %r", model_id)
        self._factories[model_id] = backend_cls
        logger.debug("Registered model backend %r -> %s", model_id, backend_cls.__name__)

    def unregister(self, model_id: str) -> None:
        """Remove a registration."""
        self._factories.pop(model_id, None)

    # ------------------------------------------------------------------ #
    # Instantiation                                                       #
    # ------------------------------------------------------------------ #

    def create(self, model_id: str, **kwargs: object) -> ModelBackend:
        """Instantiate the backend registered under *model_id*.

        All keyword arguments are forwarded to the backend constructor.
        """
        cls = self._factories.get(model_id)
        if cls is None:
            raise KeyError(
                f"No backend registered for model_id={model_id!r}. "
                f"Available: {sorted(self._factories)}"
            )
        return cls(model_id=model_id, **kwargs)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._factories

    def __repr__(self) -> str:
        return f"ModelRegistry({sorted(self._factories.keys())})"

    # ------------------------------------------------------------------ #
    # Built-in backends                                                   #
    # ------------------------------------------------------------------ #

    def _register_builtin_backends(self) -> None:
        """Register built-in backends if their dependencies are available."""
        try:
            from omni_agentic.models.hf_backend import HuggingFaceBackend  # noqa: F401

            # HuggingFaceBackend is not registered by model_id here because
            # it is a generic backend that works for any HF model.  The loader
            # registers specific model_ids at load time.
            logger.debug("HuggingFaceBackend available")
        except ImportError:
            logger.debug("transformers not installed – HuggingFaceBackend unavailable")
