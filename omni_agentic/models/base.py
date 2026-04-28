"""Abstract base class for all model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omni_agentic.hardware.device import Device


class ModelBackend(ABC):
    """Common interface every model backend must implement.

    Sub-classes wrap a specific inference engine (HuggingFace Transformers,
    llama.cpp, vLLM, ONNX Runtime, etc.) and hide hardware-specific details.
    """

    def __init__(self, model_id: str, device: Device) -> None:
        self._model_id = model_id
        self._device = device
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> Device:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self) -> None:
        """Load the model weights onto the target device."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free device memory."""

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text continuation for *prompt*.

        Returns the **newly generated** text only (not including the prompt).
        """

    @abstractmethod
    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """Return a dense embedding vector for *text*."""

    # ------------------------------------------------------------------ #
    # Context manager                                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ModelBackend":
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"{self.__class__.__name__}(model_id={self._model_id!r}, device={self._device}, status={status})"
