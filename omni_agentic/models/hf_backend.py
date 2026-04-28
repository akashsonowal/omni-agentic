"""HuggingFace Transformers backend."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from omni_agentic.hardware.device import Device, DeviceType
from omni_agentic.models.base import ModelBackend

logger = logging.getLogger(__name__)


def _pick_torch_dtype(device: Device) -> "Any":
    """Return the preferred torch dtype for *device*."""
    import torch

    if device.type in (DeviceType.CUDA, DeviceType.ROCM):
        return torch.float16
    if device.type == DeviceType.MPS:
        return torch.float16
    return torch.float32  # CPU: float32 is safe


class HuggingFaceBackend(ModelBackend):
    """Inference backend backed by ``transformers`` AutoModelForCausalLM.

    Supports any model on the HuggingFace Hub that follows the causal-LM
    interface (GPT-2, LLaMA, Mistral, Qwen, …).
    """

    def __init__(
        self,
        model_id: str,
        device: Device,
        *,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__(model_id, device)
        self._trust_remote_code = trust_remote_code
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer for %r …", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=self._trust_remote_code,
        )

        torch_dtype = _pick_torch_dtype(self._device)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if self._load_in_8bit or self._load_in_4bit:
            # bitsandbytes quantisation requires CUDA
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=self._load_in_8bit,
                load_in_4bit=self._load_in_4bit,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self._device.torch_id

        logger.info("Loading model %r on %s …", self._model_id, self._device)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, **model_kwargs
        )
        self._model.eval()
        self._loaded = True
        logger.info("Model %r ready on %s", self._model_id, self._device)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        try:
            import torch

            if self._device.type in (DeviceType.CUDA, DeviceType.ROCM):
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Unloaded model %r", self._model_id)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

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
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device.torch_id)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature != 1.0 or top_p != 1.0,
            **kwargs,
        }

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Apply stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]

        return text

    def embed(self, text: str, **kwargs: Any) -> List[float]:
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import torch

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device.torch_id)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # Mean-pool the last hidden state
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        embedding = last_hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
        return embedding.float().cpu().tolist()
