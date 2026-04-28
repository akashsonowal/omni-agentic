# omni-agentic

**Agentic inference on heterogeneous hardware.**

`omni-agentic` lets you build AI agents and multi-stage inference pipelines that run across **any mix of hardware** — NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS), Intel Data-Center GPUs (XPU), and plain CPUs — without changing your agent logic.

---

## Features

| Layer | What it does |
|---|---|
| **Hardware abstraction** | Auto-detects every device on the machine and exposes a unified `Device` / `DevicePool` API |
| **Hardware scheduler** | Routes tasks to the least-loaded, best-fitting device (memory, type, or pinned) |
| **Model loader** | Loads HuggingFace causal-LM models onto the scheduled device with automatic dtype selection |
| **Agent framework** | ReAct-style agent with tool-calling, short-term message history, and long-term key-value memory |
| **Heterogeneous pipeline** | Sequential pipelines where each stage can run on a different device |

---

## Installation

```bash
pip install -e ".[dev]"
```

> **Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, `transformers` ≥ 4.38, `accelerate` ≥ 0.27, `pydantic` ≥ 2.0.

---

## Quick start

### 1 — Auto-detect hardware

```python
from omni_agentic.hardware.pool import DevicePool

pool = DevicePool.auto()
print(pool)
# DevicePool([Device(cuda:0 – NVIDIA A100 (80.0 GiB)), Device(cpu – x86_64)])

best = pool.best_accelerator(min_memory_gb=40)
print(best.torch_id)   # "cuda:0"
```

### 2 — Load a model on the best device

```python
from omni_agentic.models.loader import ModelLoader

loader = ModelLoader.from_auto()
backend = loader.load("Qwen/Qwen2-0.5B-Instruct")    # placed on best GPU automatically

response = backend.generate("Explain heterogeneous computing in one sentence.")
print(response)

loader.unload_all()
```

### 3 — Run an agent with tools

```python
from omni_agentic import Agent, tool
from omni_agentic.models.loader import ModelLoader

loader = ModelLoader.from_auto()
backend = loader.load("Qwen/Qwen2-0.5B-Instruct")

@tool(description="Calculate the square root of a number")
def sqrt(x: float) -> float:
    import math
    return math.sqrt(float(x))

agent = Agent(backend=backend)
agent.register_tool(sqrt)

answer = agent.run("What is the square root of 144?")
print(answer)   # "The square root of 144 is 12.0"
```

### 4 — Heterogeneous pipeline

```python
from omni_agentic.hardware.device import DeviceType
from omni_agentic.hardware.pool import DevicePool
from omni_agentic.pipeline.pipeline import Pipeline
from omni_agentic.pipeline.stage import PipelineStage, StageConfig

pool = DevicePool.auto()
pipeline = Pipeline(pool=pool)

# Pre-processing on CPU
pipeline.add_stage(PipelineStage(
    "preprocess",
    fn=lambda text, device: text.strip().lower(),
    config=StageConfig(preferred_device_type=DeviceType.CPU),
))

# Heavy inference on GPU
pipeline.add_stage(PipelineStage(
    "inference",
    fn=my_inference_fn,
    config=StageConfig(min_gpu_memory_gb=8),
))

# Post-processing on CPU
pipeline.add_stage(PipelineStage(
    "postprocess",
    fn=lambda result, device: result.strip(),
    config=StageConfig(preferred_device_type=DeviceType.CPU),
))

result = pipeline.run("  Hello, World!  ")
print(result.output)
print(result.stage_timings)   # {"preprocess": 0.0001, "inference": 0.42, "postprocess": 0.0001}
```

---

## Architecture

```
omni_agentic/
├── hardware/
│   ├── device.py      # Device dataclass + DeviceType enum
│   ├── pool.py        # DevicePool – discovers all devices
│   └── scheduler.py   # HardwareScheduler – routes tasks
├── models/
│   ├── base.py        # ModelBackend ABC
│   ├── hf_backend.py  # HuggingFace Transformers backend
│   ├── loader.py      # ModelLoader – device-aware loading
│   └── registry.py    # ModelRegistry – backend catalogue
├── agents/
│   ├── base.py        # Agent – ReAct loop
│   ├── memory.py      # ShortTermMemory + LongTermMemory
│   └── tools.py       # Tool, @tool decorator, ToolRegistry
└── pipeline/
    ├── pipeline.py    # Pipeline – heterogeneous stage runner
    └── stage.py       # PipelineStage + StageConfig
```

---

## Running the tests

```bash
pytest -v
```

---

## Supported hardware

| Device | Torch device string | Notes |
|---|---|---|
| CPU | `cpu` | Always available |
| NVIDIA GPU | `cuda:N` | Requires CUDA drivers + `torch` built with CUDA |
| AMD GPU | `cuda:N` | PyTorch ROCm reuses the `cuda` device string |
| Apple Silicon | `mps` | macOS 12.3+ with PyTorch ≥ 2.0 |
| Intel GPU | `xpu:N` | Requires Intel Extension for PyTorch |

---

## License

MIT
