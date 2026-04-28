"""Pipeline sub-package: stage abstraction and heterogeneous pipeline executor."""

from omni_agentic.pipeline.pipeline import Pipeline
from omni_agentic.pipeline.stage import PipelineStage

__all__ = ["Pipeline", "PipelineStage"]
