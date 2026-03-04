"""RAG evaluation runner component.

High-level exports for dataset models, experiment configuration, and
orchestration services.
"""

from __future__ import annotations

from .domain import CorpusRecord, ExperimentConfig, QueryRecord
from .errors import DatasetFormatError, EvalRunnerError, ExperimentConfigError, RagEvalError
from .services import DatasetLoader, Evaluator, IndexBuilder, ReportWriter

__all__ = [
    "CorpusRecord",
    "QueryRecord",
    "ExperimentConfig",
    "DatasetLoader",
    "IndexBuilder",
    "Evaluator",
    "ReportWriter",
    "RagEvalError",
    "DatasetFormatError",
    "ExperimentConfigError",
    "EvalRunnerError",
]
