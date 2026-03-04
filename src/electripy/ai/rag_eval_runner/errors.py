"""Domain exceptions for the RAG evaluation runner."""

from __future__ import annotations


class RagEvalError(Exception):
    """Base exception for all RAG evaluation runner errors."""


class DatasetFormatError(RagEvalError):
    """Raised when a dataset JSONL file is malformed or inconsistent."""


class ExperimentConfigError(RagEvalError):
    """Raised when the experiment configuration is invalid or unsupported."""


class EvalRunnerError(RagEvalError):
    """Raised for unexpected errors during evaluation orchestration."""
