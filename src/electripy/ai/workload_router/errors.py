"""Exception hierarchy for the workload routing engine.

All routing exceptions extend :class:`WorkloadRouterError` which itself
derives from :class:`ElectriPyError`, keeping the error hierarchy
consistent with the rest of the ElectriPy codebase.

Example:
    from electripy.ai.workload_router.errors import NoCandidateError

    try:
        decision = service.route(request)
    except NoCandidateError as exc:
        print(f"No model available: {exc}")
"""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "BudgetExceededError",
    "ConstraintViolationError",
    "NoCandidateError",
    "WorkloadRouterError",
]


class WorkloadRouterError(ElectriPyError):
    """Base exception for workload routing failures."""


class NoCandidateError(WorkloadRouterError):
    """Raised when no candidate model satisfies the routing policy."""


class ConstraintViolationError(WorkloadRouterError):
    """Raised when a required constraint cannot be satisfied."""


class BudgetExceededError(WorkloadRouterError):
    """Raised when all candidates exceed the budget ceiling."""
