# src/crisi_core/__init__.py
from .models import Scenario, ResilienceConfig, ResilienceResult

# Canonical defaults live in config; `scenarios.py` exists only as a compatibility alias.
from .config import DEFAULT_SCENARIOS, DEFAULT_CONFIG

from .scoring import compute_resilience_series

# CBA exports (support alternate naming)
try:
    from .cba import ProjectInputs, evaluate_project_cba
except Exception:  # pragma: no cover
    # fallback aliases if your cba.py uses different names
    from .cba import ProjectCBAInputs as ProjectInputs  # type: ignore
    from .cba import evaluate_project_cba  # type: ignore

__all__ = [
    "Scenario",
    "ResilienceConfig",
    "ResilienceResult",
    "DEFAULT_SCENARIOS",
    "DEFAULT_CONFIG",
    "compute_resilience_series",
    "ProjectInputs",
    "evaluate_project_cba",
]
