from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    description: str = ""


@dataclass
class ResilienceConfig:
    # Pillars
    w_exposure: float = 1 / 3
    w_sensitivity: float = 1 / 3
    w_adaptive: float = 1 / 3

    # Exposure
    w_heat: float = 0.50
    w_drought: float = 0.35
    w_flood: float = 0.15

    # Sensitivity
    w_tourism_share: float = 0.50
    w_arrivals_index: float = 0.30
    w_income_vulnerability: float = 0.20

    # Adaptive capacity
    w_education: float = 0.34
    w_health: float = 0.33
    w_governance: float = 0.33


@dataclass
class ResilienceResult:
    region_id: str
    scenario_id: str
    df: pd.DataFrame

    # New name used in some newer code
    meta: Dict[str, Any] = field(default_factory=dict)

    # Old name used by your Streamlit pages
    # (store nothing here; just expose a compatible property)
    @property
    def config_used(self) -> Dict[str, Any]:
        return self.meta


# Optional backwards alias if any files import ResilienceSeries
ResilienceSeries = ResilienceResult
