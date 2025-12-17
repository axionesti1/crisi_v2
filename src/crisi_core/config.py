from __future__ import annotations
from typing import Dict
from .models import Scenario, ResilienceConfig


DEFAULT_SCENARIOS: Dict[str, Scenario] = {
    "green_global_resilience": Scenario(
        id="green_global_resilience",
        name="Green Global Resilience",
        description=(
            "Optimistic pathway with strong climate policy, high adaptation investments, "
            "and steady tourism growth. (RCP4.5 / SSP1)"
        ),
    ),
    "business_as_usual": Scenario(
        id="business_as_usual",
        name="Business-as-Usual Drift",
        description=(
            "Continuation of current trends with moderate policy and adaptation efforts; "
            "tourism growth slows over time. (RCP6.0 / SSP2)"
        ),
    ),
    "divided_disparity": Scenario(
        id="divided_disparity",
        name="Divided Disparity",
        description=(
            "A world of widening inequality; high technology for elites but low adaptation "
            "elsewhere; two-tier tourism system. (SSP4 / RCP6.0-like)"
        ),
    ),
    "techno_optimism": Scenario(
        id="techno_optimism",
        name="Techno-Optimism on a Hot Planet",
        description=(
            "High-growth world powered by fossil fuels and advanced tech; minimal "
            "mitigation triggers severe warming. (RCP8.5 / SSP5)"
        ),
    ),
    "regional_fortress": Scenario(
        id="regional_fortress",
        name="Regional Fortress World",
        description=(
            "World fragments into self-reliant blocs; little cooperation; tourism declines "
            "after the mid-2030s. (RCP7.0 / SSP3)"
        ),
    ),
}

DEFAULT_CONFIG = ResilienceConfig()
