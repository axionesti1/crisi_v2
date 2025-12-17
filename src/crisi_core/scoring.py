from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .models import ResilienceConfig, ResilienceResult
from .config import DEFAULT_CONFIG
from .data_io import load_hazard_series, load_socio_series, load_flood_risk


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_weights(values: Dict[str, float]) -> Dict[str, float]:
    vals = {k: max(0.0, float(v)) for k, v in values.items()}
    s = float(sum(vals.values()))
    if s <= 0.0:
        n = len(vals)
        return {k: 1.0 / n for k in vals}
    return {k: v / s for k, v in vals.items()}


def _merge_cfg_and_override(cfg: ResilienceConfig, weights_override: Optional[Dict[str, float]]) -> ResilienceConfig:
    if not weights_override:
        return cfg
    base = asdict(cfg)
    for k, v in weights_override.items():
        if k in base:
            base[k] = float(v)
    return ResilienceConfig(**base)


def _scenario_params(scenario_id: str) -> Dict[str, float]:
    """
    Conservative scenario-to-trend mapping applied over the horizon (0..1 time scale).
    The goal is: sensitivity + adaptive are NOT flat and differ by scenario and region.
    """
    if scenario_id == "green_global_resilience":
        return dict(tourism=-0.18, arrivals=-0.05, gdp=+0.18, edu=+0.10, health=+0.10, gov=+0.12)
    if scenario_id == "business_as_usual":
        return dict(tourism=-0.04, arrivals=+0.03, gdp=+0.08, edu=+0.04, health=+0.03, gov=+0.02)
    if scenario_id == "divided_disparity":
        return dict(tourism=+0.08, arrivals=+0.10, gdp=+0.02, edu=+0.01, health=+0.01, gov=-0.03)
    if scenario_id == "techno_optimism":
        return dict(tourism=+0.10, arrivals=+0.12, gdp=+0.14, edu=+0.03, health=+0.03, gov=-0.02)
    if scenario_id == "regional_fortress":
        return dict(tourism=+0.05, arrivals=+0.20, gdp=-0.04, edu=-0.02, health=-0.02, gov=-0.06)
    return dict(tourism=0.0, arrivals=0.0, gdp=0.0, edu=0.0, health=0.0, gov=0.0)


def _wildcard_shock(year: int, wildcard: Dict[str, Any], year_min: int, year_max: int) -> float:
    """
    Deterministic-ish power-law shock magnitude in [0.05, 0.65] * intensity, with decay.
    """
    enabled = bool(wildcard.get("enabled", True))
    if not enabled:
        return 0.0

    event_year = int(wildcard.get("year", (year_min + year_max) // 2))
    if year < event_year:
        return 0.0

    intensity = float(wildcard.get("intensity", 0.6))
    intensity = max(0.0, min(1.0, intensity))

    # Deterministic random based on seed+event_year so it is stable across reruns.
    alpha = float(wildcard.get("alpha", 2.2))
    seed = int(wildcard.get("seed", 0))
    rng = np.random.default_rng(seed + event_year)

    # Pareto-like draw
    u = float(rng.random())
    raw = 0.05 * (u ** (-1.0 / max(0.8, (alpha - 1.0))))
    mag = max(0.05, min(0.65, raw)) * intensity

    # Decay after event
    dt = float(year - event_year)
    decay_years = float(wildcard.get("decay_years", 6.0))
    decay_years = max(1.0, decay_years)
    decay = float(np.exp(-dt / decay_years))

    return float(mag * decay)


def compute_resilience_series(
    region_id: str,
    scenario_id: str,
    cfg: Optional[ResilienceConfig] = None,
    weights: Optional[Dict[str, float]] = None,
    wildcard: Optional[Dict[str, Any]] = None,
    year_min: int = 2025,
    year_max: int = 2055,
    **kwargs: Any,  # allows older/newer callers without breaking
) -> ResilienceResult:
    """
    CRISI formula:
      R = a*(1-E) + b*(1-S) + c*A
    E,S,A are in [0,1], R in [0,1], then scaled to 0..100.
    """

    cfg0 = cfg or DEFAULT_CONFIG
    cfg_eff = _merge_cfg_and_override(cfg0, weights)

    # Normalize weights
    pillar_w = _norm_weights(
        {"w_exposure": cfg_eff.w_exposure, "w_sensitivity": cfg_eff.w_sensitivity, "w_adaptive": cfg_eff.w_adaptive}
    )
    exp_w = _norm_weights({"w_heat": cfg_eff.w_heat, "w_drought": cfg_eff.w_drought, "w_flood": cfg_eff.w_flood})
    sen_w = _norm_weights(
        {"w_tourism_share": cfg_eff.w_tourism_share, "w_arrivals_index": cfg_eff.w_arrivals_index, "w_income_vulnerability": cfg_eff.w_income_vulnerability}
    )
    ada_w = _norm_weights({"w_education": cfg_eff.w_education, "w_health": cfg_eff.w_health, "w_governance": cfg_eff.w_governance})

    hazard = load_hazard_series(region_id=region_id, scenario_id=scenario_id)
    if hazard.empty:
        raise ValueError(f"No hazard data for region_id={region_id} scenario_id={scenario_id}")
    hazard = hazard[(hazard["year"] >= int(year_min)) & (hazard["year"] <= int(year_max))].copy()
    hazard = hazard.sort_values("year")

    years = hazard["year"].astype(int).to_numpy()
    y0 = int(years.min())
    yN = int(years.max())
    horizon = max(1, (yN - y0))

    # Flood is scalar 0..1 for now; (you can upgrade to time series later)
    flood_base = _clamp01(float(load_flood_risk(region_id=region_id)))

    socio = load_socio_series(region_id=region_id)
    if socio is None or socio.empty:
        raise ValueError(f"No socio data for region_id={region_id}")

    # Use latest row as baseline (if socio has years), otherwise first row
    if "year" in socio.columns:
        base = socio.sort_values("year").iloc[-1]
    else:
        base = socio.iloc[0]

    # Required baseline columns (as created by your socio builder)
    base_tourism = _clamp01(float(base["tourism_share"]))
    base_arrivals = _clamp01(float(base["arrivals_index"]))
    base_gdp = _clamp01(float(base["gdp_index"]))
    base_edu = _clamp01(float(base["education_index"]))
    base_health = _clamp01(float(base["health_index"]))
    base_gov = _clamp01(float(base["governance_index"]))

    p = _scenario_params(scenario_id)

    rows = []
    for _, h in hazard.iterrows():
        year = int(h["year"])
        t = (year - y0) / horizon  # 0..1

        # Region heterogeneity scaling: tourism dependence amplifies certain effects
        dep = base_tourism  # 0..1

        tourism_share = _clamp01(base_tourism * (1.0 + p["tourism"] * t))
        arrivals_index = _clamp01(base_arrivals * (1.0 + (p["arrivals"] * (0.4 + 0.6 * dep)) * t))
        gdp_index = _clamp01(base_gdp * (1.0 + p["gdp"] * t))

        education_index = _clamp01(base_edu + p["edu"] * t)
        health_index = _clamp01(base_health + p["health"] * t)
        governance_index = _clamp01(base_gov + p["gov"] * (0.4 + 0.6 * dep) * t)

        # Hazards
        heat = _clamp01(float(h["heat_index"]))
        drought = _clamp01(float(h["drought_index"]))
        flood = flood_base

        # Apply wildcard shock (if any) by directly modifying underlying drivers
        if wildcard:
            shock = _wildcard_shock(year, wildcard, year_min, year_max)
            if shock > 0.0:
                kind = str(wildcard.get("type", wildcard.get("polarity", "bad"))).lower().strip()
                good = kind in {"good", "positive", "upside"}

                # bad shock increases hazards, reduces capacity and GDP; good shock does the opposite
                sgn = -1.0 if good else 1.0

                heat = _clamp01(heat + sgn * shock)
                drought = _clamp01(drought + sgn * shock)
                flood = _clamp01(flood + sgn * 0.6 * shock)

                # socio impacts (bad: tourism dependence up, arrivals down, gdp down, capacity down)
                tourism_share = _clamp01(tourism_share + sgn * 0.35 * shock)
                arrivals_index = _clamp01(arrivals_index - sgn * 0.30 * shock)
                gdp_index = _clamp01(gdp_index - sgn * 0.25 * shock)

                education_index = _clamp01(education_index - sgn * 0.15 * shock)
                health_index = _clamp01(health_index - sgn * 0.15 * shock)
                governance_index = _clamp01(governance_index - sgn * 0.20 * shock)

        # Pillars
        exposure = _clamp01(exp_w["w_heat"] * heat + exp_w["w_drought"] * drought + exp_w["w_flood"] * flood)

        income_vulnerability = _clamp01(1.0 - gdp_index)
        sensitivity = _clamp01(
            sen_w["w_tourism_share"] * tourism_share
            + sen_w["w_arrivals_index"] * arrivals_index
            + sen_w["w_income_vulnerability"] * income_vulnerability
        )

        adaptive = _clamp01(
            ada_w["w_education"] * education_index
            + ada_w["w_health"] * health_index
            + ada_w["w_governance"] * governance_index
        )

        a = pillar_w["w_exposure"]
        b = pillar_w["w_sensitivity"]
        c = pillar_w["w_adaptive"]

        resilience = _clamp01(a * (1.0 - exposure) + b * (1.0 - sensitivity) + c * adaptive)
        risk = _clamp01(a * exposure + b * sensitivity + c * (1.0 - adaptive))
        
        rows.append(
            dict(
                year=year,
                heat_index=heat,
                drought_index=drought,
                flood_risk_index=flood,
                flood_risk=flood,
                tourism_share=tourism_share,
                arrivals_index=arrivals_index,
                gdp_index=gdp_index,
                income_vulnerability=income_vulnerability,
                education_index=education_index,
                health_index=health_index,
                governance_index=governance_index,
                education=education_index,
                health=health_index,
                governance=governance_index,
                exposure=exposure,
                sensitivity=sensitivity,
                adaptive=adaptive,
                resilience=resilience,
                risk=risk,
                resilience_100=100.0 * resilience,
                risk_100=100.0 * risk,
            )
        )

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    meta = dict(
        pillar_weights=pillar_w,
        exposure_weights=exp_w,
        sensitivity_weights=sen_w,
        adaptive_weights=ada_w,
    )

    return ResilienceResult(region_id=region_id, scenario_id=scenario_id, df=df, meta=meta)
