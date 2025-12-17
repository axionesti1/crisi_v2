from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from .models import ResilienceConfig
from .cba import ProjectInputs, evaluate_project_cba


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_weights(d: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    vals = np.array([max(0.0, float(d.get(k, 0.0))) for k in keys], dtype=float)
    s = float(vals.sum())
    if s <= 1e-12:
        vals[:] = 1.0 / len(keys)
    else:
        vals = vals / s
    return {k: float(v) for k, v in zip(keys, vals)}


def _closest_year_row(df: pd.DataFrame, year: int) -> pd.Series:
    if "year" not in df.columns or df.empty:
        raise ValueError("Expected non-empty df with a 'year' column.")
    years = df["year"].astype(int)
    idx = (years - int(year)).abs().idxmin()
    return df.loc[idx]


# ==========================================================
# Resilience XAI
# ==========================================================
def resilience_xai_breakdown(
    df: pd.DataFrame,
    config_used: Dict[str, float],
    year: int,
    scale_100: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Produces an explainable decomposition consistent with:
      R = a*(1-E) + b*(1-S) + c*A
    with E,S,A computed as weighted sums of indicators.

    Returns dict of DataFrames:
      - pillars: contributions of (1-E), (1-S), A to R
      - exposure: hazard contributions to the (1-E) part
      - sensitivity: indicator contributions to the (1-S) part
      - adaptive: indicator contributions to the A part
      - drivers_lowR: "bad drivers" contributions (hazards, sensitivity, 1-A)
    """
    row = _closest_year_row(df, year)

    # Ensure normalized weights
    w_p = _norm_weights(config_used, ["w_exposure", "w_sensitivity", "w_adaptive"])
    w_e = _norm_weights(config_used, ["w_heat", "w_drought", "w_flood"])
    w_s = _norm_weights(config_used, ["w_tourism_share", "w_arrivals_index", "w_income_vulnerability"])
    w_a = _norm_weights(config_used, ["w_education", "w_health", "w_governance"])

    a, b, c = w_p["w_exposure"], w_p["w_sensitivity"], w_p["w_adaptive"]

    # Pull indicators defensively
    heat = _clip01(float(row.get("heat_index", 0.0)))
    drought = _clip01(float(row.get("drought_index", 0.0)))
    flood = _clip01(float(row.get("flood_risk", 0.0)))

    tshare = _clip01(float(row.get("tourism_share", 0.0)))
    arrivals = _clip01(float(row.get("arrivals_index", 0.0)))
    incomev = _clip01(float(row.get("income_vulnerability", 0.0)))

    edu = _clip01(float(row.get("education", 0.0)))
    health = _clip01(float(row.get("health", 0.0)))
    gov = _clip01(float(row.get("governance", 0.0)))

    # Recompute E,S,A from indicators (authoritative for XAI)
    E = _clip01(w_e["w_heat"] * heat + w_e["w_drought"] * drought + w_e["w_flood"] * flood)
    S = _clip01(w_s["w_tourism_share"] * tshare + w_s["w_arrivals_index"] * arrivals + w_s["w_income_vulnerability"] * incomev)
    A = _clip01(w_a["w_education"] * edu + w_a["w_health"] * health + w_a["w_governance"] * gov)

    R = _clip01(a * (1.0 - E) + b * (1.0 - S) + c * A)

    factor = 100.0 if scale_100 else 1.0

    pillars = pd.DataFrame(
        [
            {"component": "Lack of exposure (1−E)", "value": (1.0 - E), "weight": a, "contribution": a * (1.0 - E)},
            {"component": "Lack of sensitivity (1−S)", "value": (1.0 - S), "weight": b, "contribution": b * (1.0 - S)},
            {"component": "Adaptive capacity (A)", "value": A, "weight": c, "contribution": c * A},
            {"component": "Total resilience (R)", "value": R, "weight": 1.0, "contribution": R},
        ]
    )
    pillars[["value", "weight", "contribution"]] = pillars[["value", "weight", "contribution"]] * factor

    # Exposure contributions to the "good" term a*(1-E)
    exposure = pd.DataFrame(
        [
            {"indicator": "Heat (1−heat)", "indicator_value": heat, "subweight": w_e["w_heat"], "good_part": (1.0 - heat)},
            {"indicator": "Drought (1−drought)", "indicator_value": drought, "subweight": w_e["w_drought"], "good_part": (1.0 - drought)},
            {"indicator": "Flood (1−flood)", "indicator_value": flood, "subweight": w_e["w_flood"], "good_part": (1.0 - flood)},
        ]
    )
    exposure["contribution_to_R"] = a * exposure["subweight"] * exposure["good_part"]
    exposure[["indicator_value", "subweight", "good_part", "contribution_to_R"]] = exposure[
        ["indicator_value", "subweight", "good_part", "contribution_to_R"]
    ] * factor

    # Sensitivity contributions to the "good" term b*(1-S)
    sensitivity = pd.DataFrame(
        [
            {"indicator": "Tourism share (1−tourism)", "indicator_value": tshare, "subweight": w_s["w_tourism_share"], "good_part": (1.0 - tshare)},
            {"indicator": "Arrivals (1−arrivals)", "indicator_value": arrivals, "subweight": w_s["w_arrivals_index"], "good_part": (1.0 - arrivals)},
            {"indicator": "Income vulnerability (1−income)", "indicator_value": incomev, "subweight": w_s["w_income_vulnerability"], "good_part": (1.0 - incomev)},
        ]
    )
    sensitivity["contribution_to_R"] = b * sensitivity["subweight"] * sensitivity["good_part"]
    sensitivity[["indicator_value", "subweight", "good_part", "contribution_to_R"]] = sensitivity[
        ["indicator_value", "subweight", "good_part", "contribution_to_R"]
    ] * factor

    # Adaptive contributions to the "good" term c*A
    adaptive = pd.DataFrame(
        [
            {"indicator": "Education", "indicator_value": edu, "subweight": w_a["w_education"], "good_part": edu},
            {"indicator": "Health", "indicator_value": health, "subweight": w_a["w_health"], "good_part": health},
            {"indicator": "Governance", "indicator_value": gov, "subweight": w_a["w_governance"], "good_part": gov},
        ]
    )
    adaptive["contribution_to_R"] = c * adaptive["subweight"] * adaptive["good_part"]
    adaptive[["indicator_value", "subweight", "good_part", "contribution_to_R"]] = adaptive[
        ["indicator_value", "subweight", "good_part", "contribution_to_R"]
    ] * factor

    # "Bad driver" view: what pushes resilience down (hazards, sensitivity, lack of A)
    drivers_lowR = pd.DataFrame(
        [
            {"driver": "Heat hazard", "term": "a*E", "value": heat, "weight": a * w_e["w_heat"], "impact": a * w_e["w_heat"] * heat},
            {"driver": "Drought hazard", "term": "a*E", "value": drought, "weight": a * w_e["w_drought"], "impact": a * w_e["w_drought"] * drought},
            {"driver": "Flood hazard", "term": "a*E", "value": flood, "weight": a * w_e["w_flood"], "impact": a * w_e["w_flood"] * flood},
            {"driver": "Tourism sensitivity", "term": "b*S", "value": tshare, "weight": b * w_s["w_tourism_share"], "impact": b * w_s["w_tourism_share"] * tshare},
            {"driver": "Arrivals sensitivity", "term": "b*S", "value": arrivals, "weight": b * w_s["w_arrivals_index"], "impact": b * w_s["w_arrivals_index"] * arrivals},
            {"driver": "Income sensitivity", "term": "b*S", "value": incomev, "weight": b * w_s["w_income_vulnerability"], "impact": b * w_s["w_income_vulnerability"] * incomev},
            {"driver": "Lack of education", "term": "c*(1−A)", "value": (1.0 - edu), "weight": c * w_a["w_education"], "impact": c * w_a["w_education"] * (1.0 - edu)},
            {"driver": "Lack of health", "term": "c*(1−A)", "value": (1.0 - health), "weight": c * w_a["w_health"], "impact": c * w_a["w_health"] * (1.0 - health)},
            {"driver": "Lack of governance", "term": "c*(1−A)", "value": (1.0 - gov), "weight": c * w_a["w_governance"], "impact": c * w_a["w_governance"] * (1.0 - gov)},
        ]
    )
    drivers_lowR[["value", "weight", "impact"]] = drivers_lowR[["value", "weight", "impact"]] * factor
    drivers_lowR = drivers_lowR.sort_values("impact", ascending=False).reset_index(drop=True)

    return {
        "pillars": pillars,
        "exposure": exposure,
        "sensitivity": sensitivity,
        "adaptive": adaptive,
        "drivers_lowR": drivers_lowR,
    }


# ==========================================================
# CBA XAI
# ==========================================================
def cba_npvdiff_decomposition(
    inputs: ProjectInputs,
    cfg: ResilienceConfig,
    wildcard: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Decompose NPV into interpretable deltas:
      - Total NPV
      - NPV with resilience→revenue link disabled (set sensitivity=0)
      - Delta due to resilience→revenue link
      - NPV with adaptation disabled (adaptation_share=0)
      - Delta due to adaptation (given your adaptation implementation)
    """
    base = evaluate_project_cba(inputs, cfg, wildcard=wildcard).npv

    no_link_inputs = replace(inputs, resilience_revenue_sensitivity=0.0)
    npv_no_link = evaluate_project_cba(no_link_inputs, cfg, wildcard=wildcard).npv
    delta_link = base - npv_no_link

    no_adapt_inputs = replace(inputs, adaptation_share_of_capex=0.0)
    npv_no_adapt = evaluate_project_cba(no_adapt_inputs, cfg, wildcard=wildcard).npv
    delta_adapt = base - npv_no_adapt

    return pd.DataFrame(
        [
            {"item": "Total NPV", "npv": float(base)},
            {"item": "NPV with resilience→revenue disabled", "npv": float(npv_no_link)},
            {"item": "Δ due to resilience→revenue link", "npv": float(delta_link)},
            {"item": "NPV with adaptation disabled", "npv": float(npv_no_adapt)},
            {"item": "Δ due to adaptation (vs none)", "npv": float(delta_adapt)},
        ]
    )


def cba_tornado(
    inputs: ProjectInputs,
    cfg: ResilienceConfig,
    wildcard: Optional[dict] = None,
    rel_step: float = 0.10,
) -> pd.DataFrame:
    """
    Simple local sensitivity (tornado):
      - perturb each parameter by +/- rel_step (or +/- abs step for rates)
      - recompute NPV
    """
    base = evaluate_project_cba(inputs, cfg, wildcard=wildcard).npv

    specs = [
        ("capex_total", "CAPEX", "rel"),
        ("initial_revenue", "Revenue (level)", "rel"),
        ("initial_opex", "OPEX (level)", "rel"),
        ("revenue_growth", "Revenue growth", "abs_rate"),
        ("opex_growth", "OPEX growth", "abs_rate"),
        ("discount_rate", "Discount rate", "abs_rate"),
        ("tax_rate", "Tax rate", "abs_rate"),
        ("adaptation_share_of_capex", "Adaptation share", "abs_share"),
        ("subsidy_rate", "Subsidy rate", "abs_share"),
        ("resilience_revenue_sensitivity", "Resilience→Revenue sensitivity", "rel"),
    ]

    rows = []
    for field, label, kind in specs:
        v0 = float(getattr(inputs, field))

        if kind == "rel":
            v_low = v0 * (1.0 - rel_step)
            v_high = v0 * (1.0 + rel_step)
        elif kind == "abs_rate":
            # For rates: +/- 1 percentage point by default
            v_low = v0 - 0.01
            v_high = v0 + 0.01
        elif kind == "abs_share":
            # For shares: +/- 5 percentage points
            v_low = v0 - 0.05
            v_high = v0 + 0.05
        else:
            v_low = v0
            v_high = v0

        # clip shares/rates to plausible bounds
        if "rate" in kind:
            v_low = max(-0.50, min(0.90, v_low))
            v_high = max(-0.50, min(0.90, v_high))
        if "share" in kind:
            v_low = max(0.0, min(1.0, v_low))
            v_high = max(0.0, min(1.0, v_high))

        low_inputs = replace(inputs, **{field: v_low})
        high_inputs = replace(inputs, **{field: v_high})

        npv_low = evaluate_project_cba(low_inputs, cfg, wildcard=wildcard).npv
        npv_high = evaluate_project_cba(high_inputs, cfg, wildcard=wildcard).npv

        rows.append(
            {
                "Parameter": label,
                "Base value": v0,
                "Low value": v_low,
                "High value": v_high,
                "NPV (low)": float(npv_low),
                "NPV (base)": float(base),
                "NPV (high)": float(npv_high),
                "Impact range (abs)": float(max(npv_low, npv_high) - min(npv_low, npv_high)),
            }
        )

    out = pd.DataFrame(rows).sort_values("Impact range (abs)", ascending=False).reset_index(drop=True)
    return out
