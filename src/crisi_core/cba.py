from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .models import ResilienceConfig
from .scoring import compute_resilience_series

ADAPTATION_EFFECTIVENESS = 0.5


@dataclass
class ProjectInputs:
    region_id: str
    scenario_id: str

    start_year: int = 2025
    project_life: int = 20
    construction_years: int = 1

    discount_rate: float = 0.04

    capex_total: float = 3_000_000.0
    adaptation_share_of_capex: float = 0.0
    subsidy_rate: float = 0.0

    salvage_value: float = 0.0
    depreciation_years: int = 20

    initial_revenue: float = 1_000_000.0
    revenue_growth: float = 0.02
    ramp_up_years: int = 3

    initial_opex: float = 400_000.0
    opex_growth: float = 0.02

    tax_rate: float = 0.25
    resilience_revenue_sensitivity: float = 0.5


@dataclass
class CBAMetrics:
    npv: float
    irr: Optional[float]
    payback_year: Optional[int]
    cashflow_df: pd.DataFrame


def _compute_irr(cashflows: np.ndarray) -> Optional[float]:
    years = np.arange(len(cashflows), dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(cashflows / (1.0 + rate) ** years))

    low, high = -0.9, 1.0
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return None

    for _ in range(120):
        mid = 0.5 * (low + high)
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    return 0.5 * (low + high)


def evaluate_project_cba(
    inputs: ProjectInputs,
    cfg: ResilienceConfig,
    weights: Optional[Dict[str, float]] = None,
    wildcard: Optional[dict] = None,
) -> CBAMetrics:
    """
    Core CBA engine. Key design:
    - Resilience trajectory affects revenue via resilience_revenue_sensitivity.
    - Adaptation share increases resilience within the project horizon (bounded to 0..1).
    - Optional sandbox weights and wildcard events propagate to the resilience engine.
    """
    res = compute_resilience_series(
        region_id=inputs.region_id,
        scenario_id=inputs.scenario_id,
        cfg=cfg,
        weights=weights,
        wildcard=wildcard,
    )

    n_years = max(1, int(inputs.project_life))
    years_idx = np.arange(n_years, dtype=int)
    calendar_years = int(inputs.start_year) + years_idx

    # Align resilience to project calendar
    res_series = res.df.set_index("year")["resilience"]
    res_aligned = res_series.reindex(calendar_years).ffill().bfill().to_numpy()

    # Apply adaptation uplift to resilience within project horizon
    res_project = res_aligned.copy()
    adapt_share = float(np.clip(inputs.adaptation_share_of_capex, 0.0, 1.0))
    if adapt_share > 0.0 and ADAPTATION_EFFECTIVENESS > 0.0:
        k = ADAPTATION_EFFECTIVENESS * adapt_share
        res_project = res_project + k * (1.0 - res_project)
        res_project = np.clip(res_project, 0.0, 1.0)

    # Revenue baseline
    base_revenue = inputs.initial_revenue * (1.0 + inputs.revenue_growth) ** years_idx
    if int(inputs.ramp_up_years) > 0:
        ramp = np.minimum(1.0, (years_idx + 1) / float(inputs.ramp_up_years))
        base_revenue = base_revenue * ramp

    # OPEX baseline
    opex = inputs.initial_opex * (1.0 + inputs.opex_growth) ** years_idx

    # Map resilience changes to revenue multiplier (relative to first project year)
    R0 = float(res_project[0])
    eps = max(abs(R0), 1e-6)
    rel_delta = (res_project - R0) / eps
    multiplier = 1.0 + float(inputs.resilience_revenue_sensitivity) * rel_delta
    revenue = base_revenue * multiplier

    ebit = revenue - opex

    # Depreciation
    depreciation = np.zeros(n_years, dtype=float)
    dep_years = max(1, int(inputs.depreciation_years))
    depreciation_amount = float(inputs.capex_total) / dep_years
    for t in range(min(dep_years, n_years)):
        depreciation[t] = depreciation_amount

    taxable_income = ebit - depreciation
    tax = np.where(taxable_income > 0, float(inputs.tax_rate) * taxable_income, 0.0)

    # CAPEX spread over construction years (negative cashflow)
    capex = np.zeros(n_years, dtype=float)
    c_years = max(1, int(inputs.construction_years))
    for t in range(min(c_years, n_years)):
        capex[t] = -float(inputs.capex_total) / c_years

    # Subsidy (applied at t=0, proportional to adaptation share)
    subsidy = np.zeros(n_years, dtype=float)
    subsidy[0] = float(inputs.subsidy_rate) * adapt_share * float(inputs.capex_total)

    # Salvage at end (after tax)
    salvage_after_tax = np.zeros(n_years, dtype=float)
    if float(inputs.salvage_value) != 0.0:
        salvage_tax = float(inputs.tax_rate) * max(0.0, float(inputs.salvage_value))
        salvage_after_tax[-1] = float(inputs.salvage_value) - salvage_tax

    net_cashflow = (ebit - tax) + depreciation + capex + subsidy + salvage_after_tax

    # Discounting
    disc = float(inputs.discount_rate)
    discount_factor = 1.0 / (1.0 + disc) ** years_idx
    discounted_cashflow = net_cashflow * discount_factor
    cumulative_discounted = np.cumsum(discounted_cashflow)

    npv = float(discounted_cashflow.sum())
    irr = _compute_irr(net_cashflow)

    payback = None
    for i in range(n_years):
        if cumulative_discounted[i] >= 0:
            payback = int(calendar_years[i])
            break

    df = pd.DataFrame(
        {
            "year": calendar_years,
            "resilience": res_project,
            "base_revenue": base_revenue,
            "revenue": revenue,
            "opex": opex,
            "ebit": ebit,
            "depreciation": depreciation,
            "taxable_income": taxable_income,
            "tax": tax,
            "subsidy": subsidy,
            "capex": capex,
            "salvage_after_tax": salvage_after_tax,
            "net_cashflow": net_cashflow,
            "discount_factor": discount_factor,
            "discounted_cashflow": discounted_cashflow,
            "cumulative_discounted": cumulative_discounted,
        }
    )

    return CBAMetrics(npv=npv, irr=irr, payback_year=payback, cashflow_df=df)
