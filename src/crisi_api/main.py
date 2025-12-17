from __future__ import annotations

from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crisi_core import (
    ResilienceConfig,
    DEFAULT_SCENARIOS,
    DEFAULT_CONFIG,
    compute_resilience_series,
    ProjectInputs,
    evaluate_project_cba,
)

app = FastAPI(title="CRISI API v2")


class ScenarioOut(BaseModel):
    id: str
    name: str
    description: str


class ResilienceWeights(BaseModel):
    # Pillar weights
    w_exposure: Optional[float] = None
    w_sensitivity: Optional[float] = None
    w_adaptive: Optional[float] = None

    # Exposure breakdown
    w_heat: Optional[float] = None
    w_drought: Optional[float] = None
    w_flood: Optional[float] = None  # safe even if core ignores it

    # Sensitivity breakdown
    w_tourism_share: Optional[float] = None
    w_arrivals_index: Optional[float] = None
    w_income_vulnerability: Optional[float] = None

    # Adaptive capacity breakdown
    w_education: Optional[float] = None
    w_health: Optional[float] = None
    w_governance: Optional[float] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/scenarios", response_model=List[ScenarioOut])
def list_scenarios() -> List[ScenarioOut]:
    return [
        ScenarioOut(id=sid, name=s.name, description=s.description)
        for sid, s in DEFAULT_SCENARIOS.items()
    ]


class ResilienceRequest(BaseModel):
    region_id: str
    scenario_id: str
    weights: Optional[ResilienceWeights] = None
    wildcard: Optional[Dict[str, Any]] = None


class ResiliencePoint(BaseModel):
    year: int
    exposure: float
    sensitivity: float
    adaptive: float
    resilience: float


class ResilienceResponse(BaseModel):
    region_id: str
    scenario_id: str
    series: List[ResiliencePoint]


@app.post("/resilience", response_model=ResilienceResponse)
def get_resilience(req: ResilienceRequest) -> ResilienceResponse:
    if req.scenario_id not in DEFAULT_SCENARIOS:
        raise HTTPException(status_code=404, detail="Unknown scenario_id")

    weights_dict: Optional[Dict[str, float]] = (
        req.weights.model_dump(exclude_none=True) if req.weights is not None else None
    )

    res = compute_resilience_series(
        region_id=req.region_id,
        scenario_id=req.scenario_id,
        weights=weights_dict,
        wildcard=req.wildcard,
    )

    df = res.df
    series = [
        ResiliencePoint(
            year=int(row["year"]),
            exposure=float(row["exposure"]),
            sensitivity=float(row["sensitivity"]),
            adaptive=float(row["adaptive"]),
            resilience=float(row["resilience"]),
        )
        for _, row in df.iterrows()
    ]

    return ResilienceResponse(region_id=req.region_id, scenario_id=req.scenario_id, series=series)


class CBARequest(BaseModel):
    region_id: str
    scenario_id: str

    start_year: int = 2025
    project_life: int = 20
    construction_years: int = 1
    discount_rate: float = 0.04

    capex_total: float
    adaptation_share_of_capex: float = 0.0
    subsidy_rate: float = 0.0

    salvage_value: float = 0.0
    depreciation_years: int = 20

    initial_revenue: float
    revenue_growth: float = 0.02
    ramp_up_years: int = 3

    initial_opex: float
    opex_growth: float = 0.02

    tax_rate: float = 0.25
    resilience_revenue_sensitivity: float = 0.5

    weights: Optional[ResilienceWeights] = None
    wildcard: Optional[Dict[str, Any]] = None


class CBACashflowPoint(BaseModel):
    year: int
    resilience: float
    base_revenue: float
    revenue: float
    opex: float
    ebit: float
    depreciation: float
    taxable_income: float
    tax: float
    subsidy: float
    capex: float
    salvage_after_tax: float
    net_cashflow: float
    discount_factor: float
    discounted_cashflow: float
    cumulative_discounted: float


class CBAResponse(BaseModel):
    region_id: str
    scenario_id: str
    start_year: int
    project_life: int
    npv: float
    irr: Optional[float]
    payback_year: Optional[int]
    series: List[CBACashflowPoint]


@app.post("/cba/evaluate", response_model=CBAResponse)
def evaluate_cba(req: CBARequest) -> CBAResponse:
    if req.scenario_id not in DEFAULT_SCENARIOS:
        raise HTTPException(status_code=404, detail="Unknown scenario_id")

    weights_dict: Optional[Dict[str, float]] = (
        req.weights.model_dump(exclude_none=True) if req.weights is not None else None
    )

    inputs = ProjectInputs(
        region_id=req.region_id,
        scenario_id=req.scenario_id,
        start_year=req.start_year,
        project_life=req.project_life,
        construction_years=req.construction_years,
        discount_rate=req.discount_rate,
        capex_total=req.capex_total,
        adaptation_share_of_capex=req.adaptation_share_of_capex,
        subsidy_rate=req.subsidy_rate,
        salvage_value=req.salvage_value,
        depreciation_years=req.depreciation_years,
        initial_revenue=req.initial_revenue,
        revenue_growth=req.revenue_growth,
        ramp_up_years=req.ramp_up_years,
        initial_opex=req.initial_opex,
        opex_growth=req.opex_growth,
        tax_rate=req.tax_rate,
        resilience_revenue_sensitivity=req.resilience_revenue_sensitivity,
    )

    metrics = evaluate_project_cba(
        inputs=inputs,
        cfg=DEFAULT_CONFIG,
        weights=weights_dict,
        wildcard=req.wildcard,
    )

    df = metrics.cashflow_df

    series = [
        CBACashflowPoint(
            year=int(row["year"]),
            resilience=float(row["resilience"]),
            base_revenue=float(row["base_revenue"]),
            revenue=float(row["revenue"]),
            opex=float(row["opex"]),
            ebit=float(row["ebit"]),
            depreciation=float(row["depreciation"]),
            taxable_income=float(row["taxable_income"]),
            tax=float(row["tax"]),
            subsidy=float(row["subsidy"]),
            capex=float(row["capex"]),
            salvage_after_tax=float(row["salvage_after_tax"]),
            net_cashflow=float(row["net_cashflow"]),
            discount_factor=float(row["discount_factor"]),
            discounted_cashflow=float(row["discounted_cashflow"]),
            cumulative_discounted=float(row["cumulative_discounted"]),
        )
        for _, row in df.iterrows()
    ]

    return CBAResponse(
        region_id=req.region_id,
        scenario_id=req.scenario_id,
        start_year=req.start_year,
        project_life=req.project_life,
        npv=metrics.npv,
        irr=metrics.irr,
        payback_year=metrics.payback_year,
        series=series,
    )
