from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.cba import ProjectInputs, evaluate_project_cba


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "surrogate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2025, 2055

# Keep in sync with your app region list
GREEK_NUTS2 = [
    "EL11", "EL12", "EL13", "EL14", "EL21", "EL22", "EL23",
    "EL24", "EL25", "EL30", "EL41", "EL42", "EL43"
]


def _dirichlet3(rng: np.random.Generator) -> Tuple[float, float, float]:
    a = rng.dirichlet([1.5, 1.5, 1.5])
    return float(a[0]), float(a[1]), float(a[2])


def _dirichlet_k(rng: np.random.Generator, k: int) -> List[float]:
    a = rng.dirichlet([1.5] * k)
    return [float(x) for x in a]


def sample_weights(rng: np.random.Generator) -> Dict[str, float]:
    w_exposure, w_sensitivity, w_adaptive = _dirichlet3(rng)

    # subweights
    w_heat, w_drought, w_flood = _dirichlet_k(rng, 3)
    w_tour, w_arr, w_inc = _dirichlet_k(rng, 3)
    w_edu, w_health, w_gov = _dirichlet_k(rng, 3)

    return {
        "w_exposure": w_exposure,
        "w_sensitivity": w_sensitivity,
        "w_adaptive": w_adaptive,
        "w_heat": w_heat,
        "w_drought": w_drought,
        "w_flood": w_flood,
        "w_tourism_share": w_tour,
        "w_arrivals_index": w_arr,
        "w_income_vulnerability": w_inc,
        "w_education": w_edu,
        "w_health": w_health,
        "w_governance": w_gov,
    }


def cfg_from_weights(base: ResilienceConfig, weights: Dict[str, float]) -> ResilienceConfig:
    d = base.__dict__.copy()
    for k, v in weights.items():
        if k in d:
            d[k] = float(v)
    return ResilienceConfig(**d)


def _closest_year_row(df: pd.DataFrame, year: int) -> pd.Series:
    years = df["year"].astype(int)
    idx = (years - int(year)).abs().idxmin()
    return df.loc[idx]


def generate_resilience_samples(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_cfg = ResilienceConfig()
    scenarios = list(DEFAULT_SCENARIOS.keys())

    rows = []
    for _ in range(n):
        region_id = str(rng.choice(GREEK_NUTS2))
        scenario_id = str(rng.choice(scenarios))
        year = int(rng.integers(YEAR_MIN, YEAR_MAX + 1))

        weights = sample_weights(rng)
        cfg = cfg_from_weights(base_cfg, weights)

        res = compute_resilience_series(region_id=region_id, scenario_id=scenario_id, cfg=cfg)
        df = res.df
        r = _closest_year_row(df, year)

        rows.append(
            {
                "region_id": region_id,
                "scenario_id": scenario_id,
                "year": int(r.get("year", year)),
                # indicators
                "heat_index": float(r.get("heat_index", 0.0)),
                "drought_index": float(r.get("drought_index", 0.0)),
                "flood_risk": float(r.get("flood_risk", 0.0)),
                "tourism_share": float(r.get("tourism_share", 0.0)),
                "arrivals_index": float(r.get("arrivals_index", 0.0)),
                "income_vulnerability": float(r.get("income_vulnerability", 0.0)),
                "education": float(r.get("education", 0.0)),
                "health": float(r.get("health", 0.0)),
                "governance": float(r.get("governance", 0.0)),
                # weights (features)
                **{k: float(v) for k, v in weights.items()},
                # targets
                "resilience_100": float(r.get("resilience_100", np.nan)),
                "risk_100": float(r.get("risk_100", np.nan)),
                "exposure": float(r.get("exposure", np.nan)),
                "sensitivity": float(r.get("sensitivity", np.nan)),
                "adaptive": float(r.get("adaptive", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def generate_cba_samples(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 77)
    base_cfg = ResilienceConfig()
    scenarios = list(DEFAULT_SCENARIOS.keys())

    rows = []
    for _ in range(n):
        region_id = str(rng.choice(GREEK_NUTS2))
        scenario_id = str(rng.choice(scenarios))

        # start year so project ends within CRISI horizon
        start_year = int(rng.integers(2025, 2046))
        project_life = int(rng.integers(10, 26))
        if start_year + project_life > 2055:
            project_life = max(5, 2055 - start_year)

        weights = sample_weights(rng)
        cfg = cfg_from_weights(base_cfg, weights)

        # economic parameters (moderate ranges; tune later)
        capex = float(rng.uniform(1_000_000, 15_000_000))
        adaptation_share = float(rng.uniform(0.0, 0.5))
        subsidy_rate = float(rng.uniform(0.0, 0.5))
        discount_rate = float(rng.uniform(0.02, 0.10))

        initial_revenue = float(rng.uniform(50_000, 2_000_000))
        revenue_growth = float(rng.uniform(-0.02, 0.08))
        initial_opex = float(rng.uniform(20_000, 1_500_000))
        opex_growth = float(rng.uniform(-0.01, 0.06))

        tax_rate = float(rng.uniform(0.10, 0.35))
        res_sens = float(rng.uniform(0.0, 1.5))

        salvage = float(rng.uniform(0.0, 0.2) * capex)
        dep_years = int(rng.integers(10, 31))

        inputs = ProjectInputs(
            region_id=region_id,
            scenario_id=scenario_id,
            start_year=start_year,
            project_life=project_life,
            discount_rate=discount_rate,
            capex_total=capex,
            adaptation_share_of_capex=adaptation_share,
            subsidy_rate=subsidy_rate,
            salvage_value=salvage,
            depreciation_years=dep_years,
            initial_revenue=initial_revenue,
            revenue_growth=revenue_growth,
            initial_opex=initial_opex,
            opex_growth=opex_growth,
            tax_rate=tax_rate,
            resilience_revenue_sensitivity=res_sens,
        )

        metrics = evaluate_project_cba(inputs, cfg)

        irr_val = getattr(metrics, "irr", None)
        irr_num = np.nan if irr_val is None else float(irr_val)

        rows.append(
            {
                "region_id": region_id,
                "scenario_id": scenario_id,
                "start_year": start_year,
                "project_life": project_life,
                # economics (features)
                "discount_rate": discount_rate,
                "capex_total": capex,
                "adaptation_share_of_capex": adaptation_share,
                "subsidy_rate": subsidy_rate,
                "salvage_value": salvage,
                "depreciation_years": dep_years,
                "initial_revenue": initial_revenue,
                "revenue_growth": revenue_growth,
                "initial_opex": initial_opex,
                "opex_growth": opex_growth,
                "tax_rate": tax_rate,
                "resilience_revenue_sensitivity": res_sens,
                # weights (features)
                **{k: float(v) for k, v in weights.items()},
                # targets
                "npv": float(getattr(metrics, "npv", np.nan)),
                "irr": irr_num,
            }
        )

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)

    # Keep training data clean
    out = out.dropna(subset=["npv"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_resilience", type=int, default=2500)
    ap.add_argument("--n_cba", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating resilience samples: n={args.n_resilience}")
    df_r = generate_resilience_samples(args.n_resilience, args.seed)
    out_r = OUT_DIR / "resilience_samples.parquet"
    df_r.to_parquet(out_r, index=False)
    print(f"Wrote: {out_r} rows={len(df_r)}")

    print(f"Generating CBA samples: n={args.n_cba}")
    df_c = generate_cba_samples(args.n_cba, args.seed)
    out_c = OUT_DIR / "cba_samples.parquet"
    df_c.to_parquet(out_c, index=False)
    print(f"Wrote: {out_c} rows={len(df_c)}")

    print("Done.")


if __name__ == "__main__":
    main()
