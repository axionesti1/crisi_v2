from __future__ import annotations

import sys
import pathlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import streamlit as st

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -----------------------------
# Regions + label mapping (UI shows NEW codes; internal uses old codes)
# -----------------------------
try:
    from crisi_app.regions import NUTS2_REGIONS, ui_region_label, CRISI_TO_GEO
except Exception:
    # Fallback if regions.py not yet updated
    from crisi_app.regions import NUTS2_REGIONS  # type: ignore

    CRISI_TO_GEO: Dict[str, str] = {
        "EL11": "EL51",
        "EL12": "EL52",
        "EL13": "EL53",
        "EL14": "EL61",
        "EL21": "EL54",
        "EL22": "EL62",
        "EL23": "EL63",
        "EL24": "EL64",
        "EL25": "EL65",
        "EL30": "EL30",
        "EL41": "EL41",
        "EL42": "EL42",
        "EL43": "EL43",
    }

    def ui_region_label(internal_id: str) -> str:
        name = NUTS2_REGIONS.get(internal_id, internal_id)
        new_id = CRISI_TO_GEO.get(internal_id, internal_id)
        return f"{new_id} — {name}"


def display_code(internal_id: str) -> str:
    return CRISI_TO_GEO.get(internal_id, internal_id)


from crisi_app.ui_sandbox import weights_panel
from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.cba import ProjectInputs, evaluate_project_cba
from crisi_core.surrogate import load_surrogate_bundle, predict_and_shap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ART_DIR = PROJECT_ROOT / "artifacts" / "surrogates"

YEAR_MIN, YEAR_MAX = 2025, 2055


# ----------------------------
# Helpers: artifacts + XAI formatting
# ----------------------------
def _artifact_check(sub: str) -> Path | None:
    p = ART_DIR / sub
    needed = ["preprocessor.joblib", "model.joblib", "feature_names.json", "background.npy", "metrics.json"]
    if not p.exists():
        return None
    for f in needed:
        if not (p / f).exists():
            return None
    return p


def _load_metrics_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"raw": path.read_text(encoding="utf-8")}


def _pretty_feature_name(feat: str) -> str:
    """
    Convert encoded feature names (e.g., 'cat__region_id_EL30', 'num__heat_index')
    into something a non-technical user can read.
    """
    if feat.startswith("cat__"):
        rest = feat[len("cat__") :]

        if rest.startswith("region_id_"):
            internal = rest.replace("region_id_", "").strip()
            new_id = display_code(internal)
            name = NUTS2_REGIONS.get(internal, "")
            # NO parentheses / NO internal exposure
            return f"Region = {new_id}" + (f" — {name}" if name else "")

        if rest.startswith("scenario_id_"):
            sid = rest.replace("scenario_id_", "").strip()
            sname = DEFAULT_SCENARIOS.get(sid).name if sid in DEFAULT_SCENARIOS else sid
            return f"Scenario = {sname}"

        return rest

    if feat.startswith("num__"):
        base = feat[len("num__") :]
        mapping = {
            "year": "Year",
            "heat_index": "Heat index (hazard indicator)",
            "drought_index": "Drought index (hazard indicator)",
            "flood_risk": "Flood risk (hazard indicator)",
            "tourism_share": "Tourism dependence",
            "arrivals_index": "Arrivals dynamics index",
            "income_vulnerability": "Income vulnerability",
            "education": "Education capacity indicator",
            "health": "Health capacity indicator",
            "governance": "Governance capacity indicator",
            # Weights
            "w_exposure": "Weight: Exposure pillar (a)",
            "w_sensitivity": "Weight: Sensitivity pillar (b)",
            "w_adaptive": "Weight: Adaptive capacity pillar (c)",
            "w_heat": "Weight: Heat within exposure",
            "w_drought": "Weight: Drought within exposure",
            "w_flood": "Weight: Flood within exposure",
            "w_tourism_share": "Weight: Tourism dependence within sensitivity",
            "w_arrivals_index": "Weight: Arrivals within sensitivity",
            "w_income_vulnerability": "Weight: Income vulnerability within sensitivity",
            "w_education": "Weight: Education within adaptive capacity",
            "w_health": "Weight: Health within adaptive capacity",
            "w_governance": "Weight: Governance within adaptive capacity",
            # CBA economics
            "start_year": "CBA start year",
            "project_life": "CBA project life (years)",
            "discount_rate": "Discount rate",
            "capex_total": "CAPEX total (€)",
            "adaptation_share_of_capex": "Adaptation share of CAPEX",
            "subsidy_rate": "Subsidy rate",
            "salvage_value": "Salvage value (€)",
            "depreciation_years": "Depreciation period (years)",
            "initial_revenue": "Initial revenue (€)",
            "revenue_growth": "Revenue growth rate",
            "initial_opex": "Initial OPEX (€)",
            "opex_growth": "OPEX growth rate",
            "tax_rate": "Tax rate",
            "resilience_revenue_sensitivity": "Resilience → revenue sensitivity",
        }
        return mapping.get(base, base)

    return feat


def _feature_group(feat: str) -> str:
    if feat.startswith("cat__region_id_"):
        return "Context: Region"
    if feat.startswith("cat__scenario_id_"):
        return "Context: Scenario"

    if feat.startswith("num__"):
        base = feat[len("num__") :]
        if base in {"heat_index", "drought_index", "flood_risk"}:
            return "Hazards (Exposure inputs)"
        if base in {"tourism_share", "arrivals_index", "income_vulnerability"}:
            return "Sensitivity inputs"
        if base in {"education", "health", "governance"}:
            return "Adaptive capacity inputs"
        if base.startswith("w_"):
            return "Model weights (sandbox levers)"
        if base in {
            "start_year",
            "project_life",
            "discount_rate",
            "capex_total",
            "adaptation_share_of_capex",
            "subsidy_rate",
            "salvage_value",
            "depreciation_years",
            "initial_revenue",
            "revenue_growth",
            "initial_opex",
            "opex_growth",
            "tax_rate",
            "resilience_revenue_sensitivity",
        }:
            return "CBA economic inputs"
        if base == "year":
            return "Context: Year"

    return "Other"


def _render_model_quality(metrics: Dict, target_label: str):
    st.markdown("### Model quality (what the surrogate learned)")
    st.markdown(
        "The surrogate is trained to **approximate the CRISI engine** based on synthetic runs. "
        "These numbers tell you how well it matches the engine on held-out synthetic samples."
    )

    if "raw" in metrics:
        st.code(metrics["raw"], language="json")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{metrics.get('rows', '—')}")
    c2.metric(
        "R² (test)",
        f"{metrics.get('r2_test', float('nan')):.3f}" if isinstance(metrics.get("r2_test"), (int, float)) else "—",
    )
    c3.metric(
        "MAE (test)",
        f"{metrics.get('mae_test', float('nan')):.3f}" if isinstance(metrics.get("mae_test"), (int, float)) else "—",
    )
    c4.metric("Target", str(metrics.get("target", target_label)))

    with st.expander("How to interpret model quality", expanded=False):
        st.markdown(
            "- **R² (test)**: 1.0 is perfect match to the engine; near 0 means weak predictive power.\n"
            "- **MAE (test)**: average absolute error on the held-out synthetic set.\n"
            "- If **MAE is non-trivial**, treat SHAP as *qualitative* and validate conclusions using the **true model**.\n"
            "- The surrogate mirrors the *engine*, not reality."
        )


def _render_shap_explanation(shap_df: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    st.markdown(
        "SHAP provides a **local explanation**: for the current case, it attributes how each input pushes the prediction **up** or **down**.\n\n"
        "- **Positive SHAP** → pushes the prediction **higher**.\n"
        "- **Negative SHAP** → pushes the prediction **lower**.\n"
        "- The chart shows **absolute** contributions (importance magnitude), not direction.\n\n"
        "Important: some features are **encoded** (e.g., scenario and region as one-hot flags). "
        "These are context identifiers, not continuous variables."
    )

    top_n = st.slider("How many features to show", 10, 50, 20, 5, key=f"{title}_topn")

    df = shap_df.copy()
    df["feature_pretty"] = df["feature"].apply(_pretty_feature_name)
    df["group"] = df["feature"].apply(_feature_group)

    st.markdown("#### Top drivers (feature-level)")
    show = df[["feature_pretty", "value", "shap_value", "abs_shap", "group"]].head(top_n).copy()
    show = show.rename(
        columns={
            "feature_pretty": "Feature",
            "value": "Encoded value",
            "shap_value": "SHAP (direction)",
            "abs_shap": "|SHAP|",
            "group": "Group",
        }
    )
    st.dataframe(show, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Biggest upward pushes")
        up = df.sort_values("shap_value", ascending=False).head(min(10, len(df)))
        st.dataframe(up.assign(Feature=up["feature_pretty"])[["Feature", "shap_value"]], use_container_width=True)
    with colB:
        st.markdown("#### Biggest downward pushes")
        down = df.sort_values("shap_value", ascending=True).head(min(10, len(df)))
        st.dataframe(down.assign(Feature=down["feature_pretty"])[["Feature", "shap_value"]], use_container_width=True)

    st.markdown("#### Magnitude chart (feature-level)")
    mag = df.head(top_n).copy()
    mag.index = mag["feature_pretty"]
    st.bar_chart(mag["abs_shap"])

    st.markdown("#### Grouped attribution (recommended for non-technical users)")
    grp = df.groupby("group", as_index=False)["abs_shap"].sum().sort_values("abs_shap", ascending=False)
    grp = grp.rename(columns={"group": "Group", "abs_shap": "Total |SHAP|"})
    st.dataframe(grp, use_container_width=True)
    st.bar_chart(grp.set_index("Group")["Total |SHAP|"])

    with st.expander("How to use SHAP correctly (and common mistakes)", expanded=False):
        st.markdown(
            "- SHAP explains the **surrogate’s prediction**, not the true model directly, and not reality.\n"
            "- If the **surrogate error is high**, do not trust the explanation.\n"
            "- Do not interpret region/scenario one-hot flags as policy levers—they are **context**.\n"
            "- Use SHAP to identify **what matters**, then validate via sensitivity runs on the core pages."
        )


# ----------------------------
# Page header
# ----------------------------
st.set_page_config(page_title="CRISI v2 – AI/XAI Lab", layout="wide")

st.title("AI/XAI Lab – Surrogates + SHAP")
st.info(
    "This page is a **laboratory** for experimentation and explainability.\n\n"
    "It uses two surrogate models trained on synthetic CRISI runs:\n"
    "1) A surrogate for **Resilience (0–100)**\n"
    "2) A surrogate for **CBA NPV (€)**\n\n"
    "Use this for **speed and insight**, then validate conclusions with the true CRISI engine."
)

with st.expander("How to use this page (step-by-step)", expanded=True):
    st.markdown(
        "1) Pick **Region**, **Scenario**, and (for resilience) **Year**.\n"
        "2) Optionally adjust **sandbox weights**.\n"
        "3) Check **True vs Surrogate** and the **Abs error**.\n"
        "4) Use SHAP grouped attribution first; then feature-level.\n"
        "5) If Abs error is large, treat SHAP as exploratory."
    )

tabR, tabC = st.tabs(["Resilience surrogate", "CBA surrogate (NPV)"])


# ----------------------------
# Resilience surrogate tab
# ----------------------------
with tabR:
    p = _artifact_check("resilience")
    if p is None:
        st.error(
            "Resilience surrogate artifacts missing.\n\n"
            "Run:\n"
            "- tools/generate_surrogate_data.py\n"
            "- tools/train_surrogates.py"
        )
        st.stop()

    bundle = load_surrogate_bundle(p)
    metrics = _load_metrics_json(p / "metrics.json")
    _render_model_quality(metrics, target_label="resilience_100")

    st.markdown("---")
    st.markdown("### Inputs (Resilience)")

    left, right = st.columns([1, 2])

    with left:
        region_id = st.selectbox(
            "Region (NUTS2)",
            list(NUTS2_REGIONS.keys()),
            format_func=ui_region_label,  # NEW code label, no parentheses
            key="lab_r_region",
        )

        scenario_id = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="lab_r_scenario",
        )

        st.caption(DEFAULT_SCENARIOS[scenario_id].description if scenario_id in DEFAULT_SCENARIOS else "")
        year = st.slider("Year", YEAR_MIN, YEAR_MAX, 2035, 1, key="lab_r_year")

        weight_mode, weights_payload = weights_panel(prefix="lab_r")

        cfg = ResilienceConfig()
        if weights_payload:
            d = cfg.__dict__.copy()
            for k, v in weights_payload.items():
                if k in d:
                    d[k] = float(v)
            cfg = ResilienceConfig(**d)

    with right:
        res = compute_resilience_series(region_id=region_id, scenario_id=scenario_id, cfg=cfg)
        df = res.df.copy()
        idx = (df["year"].astype(int) - int(year)).abs().idxmin()
        row = df.loc[idx]

        true_y = float(row["resilience_100"])

        # NOTE: surrogate expects INTERNAL region_id/scenario_id categories.
        raw = pd.DataFrame([{
            "region_id": region_id,
            "scenario_id": scenario_id,
            "year": int(row["year"]),
            "heat_index": float(row.get("heat_index", 0.0)),
            "drought_index": float(row.get("drought_index", 0.0)),
            "flood_risk": float(row.get("flood_risk", 0.0)),
            "tourism_share": float(row.get("tourism_share", 0.0)),
            "arrivals_index": float(row.get("arrivals_index", 0.0)),
            "income_vulnerability": float(row.get("income_vulnerability", 0.0)),
            "education": float(row.get("education", 0.0)),
            "health": float(row.get("health", 0.0)),
            "governance": float(row.get("governance", 0.0)),
            # weights
            "w_exposure": float(cfg.w_exposure),
            "w_sensitivity": float(cfg.w_sensitivity),
            "w_adaptive": float(cfg.w_adaptive),
            "w_heat": float(cfg.w_heat),
            "w_drought": float(cfg.w_drought),
            "w_flood": float(getattr(cfg, "w_flood", 0.0)),
            "w_tourism_share": float(cfg.w_tourism_share),
            "w_arrivals_index": float(cfg.w_arrivals_index),
            "w_income_vulnerability": float(cfg.w_income_vulnerability),
            "w_education": float(cfg.w_education),
            "w_health": float(cfg.w_health),
            "w_governance": float(cfg.w_governance),
        }])

        pred, shap_df = predict_and_shap(bundle, raw)

        st.markdown("### Outputs")
        c1, c2, c3 = st.columns(3)
        c1.metric("True resilience (engine)", f"{true_y:.2f}")
        c2.metric("Surrogate prediction", f"{pred:.2f}")
        c3.metric("Abs error", f"{abs(pred-true_y):.2f}")

        with st.expander("Show the exact input row used for SHAP", expanded=False):
            # show a UI-friendly region column too (no internal exposure)
            show_raw = raw.copy()
            show_raw.insert(0, "region_ui", display_code(region_id))
            st.dataframe(show_raw, use_container_width=True)

        st.markdown("---")
        _render_shap_explanation(shap_df, title="SHAP explanation – Resilience surrogate")


# ----------------------------
# CBA surrogate tab
# ----------------------------
with tabC:
    p = _artifact_check("cba")
    if p is None:
        st.error(
            "CBA surrogate artifacts missing.\n\n"
            "Run:\n"
            "- tools/generate_surrogate_data.py\n"
            "- tools/train_surrogates.py"
        )
        st.stop()

    bundle = load_surrogate_bundle(p)
    metrics = _load_metrics_json(p / "metrics.json")
    _render_model_quality(metrics, target_label="npv")

    st.markdown("---")
    st.markdown("### Inputs (CBA)")

    left, right = st.columns([1, 2])

    with left:
        region_id = st.selectbox(
            "Region (NUTS2)",
            list(NUTS2_REGIONS.keys()),
            format_func=ui_region_label,  # NEW code label, no parentheses
            key="lab_c_region",
        )
        scenario_id = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="lab_c_scenario",
        )
        st.caption(DEFAULT_SCENARIOS[scenario_id].description if scenario_id in DEFAULT_SCENARIOS else "")

        start_year = st.number_input("Start year", 2025, 2050, 2025, 1, key="lab_c_start")
        project_life = st.number_input("Project life (years)", 5, 30, 20, 1, key="lab_c_life")
        discount_rate = st.number_input("Discount rate", 0.0, 0.20, 0.04, 0.005, format="%.3f", key="lab_c_disc")

        capex = st.number_input("CAPEX (€)", 0.0, value=3_000_000.0, step=100_000.0, key="lab_c_capex")
        adaptation_share = st.slider("Adaptation share", 0.0, 1.0, 0.20, 0.01, key="lab_c_adapt")
        subsidy_rate = st.slider("Subsidy rate", 0.0, 1.0, 0.0, 0.01, key="lab_c_subsidy")

        initial_revenue = st.number_input("Initial revenue (€)", 0.0, value=200_000.0, step=50_000.0, key="lab_c_rev0")
        revenue_growth = st.number_input("Revenue growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="lab_c_revg")

        initial_opex = st.number_input("Initial OPEX (€)", 0.0, value=120_000.0, step=25_000.0, key="lab_c_opex0")
        opex_growth = st.number_input("OPEX growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="lab_c_opexg")

        tax_rate = st.slider("Tax rate", 0.0, 0.50, 0.25, 0.01, key="lab_c_tax")
        res_sens = st.slider("Resilience → Revenue sensitivity", 0.0, 2.0, 0.5, 0.05, key="lab_c_rsens")

        salvage = st.number_input("Salvage (€)", 0.0, value=0.0, step=50_000.0, key="lab_c_salvage")
        dep_years = st.number_input("Depreciation (years)", 1, 60, 20, 1, key="lab_c_dep")

        weight_mode, weights_payload = weights_panel(prefix="lab_c")
        cfg = ResilienceConfig()
        if weights_payload:
            d = cfg.__dict__.copy()
            for k, v in weights_payload.items():
                if k in d:
                    d[k] = float(v)
            cfg = ResilienceConfig(**d)

    with right:
        inputs = ProjectInputs(
            region_id=region_id,
            scenario_id=scenario_id,
            start_year=int(start_year),
            project_life=int(project_life),
            discount_rate=float(discount_rate),
            capex_total=float(capex),
            adaptation_share_of_capex=float(adaptation_share),
            subsidy_rate=float(subsidy_rate),
            salvage_value=float(salvage),
            depreciation_years=int(dep_years),
            initial_revenue=float(initial_revenue),
            revenue_growth=float(revenue_growth),
            initial_opex=float(initial_opex),
            opex_growth=float(opex_growth),
            tax_rate=float(tax_rate),
            resilience_revenue_sensitivity=float(res_sens),
        )

        true_npv = float(evaluate_project_cba(inputs, cfg).npv)

        raw = pd.DataFrame([{
            "region_id": region_id,
            "scenario_id": scenario_id,
            "start_year": int(start_year),
            "project_life": int(project_life),
            "discount_rate": float(discount_rate),
            "capex_total": float(capex),
            "adaptation_share_of_capex": float(adaptation_share),
            "subsidy_rate": float(subsidy_rate),
            "salvage_value": float(salvage),
            "depreciation_years": int(dep_years),
            "initial_revenue": float(initial_revenue),
            "revenue_growth": float(revenue_growth),
            "initial_opex": float(initial_opex),
            "opex_growth": float(opex_growth),
            "tax_rate": float(tax_rate),
            "resilience_revenue_sensitivity": float(res_sens),
            # weights
            "w_exposure": float(cfg.w_exposure),
            "w_sensitivity": float(cfg.w_sensitivity),
            "w_adaptive": float(cfg.w_adaptive),
            "w_heat": float(cfg.w_heat),
            "w_drought": float(cfg.w_drought),
            "w_flood": float(getattr(cfg, "w_flood", 0.0)),
            "w_tourism_share": float(cfg.w_tourism_share),
            "w_arrivals_index": float(cfg.w_arrivals_index),
            "w_income_vulnerability": float(cfg.w_income_vulnerability),
            "w_education": float(cfg.w_education),
            "w_health": float(cfg.w_health),
            "w_governance": float(cfg.w_governance),
        }])

        pred, shap_df = predict_and_shap(bundle, raw)

        st.markdown("### Outputs")
        c1, c2, c3 = st.columns(3)
        c1.metric("True NPV (€) (engine)", f"{true_npv:,.0f}")
        c2.metric("Surrogate NPV (€)", f"{pred:,.0f}")
        c3.metric("Abs error (€)", f"{abs(pred-true_npv):,.0f}")

        with st.expander("Show the exact input row used for SHAP", expanded=False):
            show_raw = raw.copy()
            show_raw.insert(0, "region_ui", display_code(region_id))
            st.dataframe(show_raw, use_container_width=True)

        st.markdown("---")
        _render_shap_explanation(shap_df, title="SHAP explanation – CBA NPV surrogate")
