from __future__ import annotations

import sys
import pathlib
from typing import Dict, Any

import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path
SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -----------------------------
# Regions + label mapping (UI shows NEW codes; internal uses old codes)
# -----------------------------
try:
    from crisi_app.regions import NUTS2_REGIONS, ui_region_label, CRISI_TO_GEO
except Exception:
    # Fallback: page still runs even if regions.py not updated yet
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

from crisi_app.ui_sandbox import weights_panel, wildcard_panel
from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.xai import resilience_xai_breakdown

st.set_page_config(page_title="CRISI v2 – Scenario Comparison", layout="wide")

st.title("Scenario Comparison (5 scenarios)")
st.caption(
    "Compare how one region behaves under all CRISI scenarios. "
    "Includes deterministic XAI: pillar and indicator contributions consistent with the CRISI formula."
)

YEAR_MIN, YEAR_MAX = 2025, 2055

left, right = st.columns([1, 2])


def _metric_label(metric: str) -> str:
    labels = {
        "resilience_100": "Resilience (0–100)",
        "risk_100": "Risk (0–100)",
        "exposure": "Exposure (0–1)",
        "sensitivity": "Sensitivity (0–1)",
        "adaptive": "Adaptive capacity (0–1)",
        "heat_index": "Heat index",
        "drought_index": "Drought index",
        "flood_risk": "Flood risk",
    }
    return labels.get(metric, metric)


def _pillars_to_row(xai: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    p = xai["pillars"].copy()

    def _get(comp: str, col: str = "contribution") -> float:
        r = p.loc[p["component"] == comp, col]
        return float(r.iloc[0]) if len(r) else float("nan")

    return {
        "Lack of exposure (pts)": _get("Lack of exposure (1−E)"),
        "Lack of sensitivity (pts)": _get("Lack of sensitivity (1−S)"),
        "Adaptive capacity (pts)": _get("Adaptive capacity (A)"),
        "Resilience (pts)": _get("Total resilience (R)"),
    }


with left:
    region_id = st.selectbox(
        "NUTS-2 region",
        list(NUTS2_REGIONS.keys()),
        format_func=ui_region_label,  # NEW codes shown, no parentheses
        key="p02_region",
    )

    metric = st.selectbox(
        "Metric to compare",
        [
            "resilience_100",
            "risk_100",
            "exposure",
            "sensitivity",
            "adaptive",
            "heat_index",
            "drought_index",
            "flood_risk",
        ],
        format_func=_metric_label,
        key="p02_metric",
    )

    weight_mode, weights_payload = weights_panel(prefix="p02")
    wildcard = None
    if weight_mode == "sandbox":
        wildcard = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="p02")


with right:
    cfg = ResilienceConfig()
    scenario_results: Dict[str, Any] = {}

    frames = []
    for sid, sobj in DEFAULT_SCENARIOS.items():
        res = compute_resilience_series(
            region_id=region_id,
            scenario_id=sid,
            cfg=cfg,
            weights=weights_payload,
            wildcard=wildcard,
        )
        scenario_results[sid] = res

        tmp = res.df[["year", metric]].copy()
        tmp["scenario"] = sobj.name
        frames.append(tmp)

    plot_df = pd.concat(frames, ignore_index=True)
    pivot = plot_df.pivot(index="year", columns="scenario", values=metric).sort_index()

    display_code = CRISI_TO_GEO.get(region_id, region_id)
    region_name = NUTS2_REGIONS.get(region_id, region_id)

    st.subheader(f"{_metric_label(metric)} – {display_code} ({region_name})")
    st.line_chart(pivot, use_container_width=True)

    # -----------------------------
    # XAI: scenario attribution at year
    # -----------------------------
    st.markdown("---")
    st.subheader("XAI: Scenario attribution at a chosen year")

    xai_year = st.slider(
        "Explain year",
        min_value=int(plot_df["year"].min()),
        max_value=int(plot_df["year"].max()),
        value=int(plot_df["year"].median()),
        step=1,
        key="p02_xai_year",
    )

    baseline_sid = st.selectbox(
        "Baseline scenario for deltas",
        list(DEFAULT_SCENARIOS.keys()),
        index=list(DEFAULT_SCENARIOS.keys()).index("business_as_usual")
        if "business_as_usual" in DEFAULT_SCENARIOS
        else 0,
        format_func=lambda k: DEFAULT_SCENARIOS[k].name,
        key="p02_xai_baseline",
    )

    focus_sid = st.selectbox(
        "Drill-down scenario (full decomposition)",
        list(DEFAULT_SCENARIOS.keys()),
        format_func=lambda k: DEFAULT_SCENARIOS[k].name,
        key="p02_xai_focus",
    )

    # Compute XAI for all scenarios at the selected year
    xai_by_sid: Dict[str, Dict[str, pd.DataFrame]] = {}
    summary_rows = []

    for sid, res in scenario_results.items():
        df_s = res.df.copy()
        xai = resilience_xai_breakdown(
            df=df_s,
            config_used=res.config_used,
            year=xai_year,
            scale_100=True,
        )
        xai_by_sid[sid] = xai

        row = {"Scenario": DEFAULT_SCENARIOS[sid].name}
        row.update(_pillars_to_row(xai))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("Resilience (pts)", ascending=False)

    st.markdown("#### Pillar contributions (points out of 100)")
    st.dataframe(summary, use_container_width=True)

    # Delta vs baseline
    base_row = summary.loc[summary["Scenario"] == DEFAULT_SCENARIOS[baseline_sid].name].iloc[0]
    delta = summary.copy()
    for col in ["Lack of exposure (pts)", "Lack of sensitivity (pts)", "Adaptive capacity (pts)", "Resilience (pts)"]:
        delta[col] = delta[col] - float(base_row[col])
    delta = delta.sort_values("Resilience (pts)", ascending=False)

    st.markdown(f"#### Δ vs baseline: {DEFAULT_SCENARIOS[baseline_sid].name} (points)")
    st.dataframe(delta, use_container_width=True)

    st.markdown("#### Δ Resilience (pts) vs baseline (bar)")
    st.bar_chart(delta.set_index("Scenario")["Resilience (pts)"])

    # Drill-down
    st.markdown("---")
    st.subheader(f"XAI drill-down: {DEFAULT_SCENARIOS[focus_sid].name} – year {xai_year}")

    x = xai_by_sid[focus_sid]

    st.markdown("##### Pillar decomposition")
    st.dataframe(x["pillars"], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Exposure (hazards) contribution")
        st.dataframe(x["exposure"], use_container_width=True)
        st.bar_chart(x["exposure"].set_index("indicator")["contribution_to_R"])
    with c2:
        st.markdown("##### Sensitivity contribution")
        st.dataframe(x["sensitivity"], use_container_width=True)
        st.bar_chart(x["sensitivity"].set_index("indicator")["contribution_to_R"])

    st.markdown("##### Adaptive capacity contribution")
    st.dataframe(x["adaptive"], use_container_width=True)
    st.bar_chart(x["adaptive"].set_index("indicator")["contribution_to_R"])

    st.markdown("##### Drivers of low resilience (largest negative pressures first)")
    st.dataframe(x["drivers_lowR"], use_container_width=True)
    st.bar_chart(x["drivers_lowR"].set_index("driver")["impact"])

    with st.expander("Assumptions & settings", expanded=False):
        st.write(f"- Region: `{display_code}` — {region_name}")
        st.write(f"- Metric: `{metric}`")
        st.write(f"- Weight mode: `{weight_mode}`")
        st.write(f"- Baseline scenario: `{baseline_sid}` — {DEFAULT_SCENARIOS[baseline_sid].name}")
        st.write(f"- Focus scenario: `{focus_sid}` — {DEFAULT_SCENARIOS[focus_sid].name}")
        if weights_payload:
            st.json(weights_payload)
        if wildcard:
            st.json(wildcard)
