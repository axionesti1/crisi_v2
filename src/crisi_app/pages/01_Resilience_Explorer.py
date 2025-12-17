from __future__ import annotations

import sys
import pathlib
from typing import Dict

import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path
SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -----------------------------
# Regions + label mapping (new-code UI, old-code internal)
# -----------------------------
try:
    from crisi_app.regions import NUTS2_REGIONS, ui_region_label, CRISI_TO_GEO
except Exception:
    # Fallback: page still runs even if regions.py is not yet updated
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
        # Display ONLY the new code + name (user-facing)
        name = NUTS2_REGIONS.get(internal_id, internal_id)
        new_id = CRISI_TO_GEO.get(internal_id, internal_id)
        return f"{new_id} — {name}"

from crisi_app.ui_sandbox import weights_panel, wildcard_panel
from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.xai import resilience_xai_breakdown

st.set_page_config(page_title="CRISI v2 – Resilience Explorer", layout="wide")

st.title("Resilience Explorer (CRISI v2)")
st.caption(
    "Explore resilience trajectories by region and scenario. "
    "UI shows the latest Eurostat/GISCO NUTS2 codes; the engine runs on internal IDs."
)

YEAR_MIN, YEAR_MAX = 2025, 2055

left, right = st.columns([1, 2])

with left:
    st.subheader("Controls")

    # Selectbox returns internal_id, but displays NEW code label
    region_id = st.selectbox(
        "NUTS2 region (Eurostat code)",
        list(NUTS2_REGIONS.keys()),
        format_func=ui_region_label,
        key="p01_region",
    )

    scenario_id = st.selectbox(
        "Scenario",
        list(DEFAULT_SCENARIOS.keys()),
        format_func=lambda k: DEFAULT_SCENARIOS[k].name,
        key="p01_scenario",
    )

    weight_mode, weights_payload = weights_panel(prefix="p01")
    wildcard = None
    if weight_mode == "sandbox":
        wildcard = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="p01")

with right:
    st.subheader("Resilience & components")

    cfg = ResilienceConfig()
    res = compute_resilience_series(
        region_id=region_id,
        scenario_id=scenario_id,
        cfg=cfg,
        weights=weights_payload,
        wildcard=wildcard,
    )
    df = res.df.copy()

    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Resilience (final)", f"{last['resilience_100']:.1f}")
    c2.metric("Exposure (final)", f"{last['exposure']:.3f}")
    c3.metric("Risk (final)", f"{last['risk_100']:.1f}")

    st.markdown("#### Resilience vs Risk (0–100)")
    st.line_chart(df.set_index("year")[["resilience_100", "risk_100"]], use_container_width=True)

    st.markdown("#### E–S–A pillars")
    st.line_chart(df.set_index("year")[["exposure", "sensitivity", "adaptive"]], use_container_width=True)

    st.markdown("#### Hazard indicators")
    st.line_chart(df.set_index("year")[["heat_index", "drought_index", "flood_risk"]], use_container_width=True)

    st.markdown("#### Sensitivity indicators")
    st.line_chart(
        df.set_index("year")[["tourism_share", "arrivals_index", "income_vulnerability"]],
        use_container_width=True,
    )

    st.markdown("#### Adaptive indicators")
    st.line_chart(df.set_index("year")[["education", "health", "governance"]], use_container_width=True)

    # -----------------------------
    # XAI SECTION
    # -----------------------------
    st.markdown("---")
    st.subheader("XAI: Why is the score what it is?")

    xai_year = st.slider(
        "Explain year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(df["year"].median()),
        step=1,
        key="p01_xai_year",
    )

    xai = resilience_xai_breakdown(
        df=df,
        config_used=res.config_used,
        year=xai_year,
        scale_100=True,
    )

    st.markdown("##### Pillar decomposition (contribution to R, points out of 100)")
    st.dataframe(xai["pillars"], use_container_width=True)

    pillars_bar = xai["pillars"].copy()
    pillars_bar = pillars_bar[
        pillars_bar["component"].isin(
            ["Lack of exposure (1−E)", "Lack of sensitivity (1−S)", "Adaptive capacity (A)"]
        )
    ]
    st.bar_chart(pillars_bar.set_index("component")["contribution"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Exposure (hazards) contribution")
        st.dataframe(xai["exposure"], use_container_width=True)
        st.bar_chart(xai["exposure"].set_index("indicator")["contribution_to_R"])
    with col2:
        st.markdown("##### Sensitivity contribution")
        st.dataframe(xai["sensitivity"], use_container_width=True)
        st.bar_chart(xai["sensitivity"].set_index("indicator")["contribution_to_R"])

    st.markdown("##### Adaptive capacity contribution")
    st.dataframe(xai["adaptive"], use_container_width=True)
    st.bar_chart(xai["adaptive"].set_index("indicator")["contribution_to_R"])

    st.markdown("##### Drivers of low resilience (largest negative pressures, points out of 100)")
    st.dataframe(xai["drivers_lowR"], use_container_width=True)
    st.bar_chart(xai["drivers_lowR"].set_index("driver")["impact"])

    # -----------------------------
    # Settings summary (show NEW code)
    # -----------------------------
    display_code = CRISI_TO_GEO.get(region_id, region_id)
    with st.expander("Assumptions & settings", expanded=False):
        st.write(f"- Region (UI code): `{display_code}` — {NUTS2_REGIONS.get(region_id, region_id)}")
        st.write(f"- Scenario: `{scenario_id}` — {DEFAULT_SCENARIOS[scenario_id].name}")
        st.write(f"- Weight mode: `{weight_mode}`")
        if weights_payload:
            st.json(weights_payload)
        if wildcard:
            st.json(wildcard)

    st.markdown("#### Data table")
    st.dataframe(df, use_container_width=True)
