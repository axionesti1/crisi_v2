from __future__ import annotations

import sys
import pathlib
from typing import Dict

import pandas as pd
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


def region_selectbox(label: str, default_code: str, key: str) -> str:
    opts = list(NUTS2_REGIONS.keys())
    idx = opts.index(default_code) if default_code in opts else 0
    return st.selectbox(label, opts, index=idx, format_func=ui_region_label, key=key)


from crisi_app.ui_sandbox import weights_panel, wildcard_panel
from crisi_core.cba import ProjectInputs, evaluate_project_cba
from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.xai import cba_npvdiff_decomposition, cba_tornado

st.set_page_config(page_title="CRISI v2 – Project CBA Dashboard", layout="wide")
st.title("Project CBA Dashboard")
st.caption(
    "CBA linked to climate resilience. UI shows NUTS2 (new) codes; engine uses internal IDs. "
    "XAI panels explain which channels and parameters drive NPV under each scenario."
)

YEAR_MIN, YEAR_MAX = 2025, 2055
BASE_CFG = ResilienceConfig()

tab_res, tab_scen, tab_adapt = st.tabs(
    ["A) Resilience preview", "B) CBA across scenarios", "C) Adaptation vs no-adaptation"]
)

# ==========================================================
# TAB A – Resilience preview
# ==========================================================
with tab_res:
    left, right = st.columns([1, 2])

    with left:
        region_id = region_selectbox("Region (NUTS-2)", default_code="EL30", key="cbaA_region")
        scenario_id = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="cbaA_scenario",
        )

        weight_mode, weights_payload = weights_panel(prefix="cbaA")
        wildcard = None
        if weight_mode == "sandbox":
            wildcard = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="cbaA")

    with right:
        cfg = BASE_CFG
        if weights_payload:
            cfg_dict = cfg.__dict__.copy()
            for k, v in weights_payload.items():
                if k in cfg_dict:
                    cfg_dict[k] = float(v)
            cfg = ResilienceConfig(**cfg_dict)

        res = compute_resilience_series(region_id, scenario_id, cfg=cfg, wildcard=wildcard).df

        st.subheader(f"Resilience preview — {display_code(region_id)} ({NUTS2_REGIONS[region_id]})")
        st.line_chart(res.set_index("year")[["resilience_100", "risk_100"]], use_container_width=True)

        st.markdown("#### E–S–A components")
        st.line_chart(res.set_index("year")[["exposure", "sensitivity", "adaptive"]], use_container_width=True)

# ==========================================================
# TAB B – CBA across scenarios + XAI
# ==========================================================
with tab_scen:
    leftB, rightB = st.columns([1, 2])

    with leftB:
        region_id_B = region_selectbox("Region (NUTS-2)", default_code="EL30", key="cbaB_region")

        start_year = st.number_input("Start year", YEAR_MIN, YEAR_MAX, 2025, 1, key="cbaB_start")
        project_life = st.number_input("Project life (years)", 5, 40, 20, 1, key="cbaB_life")
        discount_rate = st.number_input("Discount rate", 0.0, 0.20, 0.04, 0.005, format="%.3f", key="cbaB_disc")

        capex = st.number_input("Total CAPEX (€)", 0.0, value=300_000.0, step=100_000.0, key="cbaB_capex")
        adaptation_share = st.slider("Adaptation share of CAPEX", 0.0, 1.0, 0.20, 0.01, key="cbaB_adshare")
        subsidy_rate = st.slider("Subsidy rate on adaptation CAPEX", 0.0, 1.0, 0.0, 0.01, key="cbaB_subsidy")

        initial_revenue = st.number_input("Initial annual revenue (€)", 0.0, 1_000_000.0, 50_000.0, key="cbaB_rev0")
        revenue_growth = st.number_input("Annual revenue growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="cbaB_revg")

        initial_opex = st.number_input("Initial annual OPEX (€)", 0.0, 400_000.0, 25_000.0, key="cbaB_opex0")
        opex_growth = st.number_input("Annual OPEX growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="cbaB_opexg")

        tax_rate = st.slider("Corporate tax rate", 0.0, 0.50, 0.25, 0.01, key="cbaB_tax")
        res_sens = st.slider("Resilience → revenue sensitivity", 0.0, 2.0, 0.5, 0.05, key="cbaB_res_sens")

        salvage_value = st.number_input("Salvage value at end (€)", 0.0, value=0.0, step=50_000.0, key="cbaB_salvage")
        dep_years = st.number_input("Depreciation period (years)", 1, 60, 20, 1, key="cbaB_depyears")

        weight_mode_B, weights_payload_B = weights_panel(prefix="cbaB")
        wildcard_B = None
        if weight_mode_B == "sandbox":
            wildcard_B = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="cbaB")

        tornado_step = st.slider("Tornado step (relative)", 0.02, 0.30, 0.10, 0.01, key="cbaB_tornado_step")

    with rightB:
        cfg_B = BASE_CFG
        if weights_payload_B:
            cfg_dict = cfg_B.__dict__.copy()
            for k, v in weights_payload_B.items():
                if k in cfg_dict:
                    cfg_dict[k] = float(v)
            cfg_B = ResilienceConfig(**cfg_dict)

        results = []
        for sid, sobj in DEFAULT_SCENARIOS.items():
            inputs = ProjectInputs(
                region_id=region_id_B,
                scenario_id=sid,
                start_year=int(start_year),
                project_life=int(project_life),
                discount_rate=float(discount_rate),
                capex_total=float(capex),
                adaptation_share_of_capex=float(adaptation_share),
                subsidy_rate=float(subsidy_rate),
                salvage_value=float(salvage_value),
                depreciation_years=int(dep_years),
                initial_revenue=float(initial_revenue),
                revenue_growth=float(revenue_growth),
                initial_opex=float(initial_opex),
                opex_growth=float(opex_growth),
                tax_rate=float(tax_rate),
                resilience_revenue_sensitivity=float(res_sens),
            )
            metrics = evaluate_project_cba(inputs, cfg_B, wildcard=wildcard_B)
            results.append({"Scenario": sobj.name, "NPV (€)": metrics.npv})

        res_df = pd.DataFrame(results).sort_values("NPV (€)", ascending=False)
        st.subheader(f"NPV by scenario — {display_code(region_id_B)} ({NUTS2_REGIONS[region_id_B]})")
        st.dataframe(res_df, use_container_width=True)
        st.bar_chart(res_df.set_index("Scenario")["NPV (€)"])

        st.markdown("---")
        st.subheader("XAI: NPV drivers (pick a scenario)")
        pick_sid = st.selectbox(
            "Scenario for XAI",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="cbaB_xai_scenario",
        )

        xai_inputs = ProjectInputs(
            region_id=region_id_B,
            scenario_id=pick_sid,
            start_year=int(start_year),
            project_life=int(project_life),
            discount_rate=float(discount_rate),
            capex_total=float(capex),
            adaptation_share_of_capex=float(adaptation_share),
            subsidy_rate=float(subsidy_rate),
            salvage_value=float(salvage_value),
            depreciation_years=int(dep_years),
            initial_revenue=float(initial_revenue),
            revenue_growth=float(revenue_growth),
            initial_opex=float(initial_opex),
            opex_growth=float(opex_growth),
            tax_rate=float(tax_rate),
            resilience_revenue_sensitivity=float(res_sens),
        )

        decomp = cba_npvdiff_decomposition(xai_inputs, cfg_B, wildcard=wildcard_B)
        st.markdown("##### Channel decomposition (NPV deltas)")
        st.dataframe(decomp, use_container_width=True)

        tornado = cba_tornado(xai_inputs, cfg_B, wildcard=wildcard_B, rel_step=float(tornado_step))
        st.markdown("##### Tornado sensitivity (largest impacts first)")
        st.dataframe(tornado, use_container_width=True)
        st.bar_chart(tornado.set_index("Parameter")["Impact range (abs)"])

# ==========================================================
# TAB C – Adaptation vs none + XAI
# ==========================================================
with tab_adapt:
    leftC, rightC = st.columns([1, 2])

    with leftC:
        region_id_C = region_selectbox("Region (NUTS-2)", default_code="EL30", key="cbaC_region")
        scenario_id_C = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="cbaC_scenario",
        )

        start_year_C = st.number_input("Start year", YEAR_MIN, YEAR_MAX, 2025, 1, key="cbaC_start")
        project_life_C = st.number_input("Project life (years)", 5, 40, 20, 1, key="cbaC_life")
        discount_rate_C = st.number_input("Discount rate", 0.0, 0.20, 0.04, 0.005, format="%.3f", key="cbaC_disc")

        capex_C = st.number_input("Total CAPEX (€)", 0.0, value=300_000.0, step=100_000.0, key="cbaC_capex")
        subsidy_rate_C = st.slider("Subsidy rate on adaptation CAPEX", 0.0, 1.0, 0.0, 0.01, key="cbaC_subsidy")

        st.markdown("**Case A – No adaptation**")
        adapt_A = st.slider("Adaptation share (Case A)", 0.0, 1.0, 0.0, 0.01, key="cbaC_adaptA")

        st.markdown("**Case B – With adaptation**")
        adapt_B = st.slider("Adaptation share (Case B)", 0.0, 1.0, 0.30, 0.01, key="cbaC_adaptB")

        initial_revenue_C = st.number_input("Initial annual revenue (€)", 0.0, 1_000_000.0, 50_000.0, key="cbaC_rev0")
        revenue_growth_C = st.number_input("Annual revenue growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="cbaC_revg")

        initial_opex_C = st.number_input("Initial annual OPEX (€)", 0.0, 400_000.0, 25_000.0, key="cbaC_opex0")
        opex_growth_C = st.number_input("Annual OPEX growth", -0.20, 0.20, 0.02, 0.005, format="%.3f", key="cbaC_opexg")

        tax_rate_C = st.slider("Corporate tax rate", 0.0, 0.50, 0.25, 0.01, key="cbaC_tax")
        res_sens_C = st.slider("Resilience → revenue sensitivity", 0.0, 2.0, 0.5, 0.05, key="cbaC_res_sens")

        salvage_value_C = st.number_input("Salvage value at end (€)", 0.0, value=0.0, step=50_000.0, key="cbaC_salvage")
        dep_years_C = st.number_input("Depreciation period (years)", 1, 60, 20, 1, key="cbaC_depyears")

        weight_mode_C, weights_payload_C = weights_panel(prefix="cbaC")
        wildcard_C = None
        if weight_mode_C == "sandbox":
            wildcard_C = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="cbaC")

        tornado_step_C = st.slider("Tornado step (relative)", 0.02, 0.30, 0.10, 0.01, key="cbaC_tornado_step")

    with rightC:
        cfg_C = BASE_CFG
        if weights_payload_C:
            cfg_dict = cfg_C.__dict__.copy()
            for k, v in weights_payload_C.items():
                if k in cfg_dict:
                    cfg_dict[k] = float(v)
            cfg_C = ResilienceConfig(**cfg_dict)

        inputs_A = ProjectInputs(
            region_id=region_id_C,
            scenario_id=scenario_id_C,
            start_year=int(start_year_C),
            project_life=int(project_life_C),
            discount_rate=float(discount_rate_C),
            capex_total=float(capex_C),
            adaptation_share_of_capex=float(adapt_A),
            subsidy_rate=float(subsidy_rate_C),
            salvage_value=float(salvage_value_C),
            depreciation_years=int(dep_years_C),
            initial_revenue=float(initial_revenue_C),
            revenue_growth=float(revenue_growth_C),
            initial_opex=float(initial_opex_C),
            opex_growth=float(opex_growth_C),
            tax_rate=float(tax_rate_C),
            resilience_revenue_sensitivity=float(res_sens_C),
        )

        inputs_B = ProjectInputs(
            region_id=region_id_C,
            scenario_id=scenario_id_C,
            start_year=int(start_year_C),
            project_life=int(project_life_C),
            discount_rate=float(discount_rate_C),
            capex_total=float(capex_C),
            adaptation_share_of_capex=float(adapt_B),
            subsidy_rate=float(subsidy_rate_C),
            salvage_value=float(salvage_value_C),
            depreciation_years=int(dep_years_C),
            initial_revenue=float(initial_revenue_C),
            revenue_growth=float(revenue_growth_C),
            initial_opex=float(initial_opex_C),
            opex_growth=float(opex_growth_C),
            tax_rate=float(tax_rate_C),
            resilience_revenue_sensitivity=float(res_sens_C),
        )

        mA = evaluate_project_cba(inputs_A, cfg_C, wildcard=wildcard_C)
        mB = evaluate_project_cba(inputs_B, cfg_C, wildcard=wildcard_C)

        st.subheader(f"Adaptation comparison — {display_code(region_id_C)} ({NUTS2_REGIONS[region_id_C]})")

        c1, c2 = st.columns(2)
        c1.metric("NPV (Case A)", f"{mA.npv:,.0f} €")
        c2.metric("NPV (Case B)", f"{mB.npv:,.0f} €")

        st.markdown("#### Net cashflow comparison")
        st.line_chart(
            {
                "Case A": mA.cashflow_df.set_index("year")["net_cashflow"],
                "Case B": mB.cashflow_df.set_index("year")["net_cashflow"],
            },
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("XAI: NPV drivers for Case B (current settings)")
        decompC = cba_npvdiff_decomposition(inputs_B, cfg_C, wildcard=wildcard_C)
        st.dataframe(decompC, use_container_width=True)

        tornadoC = cba_tornado(inputs_B, cfg_C, wildcard=wildcard_C, rel_step=float(tornado_step_C))
        st.dataframe(tornadoC, use_container_width=True)
        st.bar_chart(tornadoC.set_index("Parameter")["Impact range (abs)"])
