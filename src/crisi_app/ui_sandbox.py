from __future__ import annotations

import sys
import pathlib
import json
import secrets
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

# Ensure src/ is on sys.path
SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from crisi_core.config import DEFAULT_CONFIG


def _norm3(a: float, b: float, c: float) -> Tuple[float, float, float]:
    a = max(0.0, float(a))
    b = max(0.0, float(b))
    c = max(0.0, float(c))
    s = a + b + c
    if s <= 1e-12:
        return 1 / 3, 1 / 3, 1 / 3
    return a / s, b / s, c / s


def _normN(vals: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in vals.items()}
    s = sum(clean.values())
    if s <= 1e-12:
        n = len(clean) if clean else 1
        return {k: 1.0 / n for k in clean}
    return {k: v / s for k, v in clean.items()}


def weights_panel(prefix: str = "global") -> tuple[str, Optional[dict]]:
    """
    Shared weights UI.

    Returns:
      weight_mode: "policy" or "sandbox"
      weights_payload: dict or None (None => use defaults in cfg)
    """
    st.markdown("### Weights")
    weight_mode = st.radio(
        "Mode",
        ["Policy defaults", "Custom (sandbox)"],
        horizontal=True,
        key=f"{prefix}_weight_mode",
    )

    if weight_mode == "Policy defaults":
        return "policy", None

    st.caption(
        "Sandbox weights are for sensitivity analysis and transparency – "
        "they do NOT represent official policy weights."
    )

    cfg = asdict(DEFAULT_CONFIG)

    # Pillar weights
    st.markdown("**Pillar weights (E, S, A)**")
    c1, c2, c3 = st.columns(3)
    wE = c1.slider(
        "Exposure (E)",
        0.0,
        1.0,
        float(cfg.get("w_exposure", 1 / 3)),
        0.01,
        key=f"{prefix}_wE",
    )
    wS = c2.slider(
        "Sensitivity (S)",
        0.0,
        1.0,
        float(cfg.get("w_sensitivity", 1 / 3)),
        0.01,
        key=f"{prefix}_wS",
    )
    wA = c3.slider(
        "Adaptive capacity (A)",
        0.0,
        1.0,
        float(cfg.get("w_adaptive", 1 / 3)),
        0.01,
        key=f"{prefix}_wA",
    )
    wE, wS, wA = _norm3(wE, wS, wA)

    # Exposure sub-weights
    with st.expander("Exposure breakdown (hazards)", expanded=False):
        e_defaults = {
            "w_heat": float(cfg.get("w_heat", 0.55)),
            "w_drought": float(cfg.get("w_drought", 0.30)),
            "w_flood": float(cfg.get("w_flood", 0.15)),
        }
        ce1, ce2, ce3 = st.columns(3)
        e_defaults["w_heat"] = ce1.slider(
            "Heat", 0.0, 1.0, e_defaults["w_heat"], 0.01, key=f"{prefix}_w_heat"
        )
        e_defaults["w_drought"] = ce2.slider(
            "Drought", 0.0, 1.0, e_defaults["w_drought"], 0.01, key=f"{prefix}_w_drought"
        )
        e_defaults["w_flood"] = ce3.slider(
            "Flood", 0.0, 1.0, e_defaults["w_flood"], 0.01, key=f"{prefix}_w_flood"
        )
        e_norm = _normN(e_defaults)

    # Sensitivity sub-weights
    with st.expander("Sensitivity breakdown (tourism dependence)", expanded=False):
        s_defaults = {
            "w_tourism_share": float(cfg.get("w_tourism_share", 0.50)),
            "w_arrivals_index": float(cfg.get("w_arrivals_index", 0.30)),
            "w_income_vulnerability": float(cfg.get("w_income_vulnerability", 0.20)),
        }
        cs1, cs2, cs3 = st.columns(3)
        s_defaults["w_tourism_share"] = cs1.slider(
            "Tourism share",
            0.0,
            1.0,
            s_defaults["w_tourism_share"],
            0.01,
            key=f"{prefix}_w_tshare",
        )
        s_defaults["w_arrivals_index"] = cs2.slider(
            "Arrivals dynamics",
            0.0,
            1.0,
            s_defaults["w_arrivals_index"],
            0.01,
            key=f"{prefix}_w_arrivals",
        )
        s_defaults["w_income_vulnerability"] = cs3.slider(
            "Income vulnerability",
            0.0,
            1.0,
            s_defaults["w_income_vulnerability"],
            0.01,
            key=f"{prefix}_w_incomev",
        )
        s_norm = _normN(s_defaults)

    # Adaptive sub-weights
    with st.expander("Adaptive capacity breakdown", expanded=False):
        a_defaults = {
            "w_education": float(cfg.get("w_education", 0.34)),
            "w_health": float(cfg.get("w_health", 0.33)),
            "w_governance": float(cfg.get("w_governance", 0.33)),
        }
        ca1, ca2, ca3 = st.columns(3)
        a_defaults["w_education"] = ca1.slider(
            "Education",
            0.0,
            1.0,
            a_defaults["w_education"],
            0.01,
            key=f"{prefix}_w_edu",
        )
        a_defaults["w_health"] = ca2.slider(
            "Health",
            0.0,
            1.0,
            a_defaults["w_health"],
            0.01,
            key=f"{prefix}_w_health",
        )
        a_defaults["w_governance"] = ca3.slider(
            "Governance",
            0.0,
            1.0,
            a_defaults["w_governance"],
            0.01,
            key=f"{prefix}_w_gov",
        )
        a_norm = _normN(a_defaults)

    weights_payload = {
        "w_exposure": wE,
        "w_sensitivity": wS,
        "w_adaptive": wA,
        **e_norm,
        **s_norm,
        **a_norm,
    }
    return "sandbox", weights_payload


def wildcard_panel(
    year_min: int,
    year_max: int,
    prefix: str = "global",
) -> Optional[dict]:
    """
    Shared wildcard UI. Stores event in st.session_state["wildcard_event_<prefix>"].
    """
    key_base = f"wildcard_event_{prefix}"
    current = st.session_state.get(key_base)

    st.markdown("### Wildcard event (Sandbox)")
    enabled = st.toggle(
        "Enable wildcard event",
        value=bool(current),
        key=f"{prefix}_wc_on",
    )

    if not enabled:
        st.session_state[key_base] = None
        return None

    c1, c2 = st.columns(2)
    direction = c1.selectbox(
        "Direction",
        ["Bad shock", "Good shock"],
        key=f"{prefix}_wc_dir",
    )
    d = "bad" if direction.startswith("Bad") else "good"

    default_year = current["year"] if current else (year_min + year_max) // 2
    year0 = c2.slider(
        "Event year",
        year_min,
        year_max,
        value=max(year_min, min(year_max, default_year)),
        key=f"{prefix}_wc_year0",
    )

    c3, c4 = st.columns(2)
    alpha = c3.slider(
        "Power-law tail (alpha)",
        1.2,
        3.0,
        value=(current["alpha"] if current else 1.7),
        step=0.1,
        key=f"{prefix}_wc_alpha",
    )
    decay = c4.slider(
        "Decay exponent (k)",
        0.0,
        3.0,
        value=(current["decay"] if current else 1.2),
        step=0.1,
        key=f"{prefix}_wc_decay",
    )

    scale = st.slider(
        "Scale (baseline shock size)",
        0.00,
        0.60,
        value=(current["scale"] if current else 0.18),
        step=0.01,
        key=f"{prefix}_wc_scale",
    )
    cap = st.slider(
        "Max shock cap",
        0.05,
        1.00,
        value=(current["cap"] if current else 0.65),
        step=0.01,
        key=f"{prefix}_wc_cap",
    )

    targets = st.multiselect(
        "Affects",
        ["Exposure (hazards)", "Sensitivity", "Adaptive capacity"],
        default=(current["targets"] if current else ["Exposure (hazards)", "Sensitivity", "Adaptive capacity"]),
        key=f"{prefix}_wc_targets",
    )

    colb1, colb2, colb3 = st.columns([1, 1, 2])

    if colb1.button("Draw wildcard", key=f"{prefix}_wc_draw"):
        seed = secrets.randbits(32)
        rng = np.random.default_rng(seed)
        raw = rng.pareto(alpha) + 1.0  # >= 1
        magnitude = min(float(cap), float(scale) * float(raw))
        event = {
            "enabled": True,
            "direction": d,
            "year": int(year0),
            "alpha": float(alpha),
            "decay": float(decay),
            "scale": float(scale),
            "cap": float(cap),
            "seed": int(seed),
            "magnitude": float(magnitude),
            "targets": list(targets),
        }
        st.session_state[key_base] = event
        current = event

    if colb2.button("Clear wildcard", key=f"{prefix}_wc_clear"):
        st.session_state[key_base] = None
        current = None

    if current:
        st.success(
            f"Wildcard active: {current['direction'].upper()} | "
            f"year={current['year']} | magnitude={current['magnitude']:.3f} | "
            f"decay k={current['decay']:.2f}"
        )
        with st.expander("Wildcard JSON (for auditability)", expanded=False):
            st.code(json.dumps(current, indent=2), language="json")

    return current
