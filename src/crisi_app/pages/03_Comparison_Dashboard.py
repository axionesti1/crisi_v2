from __future__ import annotations

import sys
import pathlib
import json
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# Ensure src/ is on sys.path
SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -----------------------------
# Regions + label mapping (UI shows NEW codes; internal uses old codes)
# -----------------------------
try:
    from crisi_app.regions import NUTS2_REGIONS, ui_region_label, CRISI_TO_GEO, GEO_TO_CRISI
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
    GEO_TO_CRISI: Dict[str, str] = {v: k for k, v in CRISI_TO_GEO.items()}

    def ui_region_label(internal_id: str) -> str:
        name = NUTS2_REGIONS.get(internal_id, internal_id)
        new_id = CRISI_TO_GEO.get(internal_id, internal_id)
        return f"{new_id} — {name}"


def display_code(internal_id: str) -> str:
    return CRISI_TO_GEO.get(internal_id, internal_id)


from crisi_app.ui_sandbox import weights_panel, wildcard_panel
from crisi_core.config import DEFAULT_SCENARIOS
from crisi_core.models import ResilienceConfig
from crisi_core.scoring import compute_resilience_series
from crisi_core.xai import resilience_xai_breakdown

st.set_page_config(page_title="CRISI v2 – Comparison Dashboard", layout="wide")
st.title("Comparison Dashboard")
st.caption(
    "Side-by-side comparisons of regions and scenarios. "
    "UI shows NUTS2 (new) codes; the engine uses internal IDs. Map joins polygons via the crosswalk."
)

YEAR_MIN, YEAR_MAX = 2025, 2055
cfg = ResilienceConfig()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_geojson() -> Optional[dict]:
    project_root = pathlib.Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "data" / "nuts2_2021_20M_4326.geojson",
        project_root / "data" / "nuts2_2021_20M_4326.geojson.json",
        project_root / "data" / "nuts2_2021_20M_4326.json",
        project_root / "data" / "nuts2_2013_20M_4326_LEVL_2.geojson",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _get_feature_id(feat: dict) -> Optional[str]:
    props = feat.get("properties", {}) or {}
    fid = props.get("id") or props.get("NUTS_ID") or props.get("nuts_id") or feat.get("id")
    if fid is None:
        return None
    return str(fid).strip().upper()


def _metric_is_positive(metric: str) -> bool:
    return metric in {"resilience_100", "adaptive"}


def _metric_label(metric: str) -> str:
    m = {
        "resilience_100": "Resilience (0–100)",
        "risk_100": "Risk (0–100)",
        "exposure": "Exposure (0–1)",
        "sensitivity": "Sensitivity (0–1)",
        "adaptive": "Adaptive capacity (0–1)",
        "heat_index": "Heat index",
        "drought_index": "Drought index",
        "flood_risk": "Flood risk",
    }
    return m.get(metric, metric)


def _make_legend(vmin: float, vmax: float, positive: bool) -> None:
    steps = 9
    colors = []
    for i in range(steps):
        t = i / (steps - 1)
        if not positive:
            t = 1.0 - t
        r = int((1.0 - t) * 220)
        g = int(t * 220)
        b = 120
        colors.append(f"rgb({r},{g},{b})")

    blocks = "".join(
        [
            f"<span style='display:inline-block;width:18px;height:12px;background:{c};margin-right:2px;'></span>"
            for c in colors
        ]
    )
    st.markdown(
        f"""
        <div style="padding:8px 10px;border:1px solid #ddd;border-radius:8px;">
            <div style="font-weight:600;margin-bottom:6px;">Legend (green = better)</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="min-width:90px;">{vmin:.3f}</div>
                <div>{blocks}</div>
                <div style="min-width:90px;text-align:right;">{vmax:.3f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _iter_lonlat(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            yield float(obj[0]), float(obj[1])
        else:
            for item in obj:
                yield from _iter_lonlat(item)


def _geojson_bounds(feature_collection: dict) -> Tuple[float, float, float, float]:
    min_lon, min_lat = 1e9, 1e9
    max_lon, max_lat = -1e9, -1e9

    for feat in feature_collection.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords is None:
            continue
        for lon, lat in _iter_lonlat(coords):
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)

    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("Could not compute bounds from GeoJSON geometry.")
    return min_lon, min_lat, max_lon, max_lat


def _zoom_from_bounds(min_lon, min_lat, max_lon, max_lat) -> float:
    span_lon = max(max_lon - min_lon, 1e-6)
    span_lat = max(max_lat - min_lat, 1e-6)
    span = max(span_lon, span_lat)
    z = np.log2(360.0 / span) - 1.6
    return float(np.clip(z, 4.2, 6.4))


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


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tabA, tabB, tabC = st.tabs(
    ["A) Two regions – same scenario", "B) One region – all scenarios", "C) Map (NUTS-2 polygons)"]
)

# ---------------------------------------------------------------------
# TAB A – Two regions, same scenario
# ---------------------------------------------------------------------
with tabA:
    left, right = st.columns([1, 2])

    with left:
        region_a = st.selectbox(
            "Region A",
            list(NUTS2_REGIONS.keys()),
            index=0,
            format_func=ui_region_label,
            key="p03_ra",
        )
        region_b = st.selectbox(
            "Region B",
            list(NUTS2_REGIONS.keys()),
            index=1,
            format_func=ui_region_label,
            key="p03_rb",
        )

        scenario_id = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="p03_s",
        )

        weight_mode, weights_payload = weights_panel(prefix="p03A")
        wildcard = None
        if weight_mode == "sandbox":
            wildcard = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="p03A")

    with right:
        res_a = compute_resilience_series(region_a, scenario_id, cfg=cfg, weights=weights_payload, wildcard=wildcard)
        res_b = compute_resilience_series(region_b, scenario_id, cfg=cfg, weights=weights_payload, wildcard=wildcard)

        df_a = res_a.df.copy()
        df_b = res_b.df.copy()

        st.subheader(
            f"Resilience (0–100) – {DEFAULT_SCENARIOS[scenario_id].name} "
            f"({display_code(region_a)} vs {display_code(region_b)})"
        )

        plot_res = pd.DataFrame(
            {
                "year": df_a["year"],
                f"A: {display_code(region_a)}": df_a["resilience_100"],
                f"B: {display_code(region_b)}": df_b["resilience_100"],
            }
        ).set_index("year")
        st.line_chart(plot_res, use_container_width=True)

        st.markdown("#### E–S–A components")
        comp = pd.DataFrame(
            {
                "year": df_a["year"],
                "A exposure": df_a["exposure"],
                "A sensitivity": df_a["sensitivity"],
                "A adaptive": df_a["adaptive"],
                "B exposure": df_b["exposure"],
                "B sensitivity": df_b["sensitivity"],
                "B adaptive": df_b["adaptive"],
            }
        ).set_index("year")
        st.line_chart(comp, use_container_width=True)

        st.markdown("---")
        st.subheader("XAI: Region A vs Region B at a chosen year")

        xai_year = st.slider(
            "Explain year",
            min_value=int(df_a["year"].min()),
            max_value=int(df_a["year"].max()),
            value=int(df_a["year"].median()),
            step=1,
            key="p03A_xai_year",
        )

        xai_a = resilience_xai_breakdown(df=df_a, config_used=res_a.config_used, year=xai_year, scale_100=True)
        xai_b = resilience_xai_breakdown(df=df_b, config_used=res_b.config_used, year=xai_year, scale_100=True)

        sum_a = _pillars_to_row(xai_a)
        sum_b = _pillars_to_row(xai_b)

        summary = pd.DataFrame(
            [
                {"Region": f"{display_code(region_a)} — {NUTS2_REGIONS[region_a]}", **sum_a},
                {"Region": f"{display_code(region_b)} — {NUTS2_REGIONS[region_b]}", **sum_b},
            ]
        )
        st.markdown("##### Pillar contributions (points out of 100)")
        st.dataframe(summary, use_container_width=True)

        delta = {k: float(sum_a[k]) - float(sum_b[k]) for k in sum_a.keys()}
        delta_df = pd.DataFrame([{"Δ (A − B)": k, "points": v} for k, v in delta.items()]).sort_values(
            "points", ascending=False
        )
        st.markdown("##### Δ Pillars (A − B, points)")
        st.dataframe(delta_df, use_container_width=True)
        st.bar_chart(delta_df.set_index("Δ (A − B)")["points"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"##### Region A drivers of low resilience ({display_code(region_a)})")
            st.dataframe(xai_a["drivers_lowR"], use_container_width=True)
        with c2:
            st.markdown(f"##### Region B drivers of low resilience ({display_code(region_b)})")
            st.dataframe(xai_b["drivers_lowR"], use_container_width=True)

# ---------------------------------------------------------------------
# TAB B – One region, all scenarios
# ---------------------------------------------------------------------
with tabB:
    left2, right2 = st.columns([1, 2])

    with left2:
        region_id = st.selectbox(
            "Region",
            list(NUTS2_REGIONS.keys()),
            format_func=ui_region_label,
            key="p03_r2",
        )
        metric = st.selectbox(
            "Metric",
            ["resilience_100", "risk_100", "exposure", "sensitivity", "adaptive"],
            format_func=_metric_label,
            key="p03_metric",
        )

        weight_mode2, weights_payload2 = weights_panel(prefix="p03B")
        wildcard2 = None
        if weight_mode2 == "sandbox":
            wildcard2 = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="p03B")

    with right2:
        frames = []
        for sid, sobj in DEFAULT_SCENARIOS.items():
            res = compute_resilience_series(region_id, sid, cfg=cfg, weights=weights_payload2, wildcard=wildcard2)
            df = res.df
            tmp = df[["year", metric]].copy()
            tmp["scenario"] = sobj.name
            frames.append(tmp)

        plot_df = pd.concat(frames, ignore_index=True)
        pivot = plot_df.pivot(index="year", columns="scenario", values=metric).sort_index()

        st.subheader(f"{_metric_label(metric)} – {display_code(region_id)} ({NUTS2_REGIONS[region_id]})")
        st.line_chart(pivot, use_container_width=True)

# ---------------------------------------------------------------------
# TAB C – Polygon map (choropleth) with crosswalk
# ---------------------------------------------------------------------
with tabC:
    st.subheader("Map: spatial overview (Greek NUTS-2 polygons)")

    geo = _load_geojson()
    if geo is None:
        st.error(
            "GeoJSON not found. Put one of these in ./data:\n"
            "- nuts2_2021_20M_4326.geojson (or .geojson.json)\n"
            "- nuts2_2013_20M_4326_LEVL_2.geojson"
        )
        st.stop()

    left3, right3 = st.columns([1, 2])

    with left3:
        scenario_id3 = st.selectbox(
            "Scenario",
            list(DEFAULT_SCENARIOS.keys()),
            format_func=lambda k: DEFAULT_SCENARIOS[k].name,
            key="p03_map_s",
        )
        year3 = st.slider("Year", YEAR_MIN, YEAR_MAX, 2035, 1, key="p03_map_year")

        metric3 = st.selectbox(
            "Metric",
            ["resilience_100", "exposure", "sensitivity", "adaptive", "heat_index", "drought_index", "flood_risk"],
            format_func=_metric_label,
            key="p03_map_metric",
        )

        weight_mode3, weights_payload3 = weights_panel(prefix="p03C")
        wildcard3 = None
        if weight_mode3 == "sandbox":
            wildcard3 = wildcard_panel(year_min=YEAR_MIN, year_max=YEAR_MAX, prefix="p03C")

        positive = _metric_is_positive(metric3)
        st.caption("Color rule: green = better, red = worse. For hazards/exposure/sensitivity, lower is better.")

    # compute values per internal model region id
    rows = []
    for model_id, name in NUTS2_REGIONS.items():
        res = compute_resilience_series(model_id, scenario_id3, cfg=cfg, weights=weights_payload3, wildcard=wildcard3)
        df = res.df.copy()
        df["year"] = df["year"].astype(int)
        idx = (df["year"] - int(year3)).abs().idxmin()
        val = float(df.loc[idx].get(metric3, np.nan))
        rows.append({"model_id": model_id, "nuts2": display_code(model_id), "region_name": name, "value": val})

    values = pd.DataFrame(rows)
    valid = values["value"].notna()
    if not valid.any():
        st.warning("No values available for mapping.")
        st.stop()

    vmin = float(values.loc[valid, "value"].min())
    vmax = float(values.loc[valid, "value"].max())
    denom = (vmax - vmin) if vmax > vmin else 1.0
    values["norm"] = (values["value"] - vmin) / denom

    val_by_model = {r["model_id"]: r for r in values.to_dict(orient="records")}

    features_out = []
    present_model_ids = set()

    for feat in geo.get("features", []):
        fid = _get_feature_id(feat)
        if not fid:
            continue

        # GeoJSON might use new codes -> map back to internal
        if fid in NUTS2_REGIONS:
            model_id = fid
        elif fid in GEO_TO_CRISI:
            model_id = GEO_TO_CRISI[fid]
        else:
            continue

        if model_id not in NUTS2_REGIONS:
            continue

        rec = val_by_model.get(model_id)
        props = feat.get("properties", {}) or {}

        if rec is None or rec["value"] is None or (isinstance(rec["value"], float) and np.isnan(rec["value"])):
            fill = [140, 140, 140, 160]
            val = None
        else:
            t = float(rec["norm"])
            if not positive:
                t = 1.0 - t
            r = int((1.0 - t) * 220)
            g = int(t * 220)
            b = 120
            fill = [r, g, b, 190]
            val = float(rec["value"])

        props["nuts2"] = display_code(model_id)  # NEW code shown in UI/tooltip
        props["region_name"] = NUTS2_REGIONS[model_id]
        props["value"] = val
        props["fill_color"] = fill
        feat["properties"] = props

        features_out.append(feat)
        present_model_ids.add(model_id)

    missing_internal = sorted(set(NUTS2_REGIONS.keys()) - present_model_ids)
    missing_new = [display_code(x) for x in missing_internal]
    st.caption(f"Polygons present: {len(present_model_ids)}/13. Missing: {missing_new if missing_new else 'none'}")

    if not features_out:
        st.error("No Greek NUTS-2 polygons matched. Your GeoJSON likely uses a different code scheme than expected.")
        st.stop()

    geo_out = {"type": "FeatureCollection", "features": features_out}

    # fit view to shown polygons
    min_lon, min_lat, max_lon, max_lat = _geojson_bounds(geo_out)
    pad_lon = (max_lon - min_lon) * 0.06
    pad_lat = (max_lat - min_lat) * 0.06
    min_lon2, max_lon2 = min_lon - pad_lon, max_lon + pad_lon
    min_lat2, max_lat2 = min_lat - pad_lat, max_lat + pad_lat
    center_lon = (min_lon2 + max_lon2) / 2.0
    center_lat = (min_lat2 + max_lat2) / 2.0
    zoom = _zoom_from_bounds(min_lon2, min_lat2, max_lon2, max_lat2)

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)

    with right3:
        _make_legend(vmin=vmin, vmax=vmax, positive=positive)

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geo_out,
            pickable=True,
            stroked=True,
            filled=True,
            auto_highlight=True,
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{region_name}</b><br/>"
                        "NUTS2: {nuts2}<br/>"
                        f"{_metric_label(metric3)}: " + "{value}",
                "style": {"backgroundColor": "white", "color": "black"},
            },
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        )

        st.pydeck_chart(deck, use_container_width=True, height=720)

        st.markdown("#### Underlying values used for the map")
        st.dataframe(values[["nuts2", "region_name", "value"]], use_container_width=True)
