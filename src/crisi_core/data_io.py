# src/crisi_core/data_io.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

# Project root = .../crisi_v2 (this file is .../crisi_v2/src/crisi_core/data_io.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

HAZARD_PARQUET = DATA_DIR / "hazard.parquet"
SOCIO_PARQUET = DATA_DIR / "socio.parquet"
FLOOD_PARQUET = DATA_DIR / "flood_risk.parquet"

# Your mapping old -> new (GeoJSON uses new). We need new -> old for data lookup.
_OLD_TO_NEW = {
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
_NEW_TO_OLD = {v: k for k, v in _OLD_TO_NEW.items()}


def canonical_region_id(region_id: str) -> str:
    """Accept either old (EL11...) or new (EL51...) codes; use old for internal data tables."""
    return _NEW_TO_OLD.get(region_id, region_id)


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset: {path}. "
            f"Expected under: {DATA_DIR}"
        )


@lru_cache(maxsize=1)
def _hazard_table() -> pd.DataFrame:
    _require_exists(HAZARD_PARQUET)
    df = pd.read_parquet(HAZARD_PARQUET)

    # tolerate alternative naming from earlier scripts
    if "heat_index" not in df.columns and "heat_factor" in df.columns:
        df["heat_index"] = df["heat_factor"]
    if "drought_index" not in df.columns and "drought_factor" in df.columns:
        df["drought_index"] = df["drought_factor"]

    needed = {"region_id", "scenario_id", "year", "heat_index", "drought_index"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"hazard.parquet missing columns: {missing}. Found: {list(df.columns)}")

    return df


@lru_cache(maxsize=1)
def _socio_table() -> pd.DataFrame:
    _require_exists(SOCIO_PARQUET)
    df = pd.read_parquet(SOCIO_PARQUET)

    # minimal required fields (year is optional; if absent we treat as baseline only)
    needed = {"region_id", "tourism_share", "arrivals_index", "gdp_index", "education_index", "health_index", "governance_index"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"socio.parquet missing columns: {missing}. Found: {list(df.columns)}")

    return df


@lru_cache(maxsize=1)
def _flood_table() -> pd.DataFrame:
    _require_exists(FLOOD_PARQUET)
    df = pd.read_parquet(FLOOD_PARQUET)

    # tolerate naming differences
    if "flood_risk_index" not in df.columns:
        for alt in ["flood_index", "flood_risk", "flood"]:
            if alt in df.columns:
                df["flood_risk_index"] = df[alt]
                break

    needed = {"region_id", "flood_risk_index"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"flood_risk.parquet missing columns: {missing}. Found: {list(df.columns)}")

    return df


def load_hazard_series(region_id: str, scenario_id: str) -> pd.DataFrame:
    rid = canonical_region_id(region_id)
    df = _hazard_table()
    out = df[(df["region_id"] == rid) & (df["scenario_id"] == scenario_id)].copy()
    out = out.sort_values("year")
    return out[["year", "heat_index", "drought_index"]]


def load_socio_series(region_id: str) -> pd.DataFrame:
    rid = canonical_region_id(region_id)
    df = _socio_table()
    out = df[df["region_id"] == rid].copy()

    # If year exists, keep it; otherwise treat as a single baseline row.
    cols = ["region_id"]
    if "year" in out.columns:
        cols.append("year")
    cols += ["tourism_share", "arrivals_index", "gdp_index", "education_index", "health_index", "governance_index"]
    out = out[cols]

    if "year" in out.columns:
        out = out.sort_values("year")

    return out


def load_flood_risk(region_id: str) -> float:
    """Return a scalar 0..1 flood risk index for the region."""
    rid = canonical_region_id(region_id)
    df = _flood_table()
    row = df[df["region_id"] == rid]
    if row.empty:
        # conservative default if missing
        return 0.10
    return float(row.iloc[0]["flood_risk_index"])
