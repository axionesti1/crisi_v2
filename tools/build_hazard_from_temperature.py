"""
Build hazard.parquet for CRISI v2 from IPCC-based temperature dataset.

Usage (from repo root, e.g. C:\\Users\\USER\\OneDrive\\Desktop\\crisi_v2):

    .\\.venv\\Scripts\\python.exe tools\\build_hazard_from_temperature.py

This script expects the temperature CSV to be in the `data` folder with one of
these names:

    - temperature_dataset.csv
    - temperature_dataset csv.csv

It creates:

    data/hazard.parquet

with columns:
    region_id, scenario_id, year, heat_index, drought_index
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # .../crisi_v2
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CANDIDATE_FILES = [
    DATA_DIR / "temperature_dataset.csv",
    DATA_DIR / "temperature_dataset csv.csv",
]

TEMP_CSV = None
for p in CANDIDATE_FILES:
    if p.exists():
        TEMP_CSV = p
        break

if TEMP_CSV is None:
    raise FileNotFoundError(
        f"Could not find temperature CSV. Expected one of: "
        + ", ".join(str(p) for p in CANDIDATE_FILES)
    )

print(f"Using temperature dataset: {TEMP_CSV}")

# -------------------------------------------------------------------
# Load and clean
# -------------------------------------------------------------------
df = pd.read_csv(TEMP_CSV)

required_cols = {"Scenario", "NUTS 2 Area Name", "Date", "Tmax"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Temperature dataset is missing required columns: {missing}")

# Strip whitespace on names
df["NUTS 2 Area Name"] = df["NUTS 2 Area Name"].astype(str).str.strip()

# Map NUTS2 names -> CRISI region codes
NAME_TO_CODE: Dict[str, str] = {
    "Anatoliki Makedonia, Thraki": "EL11",
    "Kentriki Makedonia": "EL12",
    "Dytiki Makedonia": "EL13",
    "Thessalia": "EL14",
    "Ipeiros": "EL21",
    "Ionia Nisia": "EL22",
    "Dytiki Elláda": "EL23",
    "Dytiki Elláda ": "EL23",  # defensive
    "Sterea Elláda": "EL24",
    "Sterea Elláda ": "EL24",
    "Peloponnisos": "EL25",
    "Attiki": "EL30",
    "Voreio Aigaio": "EL41",
    "Notio Aigaio": "EL42",
    "Kriti": "EL43",
}

df["region_id"] = df["NUTS 2 Area Name"].map(NAME_TO_CODE)

unknown = df.loc[df["region_id"].isna(), "NUTS 2 Area Name"].unique()
if len(unknown) > 0:
    raise ValueError(
        "Found NUTS2 area names not in NAME_TO_CODE mapping: "
        + ", ".join(map(str, unknown))
    )

# Parse date -> year, month
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Keep only needed columns
df = df[["Scenario", "region_id", "year", "month", "Tmax"]]

# -------------------------------------------------------------------
# Map climate scenarios (SSPs) -> CRISI scenarios
# -------------------------------------------------------------------
# CRISI scenario IDs (from crisi_core.scenarios.DEFAULT_SCENARIOS):
#   - green_global_resilience
#   - business_as_usual
#   - divided_disparity
#   - techno_optimism
#   - regional_fortress
SSP_TO_CRISI: Dict[str, List[str]] = {
    "ssp126": ["green_global_resilience"],
    "ssp245": ["business_as_usual"],
    "ssp370": ["divided_disparity", "regional_fortress"],
    "ssp585": ["techno_optimism"],
}

unknown_ssp = set(df["Scenario"].unique()) - set(SSP_TO_CRISI.keys())
if unknown_ssp:
    raise ValueError(f"Temperature dataset has SSPs with no mapping: {unknown_ssp}")

records = []
for ssp, group in df.groupby("Scenario"):
    crisi_ids = SSP_TO_CRISI.get(ssp, [])
    for crisi_id in crisi_ids:
        g2 = group.copy()
        g2["scenario_id"] = crisi_id
        records.append(g2)

df_expanded = pd.concat(records, ignore_index=True)

# -------------------------------------------------------------------
# Restrict years to CRISI planning horizon, e.g. 2025–2055
# -------------------------------------------------------------------
MIN_YEAR = 2025
MAX_YEAR = 2055

df_expanded = df_expanded[(df_expanded["year"] >= MIN_YEAR) & (df_expanded["year"] <= MAX_YEAR)]

if df_expanded.empty:
    raise ValueError("No rows left after filtering to years between 2025 and 2055.")

# -------------------------------------------------------------------
# Compute annual summer heat_index (JJA mean Tmax)
# -------------------------------------------------------------------
summer = df_expanded[df_expanded["month"].isin([6, 7, 8])]
if summer.empty:
    raise ValueError("No summer months (Jun–Aug) in the filtered dataset.")

heat_agg = (
    summer
    .groupby(["region_id", "scenario_id", "year"], as_index=False)["Tmax"]
    .mean()
    .rename(columns={"Tmax": "heat_raw"})
)

hmin = float(heat_agg["heat_raw"].min())
hmax = float(heat_agg["heat_raw"].max())
span = hmax - hmin
if span <= 1e-9:
    # Degenerate (all same temperature) – unlikely
    heat_agg["heat_index"] = 0.5
else:
    heat_agg["heat_index"] = (heat_agg["heat_raw"] - hmin) / span

# -------------------------------------------------------------------
# Placeholder drought_index
# -------------------------------------------------------------------
# For now, we use a conservative proxy correlated with heat_index.
# You can later replace this with a real drought index (SPEI, SPI, etc.)
heat_agg["drought_index"] = (heat_agg["heat_index"] * 0.8).clip(0.0, 1.0)

# Keep only required columns
hazard = heat_agg[["region_id", "scenario_id", "year", "heat_index", "drought_index"]].copy()

# Sort for readability
hazard = hazard.sort_values(["region_id", "scenario_id", "year"]).reset_index(drop=True)

# -------------------------------------------------------------------
# Write parquet (and optional CSV debug)
# -------------------------------------------------------------------
HAZARD_PARQUET = DATA_DIR / "hazard.parquet"
hazard.to_parquet(HAZARD_PARQUET, index=False)
print(f"Wrote hazard table to: {HAZARD_PARQUET}")

# Optional debug CSV for inspection
HAZARD_CSV = DATA_DIR / "hazard_debug.csv"
hazard.to_csv(HAZARD_CSV, index=False)
print(f"Wrote debug CSV to: {HAZARD_CSV}")

# Basic sanity prints
print("Summary:")
print(hazard.groupby("scenario_id")["year"].agg(["min", "max"]).reset_index())
print("Regions:", sorted(hazard["region_id"].unique()))
