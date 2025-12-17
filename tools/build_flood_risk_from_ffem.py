"""
Build flood_risk.parquet for CRISI v2 from FFEM flood fatalities database.

Usage (from repo root, e.g. C:\\Users\\USER\\OneDrive\\Desktop\\crisi_v2):

    .\\.venv\\Scripts\\python.exe tools\\build_flood_risk_from_ffem.py

The script expects the FFEM DB either as:

    1) A ZIP file:
        data/FFEM_DB.zip
       containing at least:
        - 'NUTS 3.csv'
        - 'Fatalities.csv'

    OR

    2) An extracted folder:
        data/ffem_db/
       containing the same CSVs.

It creates:

    data/flood_risk.parquet
    data/flood_risk_debug.csv

Columns in flood_risk_debug.csv:

    region_id, fatalities, population, fatalities_per_100k, flood_risk_index
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]          # .../crisi_v2
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

ZIP_CANDIDATES = [
    DATA_DIR / "FFEM_DB.zip",
    DATA_DIR / "ffem_db.zip",
]

DIR_CANDIDATES = [
    DATA_DIR / "ffem_db",
    DATA_DIR / "FFEM_DB",
]


def _load_from_zip(zip_path: Path):
    print(f"Loading FFEM DB from ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("NUTS 3.csv") as f:
            nuts3 = pd.read_csv(f, sep=";", encoding="utf-8-sig")
        with z.open("Fatalities.csv") as f:
            fat = pd.read_csv(f, sep=";", encoding="utf-8-sig")
    return nuts3, fat


def _load_from_dir(dir_path: Path):
    print(f"Loading FFEM DB from folder: {dir_path}")
    nuts3_path = dir_path / "NUTS 3.csv"
    fat_path = dir_path / "Fatalities.csv"
    if not nuts3_path.exists() or not fat_path.exists():
        raise FileNotFoundError(
            f"Expected 'NUTS 3.csv' and 'Fatalities.csv' in {dir_path}, "
            f"found: {[p.name for p in dir_path.glob('*.csv')]}"
        )
    nuts3 = pd.read_csv(nuts3_path, sep=";", encoding="utf-8-sig")
    fat = pd.read_csv(fat_path, sep=";", encoding="utf-8-sig")
    return nuts3, fat


def load_ffem():
    # Try ZIPs
    for zpath in ZIP_CANDIDATES:
        if zpath.exists():
            return _load_from_zip(zpath)

    # Try folders
    for dpath in DIR_CANDIDATES:
        if dpath.exists() and dpath.is_dir():
            return _load_from_dir(dpath)

    raise FileNotFoundError(
        "Could not find FFEM DB. Place it as either:\n"
        "  - data/FFEM_DB.zip  (ZIP with 'NUTS 3.csv' and 'Fatalities.csv'), or\n"
        "  - data/ffem_db/     (folder with these CSVs)."
    )


def build_flood_risk():
    nuts3, fat = load_ffem()

    required_nuts3_cols = {"NUTS_3_ID", "NUTS_2_ID", "NUTS_0_ID", "NUTS_3_POPULATION"}
    if not required_nuts3_cols.issubset(nuts3.columns):
        missing = required_nuts3_cols - set(nuts3.columns)
        raise ValueError(f"NUTS 3.csv missing columns: {missing}")

    required_fat_cols = {"FATALITY_ID", "NUTS_3_ID", "DATE"}
    if not required_fat_cols.issubset(fat.columns):
        missing = required_fat_cols - set(fat.columns)
        raise ValueError(f"Fatalities.csv missing columns: {missing}")

    # Filter Greece (NUTS_0_ID == 'EL')
    nuts3_el = nuts3[nuts3["NUTS_0_ID"] == "EL"].copy()

    fat_el = fat.merge(
        nuts3[
            [
                "NUTS_3_ID",
                "NUTS_2_ID",
                "NUTS_2_NAME",
                "NUTS_0_ID",
                "NUTS_0_NAME",
                "NUTS_3_POPULATION",
            ]
        ],
        on="NUTS_3_ID",
        how="left",
    )
    fat_el = fat_el[fat_el["NUTS_0_ID"] == "EL"].copy()

    if fat_el.empty:
        raise ValueError("No Greek fatalities (NUTS_0_ID == 'EL') found in FFEM DB.")

    # Aggregate fatalities by NUTS2
    agg = (
        fat_el.groupby("NUTS_2_ID")
        .agg(fatalities=("FATALITY_ID", "count"))
        .reset_index()
    )

    # Aggregate population by NUTS2 (sum of NUTS3 populations)
    pop_agg = (
        nuts3_el.groupby("NUTS_2_ID")
        .agg(population=("NUTS_3_POPULATION", "sum"))
        .reset_index()
    )

    agg = agg.merge(pop_agg, on="NUTS_2_ID", how="left")

    # Compute fatalities per 100k inhabitants
    agg["fatalities_per_100k"] = (
        agg["fatalities"] / agg["population"] * 100_000.0
    )

    # Map 2013+ NUTS2 codes to CRISI NUTS2 region_ids
    CODE_MAP_2013_TO_CRISI = {
        "EL51": "EL11",
        "EL52": "EL12",
        "EL53": "EL13",
        "EL54": "EL21",
        "EL61": "EL14",
        "EL62": "EL22",
        "EL63": "EL23",
        "EL64": "EL24",
        "EL65": "EL25",
        "EL30": "EL30",
        "EL41": "EL41",
        "EL42": "EL42",
        "EL43": "EL43",
    }

    agg["region_id"] = agg["NUTS_2_ID"].map(CODE_MAP_2013_TO_CRISI)
    unknown = agg[agg["region_id"].isna()]
    if not unknown.empty:
        raise ValueError(
            "Some NUTS2 codes could not be mapped to CRISI region_id:\n"
            + unknown.to_string(index=False)
        )

    # Normalise fatalities_per_100k to 0..1 flood_risk_index
    fmin = float(agg["fatalities_per_100k"].min())
    fmax = float(agg["fatalities_per_100k"].max())
    span = fmax - fmin

    if span <= 1e-9:
        agg["flood_risk_index"] = 0.5
    else:
        agg["flood_risk_index"] = (agg["fatalities_per_100k"] - fmin) / span

    # Build final table
    flood = agg[
        [
            "region_id",
            "fatalities",
            "population",
            "fatalities_per_100k",
            "flood_risk_index",
        ]
    ].copy()

    flood = flood.sort_values("region_id").reset_index(drop=True)

    out_parquet = DATA_DIR / "flood_risk.parquet"
    out_csv = DATA_DIR / "flood_risk_debug.csv"

    flood.to_parquet(out_parquet, index=False)
    flood.to_csv(out_csv, index=False)

    print(f"Wrote flood risk table to: {out_parquet}")
    print(f"Wrote debug CSV to:       {out_csv}")
    print("Summary (fatalities_per_100k):")
    print(flood[["region_id", "fatalities_per_100k", "flood_risk_index"]])


if __name__ == "__main__":
    build_flood_risk()
