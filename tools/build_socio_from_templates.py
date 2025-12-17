from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

"""
Builds data/socio.parquet from a simple regional template.

- Input:  data/socio_template.csv
- Output: data/socio.parquet

Schema of socio.parquet:

    region_id          (str)
    year               (int)
    tourism_share      (float, 0-1)
    arrivals_index     (float, 0-1)
    gdp_index          (float, 0-1)
    education_index    (float, 0-1)
    health_index       (float, 0-1)
    governance_index   (float, 0-1)

Logic:
- socio_template.csv provides a base value and a trend (delta over 2025-2055)
  for each index and each region.
- For a given year y, we compute:
      t = (y - 2025) / (2055 - 2025)  in [0,1]
      index_y = base + trend * t
- We then clip all indices to [0,1]. No extra cross-region normalization
  is applied (so your chosen template levels are preserved).
"""

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TEMPLATE_PATH = DATA_DIR / "socio_template.csv"
OUTPUT_PATH = DATA_DIR / "socio.parquet"


def build_socio_from_template():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template not found at {TEMPLATE_PATH}. "
            "Create data/socio_template.csv first."
        )

    tpl = pd.read_csv(TEMPLATE_PATH)

    required_cols = {
        "region_id",
        "tourism_base",
        "tourism_trend",
        "arrivals_base",
        "arrivals_trend",
        "gdp_base",
        "gdp_trend",
        "education_base",
        "education_trend",
        "health_base",
        "health_trend",
        "governance_base",
        "governance_trend",
    }
    missing = required_cols - set(tpl.columns)
    if missing:
        raise ValueError(
            f"socio_template.csv is missing required columns: {missing}. "
            f"Columns found: {tpl.columns.tolist()}"
        )

    years = np.arange(2025, 2056)

    rows = []
    for _, row in tpl.iterrows():
        region_id = str(row["region_id"]).strip()

        tourism_base = float(row["tourism_base"])
        tourism_trend = float(row["tourism_trend"])
        arrivals_base = float(row["arrivals_base"])
        arrivals_trend = float(row["arrivals_trend"])
        gdp_base = float(row["gdp_base"])
        gdp_trend = float(row["gdp_trend"])
        education_base = float(row["education_base"])
        education_trend = float(row["education_trend"])
        health_base = float(row["health_base"])
        health_trend = float(row["health_trend"])
        governance_base = float(row["governance_base"])
        governance_trend = float(row["governance_trend"])

        for y in years:
            # 0..1 over the 30-year horizon
            t = (y - 2025) / (2055 - 2025)

            tourism_share = tourism_base + tourism_trend * t
            arrivals_index = arrivals_base + arrivals_trend * t
            gdp_index = gdp_base + gdp_trend * t
            education_index = education_base + education_trend * t
            health_index = health_base + health_trend * t
            governance_index = governance_base + governance_trend * t

            rows.append(
                {
                    "region_id": region_id,
                    "year": int(y),
                    "tourism_share": tourism_share,
                    "arrivals_index": arrivals_index,
                    "gdp_index": gdp_index,
                    "education_index": education_index,
                    "health_index": health_index,
                    "governance_index": governance_index,
                }
            )

    df = pd.DataFrame(rows)

    # Clip all indices to [0,1] to respect CRISI scale
    for col in [
        "tourism_share",
        "arrivals_index",
        "gdp_index",
        "education_index",
        "health_index",
        "governance_index",
    ]:
        df[col] = df[col].clip(0.0, 1.0)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote socio data to {OUTPUT_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    build_socio_from_template()
