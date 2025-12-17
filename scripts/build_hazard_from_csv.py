from pathlib import Path
import pandas as pd

# ----- CONFIG -----
INPUT_CSV = Path("data/raw/region_EL30_green.csv")
OUTPUT_PARQUET = Path("data/hazard.parquet")
REGION_ID = "EL30"
SCENARIO_ID = "green"
# -------------------


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"year", "heat", "drought"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}, found {df.columns.tolist()}")

    df_out = pd.DataFrame(
        {
            "region_id": REGION_ID,
            "scenario_id": SCENARIO_ID,
            "year": df["year"].astype(int),
            "heat_index": df["heat"].astype(float),
            "drought_index": df["drought"].astype(float),
        }
    )

    if OUTPUT_PARQUET.exists():
        existing = pd.read_parquet(OUTPUT_PARQUET)
        df_out = pd.concat([existing, df_out], ignore_index=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Written {len(df_out)} rows to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
