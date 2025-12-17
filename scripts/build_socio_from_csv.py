from pathlib import Path
import pandas as pd

# ----- CONFIG -----
INPUT_CSV = Path("data/socio_raw/EL30_socio.csv")
OUTPUT_PARQUET = Path("data/socioeconomic.parquet")
# -------------------


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {
        "region_id",
        "year",
        "tourism_share",
        "arrivals_index",
        "gdp_index",
        "education_index",
        "health_index",
        "governance_index",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}, found {df.columns.tolist()}")

    df["year"] = df["year"].astype(int)

    if OUTPUT_PARQUET.exists():
        existing = pd.read_parquet(OUTPUT_PARQUET)
        df = pd.concat([existing, df], ignore_index=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Written {len(df)} rows to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
