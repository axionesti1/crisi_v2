from __future__ import annotations
import pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2021/4326/20M/nutsrg_2.geojson"
OUT_PATH = DATA_DIR / "nuts2_2021_20M_4326.geojson"

if __name__ == "__main__":
    print(f"Downloading NUTS-2 GeoJSON from {URL}")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    OUT_PATH.write_bytes(r.content)
    print(f"Saved to {OUT_PATH}")
