from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import shap


@dataclass
class SurrogateBundle:
    preprocessor: Any
    model: Any
    feature_names: list[str]
    background: np.ndarray


def load_surrogate_bundle(artifact_dir: Path) -> SurrogateBundle:
    pre = joblib.load(artifact_dir / "preprocessor.joblib")
    model = joblib.load(artifact_dir / "model.joblib")
    feature_names = json.loads((artifact_dir / "feature_names.json").read_text(encoding="utf-8"))
    background = np.load(artifact_dir / "background.npy", allow_pickle=False)
    return SurrogateBundle(preprocessor=pre, model=model, feature_names=feature_names, background=background)


def predict_and_shap(bundle: SurrogateBundle, raw_row: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    raw_row: 1-row dataframe with the same raw columns used in training (categoricals + numerics).
    Returns:
      - prediction
      - dataframe with feature_value and shap_value (top absolute contributions first)
    """
    X = bundle.preprocessor.transform(raw_row)
    pred = float(bundle.model.predict(X)[0])

    explainer = shap.TreeExplainer(bundle.model, data=bundle.background, feature_names=bundle.feature_names)
    shap_vals = explainer.shap_values(X)

    # shap may return (n, m) or list; standardize
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals).reshape(-1)

    # feature values in encoded space
    x_vals = np.asarray(X.todense() if hasattr(X, "todense") else X).reshape(-1)

    df = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "value": x_vals,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
        }
    ).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    return pred, df
