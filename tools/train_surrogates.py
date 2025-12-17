from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor

import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "surrogate"
ART_DIR = PROJECT_ROOT / "artifacts" / "surrogates"


def _split_groups(df: pd.DataFrame, group_cols: list[str], test_size=0.2, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    groups = df[group_cols].astype(str).agg("|".join, axis=1).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx


def _train_tree_regressor(X_train, y_train) -> ExtraTreesRegressor:
    # Conservative defaults; tune later.
    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    return model


def train_resilience():
    df = pd.read_parquet(DATA_DIR / "resilience_samples.parquet").dropna(subset=["resilience_100"]).copy()

    target = "resilience_100"
    cat_cols = ["region_id", "scenario_id"]
    num_cols = [c for c in df.columns if c not in cat_cols + [target, "risk_100", "exposure", "sensitivity", "adaptive"]]

    train_idx, test_idx = _split_groups(df, ["region_id", "scenario_id"], test_size=0.2, seed=42)

    X = df[cat_cols + num_cols]
    y = df[target].values

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    X_train = pre.fit_transform(X.iloc[train_idx])
    X_test = pre.transform(X.iloc[test_idx])

    model = _train_tree_regressor(X_train, y[train_idx])

    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    metrics = {
        "target": target,
        "rows": int(len(df)),
        "r2_train": float(r2_score(y[train_idx], pred_tr)),
        "r2_test": float(r2_score(y[test_idx], pred_te)),
        "mae_train": float(mean_absolute_error(y[train_idx], pred_tr)),
        "mae_test": float(mean_absolute_error(y[test_idx], pred_te)),
    }

    out_dir = ART_DIR / "resilience"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(model, out_dir / "model.joblib")

    feat_names = pre.get_feature_names_out()
    (out_dir / "feature_names.json").write_text(json.dumps(list(map(str, feat_names)), indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # SHAP background (sample)
    bg = X_train[np.random.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)]
    np.save(out_dir / "background.npy", bg)

    print("Resilience surrogate metrics:", metrics)


def train_cba():
    df = pd.read_parquet(DATA_DIR / "cba_samples.parquet").dropna(subset=["npv"]).copy()

    target = "npv"
    cat_cols = ["region_id", "scenario_id"]
    num_cols = [c for c in df.columns if c not in cat_cols + [target, "irr"]]

    train_idx, test_idx = _split_groups(df, ["region_id", "scenario_id"], test_size=0.2, seed=42)

    X = df[cat_cols + num_cols]
    y = df[target].values

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    X_train = pre.fit_transform(X.iloc[train_idx])
    X_test = pre.transform(X.iloc[test_idx])

    model = _train_tree_regressor(X_train, y[train_idx])

    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    metrics = {
        "target": target,
        "rows": int(len(df)),
        "r2_train": float(r2_score(y[train_idx], pred_tr)),
        "r2_test": float(r2_score(y[test_idx], pred_te)),
        "mae_train": float(mean_absolute_error(y[train_idx], pred_tr)),
        "mae_test": float(mean_absolute_error(y[test_idx], pred_te)),
    }

    out_dir = ART_DIR / "cba"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(model, out_dir / "model.joblib")

    feat_names = pre.get_feature_names_out()
    (out_dir / "feature_names.json").write_text(json.dumps(list(map(str, feat_names)), indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bg = X_train[np.random.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)]
    np.save(out_dir / "background.npy", bg)

    print("CBA surrogate metrics:", metrics)


def main():
    (ART_DIR).mkdir(parents=True, exist_ok=True)
    train_resilience()
    train_cba()
    print("Done training surrogates.")


if __name__ == "__main__":
    main()
