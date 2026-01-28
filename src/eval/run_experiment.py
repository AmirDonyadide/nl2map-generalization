# src/eval/run_experiment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence

from src.eval.labels import remap_to_bundle_order

import numpy as np
import pandas as pd
import joblib

from src.eval.data import load_pairs_and_features, select_features
from src.eval.metrics import classification_metrics, regression_metrics


from src.eval.splits import make_group_splits
from src.eval.routing import route_and_predict_param_value


FeatureMode = Literal["prompt_only", "prompt_plus_map"]


@dataclass(frozen=True)
class ExperimentPaths:
    pairs_path: Path
    X_path: Path
    bundle_path: Path

def evaluate_experiment(
    *,
    name: str,
    paths: ExperimentPaths,
    feature_mode: FeatureMode,
    map_dim: int,
    prompt_dim: int,
    group_col: str = "map_id",
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    operator_col: str = "operator",
    param_value_col: str = "param_value",
    diag_col: str = "extent_diag_m",
    area_col: str = "extent_area_m2",
    distance_ops: Sequence[str] = ("aggregate", "displace", "simplify"),
    area_ops: Sequence[str] = ("select",),
    include_classification_report: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained (classifier + per-operator regressors) bundle on a dataset.

    Produces three evaluation views:
      (1) classifier-only metrics
      (2) regressors-only (oracle routing using TRUE operator labels)
      (3) end-to-end pipeline (routing using PREDICTED operator labels)

    Parameters
    ----------
    name : str
        Experiment key/name (stored in returned dict).
    paths : ExperimentPaths
        paths.pairs_path: train_pairs.parquet
        paths.X_path:     feature matrix (.npy) with layout [map | prompt]
        paths.bundle_path: cls_plus_regressors.joblib containing classifier+regressors
    feature_mode : "prompt_only" | "prompt_plus_map"
        Which features to feed into classifier/regressors for THIS evaluation.
    map_dim / prompt_dim : int
        Dimensions used to slice X into blocks.
    group_col : str
        Grouping column used to split without leakage (default: map_id).
    operator_col / param_value_col : str
        Column names in pairs parquet.
    diag_col / area_col : str
        Extent reference columns in pairs parquet.

    Returns
    -------
    Dict[str, Any]
        Results dict ready for aggregation in notebook tables.
    """
    df, X_full = load_pairs_and_features(pairs_path=paths.pairs_path, X_path=paths.X_path)
    X = select_features(X_full, feature_mode=feature_mode, map_dim=map_dim, prompt_dim=prompt_dim)

    # Split indices
    splits = make_group_splits(
        df,
        group_col=group_col,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    df_val = df.iloc[splits.val_idx].copy()
    df_test = df.iloc[splits.test_idx].copy()

    # Targets in real units (meters or m² depending on operator)
    if param_value_col not in df.columns:
        raise KeyError(f"param_value_col '{param_value_col}' not found in pairs parquet.")

    y_val_raw  = pd.to_numeric(df_val[param_value_col], errors="coerce").to_numpy(dtype=float)
    y_test_raw = pd.to_numeric(df_test[param_value_col], errors="coerce").to_numpy(dtype=float)

    X_val = X[splits.val_idx]
    X_test = X[splits.test_idx]

    # Dataset class order (make it a list[str], not np.ndarray)
    if operator_col not in df.columns:
        raise KeyError(f"operator_col '{operator_col}' not found in pairs parquet.")
    ds_class_names: list[str] = sorted(df[operator_col].dropna().astype(str).unique().tolist())

    y_val   = pd.Categorical(df_val[operator_col],   categories=ds_class_names).codes
    y_test  = pd.Categorical(df_test[operator_col],  categories=ds_class_names).codes

    # Load bundle
    if not paths.bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle: {paths.bundle_path}")
    bundle = joblib.load(paths.bundle_path)

    clf = bundle["classifier"]
    regressors_by_class = bundle["regressors_by_class"]

    # Bundle class names as list[str]
    bundle_class_names: list[str] = [str(x) for x in bundle["class_names"]]

    # Remap dataset labels -> bundle label space
    y_val_b   = remap_to_bundle_order(y_val,   src_names=ds_class_names, dst_names=bundle_class_names)
    y_test_b  = remap_to_bundle_order(y_test,  src_names=ds_class_names, dst_names=bundle_class_names)

    # inverse transform if log1p was used on param_norm during training
    use_log1p = bool(bundle.get("use_log1p", False))
    inv_t: Callable[[float], float] = (lambda x: float(np.expm1(x))) if use_log1p else (lambda x: float(x))

    # (1) classifier-only
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    clf_metrics = {
        "VAL": classification_metrics(y_val_b, val_pred, class_names=bundle_class_names, include_report=False),
        "TEST": classification_metrics(y_test_b, test_pred, class_names=bundle_class_names, include_report=include_classification_report),
    }

    # (2) regressors-only (oracle routing)
    yhat_val_or = route_and_predict_param_value(
        X=X_val,
        df=df_val,
        op_idx=y_val_b,
        class_names=bundle_class_names,
        regressors_by_class=regressors_by_class,
        inv_t=inv_t,
        distance_ops=distance_ops,
        area_ops=area_ops,
        diag_col=diag_col,
        area_col=area_col,
        clamp_nonneg=True,
    )
    yhat_test_or = route_and_predict_param_value(
        X=X_test,
        df=df_test,
        op_idx=y_test_b,
        class_names=bundle_class_names,
        regressors_by_class=regressors_by_class,
        inv_t=inv_t,
        distance_ops=distance_ops,
        area_ops=area_ops,
        diag_col=diag_col,
        area_col=area_col,
        clamp_nonneg=True,
    )

    reg_oracle = {
        "VAL_oracle": regression_metrics(y_val_raw, yhat_val_or),
        "TEST_oracle": regression_metrics(y_test_raw, yhat_test_or),
    }

    # (3) end-to-end pipeline (pred routing)
    yhat_val_pipe = route_and_predict_param_value(
        X=X_val,
        df=df_val,
        op_idx=val_pred,
        class_names=bundle_class_names,
        regressors_by_class=regressors_by_class,
        inv_t=inv_t,
        distance_ops=distance_ops,
        area_ops=area_ops,
        diag_col=diag_col,
        area_col=area_col,
        clamp_nonneg=True,
    )
    yhat_test_pipe = route_and_predict_param_value(
        X=X_test,
        df=df_test,
        op_idx=test_pred,
        class_names=bundle_class_names,
        regressors_by_class=regressors_by_class,
        inv_t=inv_t,
        distance_ops=distance_ops,
        area_ops=area_ops,
        diag_col=diag_col,
        area_col=area_col,
        clamp_nonneg=True,
    )

    reg_pipeline = {
        "VAL_pipeline": regression_metrics(y_val_raw, yhat_val_pipe),
        "TEST_pipeline": regression_metrics(y_test_raw, yhat_test_pipe),
    }

    return {
        "name": name,
        "paths": {
            "pairs_path": str(paths.pairs_path),
            "X_path": str(paths.X_path),
            "bundle_path": str(paths.bundle_path),
        },
        "notes": {
            "feature_mode": feature_mode,
            "split": {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio, "group_col": group_col},
            "dims": {"map_dim": map_dim, "prompt_dim": prompt_dim, "X_dim": int(X.shape[1])},
            "bundle": {"use_log1p": use_log1p},
        },
        "class_names": [str(x) for x in bundle_class_names],
        "classifier": clf_metrics,
        "regressor_oracle": reg_oracle,
        "regressor_pipeline": reg_pipeline,
    }
