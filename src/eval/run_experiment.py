# src/eval/run_experiment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from src.eval.data import load_pairs_and_features, select_features
from src.eval.labels import remap_to_bundle_order
from src.eval.metrics import classification_metrics, regression_metrics
from src.eval.routing import route_and_predict_param_value
from src.eval.splits import make_group_splits


FeatureMode = Literal["prompt_only", "map_only", "prompt_plus_map"]


@dataclass(frozen=True)
class ExperimentPaths:
    pairs_path: Path
    X_path: Path
    bundle_path: Path
    preproc_path: Optional[Path] = None  # optional for backward compatibility


def _unwrap_preproc(preproc_bundle: Any) -> Any:
    """
    Return the actual transformer-like object that has .transform(X),
    from either a direct object or a dict bundle.
    """
    if preproc_bundle is None:
        return None

    if hasattr(preproc_bundle, "transform"):
        return preproc_bundle

    if isinstance(preproc_bundle, dict):
        for key in ("pipeline", "preproc", "preprocessor", "transformer", "model"):
            obj = preproc_bundle.get(key)
            if obj is not None and hasattr(obj, "transform"):
                return obj

    raise TypeError(
        "Unsupported preproc bundle format. "
        "Expected an object with .transform(X) or a dict containing such an object."
    )


def _apply_preproc(X: np.ndarray, preproc_bundle: Any) -> np.ndarray:
    """
    Apply the preprocessing bundle saved during training.
    Returns transformed X as float.
    """
    if preproc_bundle is None:
        return np.asarray(X, dtype=float)

    preproc = _unwrap_preproc(preproc_bundle)
    return np.asarray(preproc.transform(X), dtype=float)


def _check_preproc_dim(X: np.ndarray, preproc_bundle: Any, *, feature_mode: FeatureMode) -> None:
    """
    Fail fast if the preprocessing object was trained for a different feature dimension.
    """
    if preproc_bundle is None:
        return
    preproc = _unwrap_preproc(preproc_bundle)

    n_in = getattr(preproc, "n_features_in_", None)
    if n_in is not None and int(n_in) != int(X.shape[1]):
        raise ValueError(
            f"Preproc expects {int(n_in)} features, but X has {int(X.shape[1])}. "
            f"Did you pass the correct preproc for feature_mode='{feature_mode}'?"
        )


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
    """
    # ---------- Load raw df + raw X ----------
    df, X_full = load_pairs_and_features(pairs_path=paths.pairs_path, X_path=paths.X_path)

    # ---------- Basic schema checks ----------
    for col in (group_col, operator_col, param_value_col, diag_col, area_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from pairs parquet: {paths.pairs_path}")

    # ---------- Feature-mode safety checks ----------
    map_dim = int(map_dim)
    prompt_dim = int(prompt_dim)

    if feature_mode == "map_only" and prompt_dim != 0:
        raise ValueError("For feature_mode='map_only', set prompt_dim=0.")
    if feature_mode == "prompt_only" and map_dim != 0:
        raise ValueError("For feature_mode='prompt_only', set map_dim=0.")

    # ---------- Select raw feature block(s) ----------
    X_sel = select_features(
        X_full,
        feature_mode=feature_mode,
        map_dim=map_dim,
        prompt_dim=prompt_dim,
    )

    # ---------- Load model bundle (needed for class ordering + log setting) ----------
    if not paths.bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle: {paths.bundle_path}")
    bundle = joblib.load(paths.bundle_path)

    clf = bundle["classifier"]
    regressors_by_class = bundle["regressors_by_class"]
    bundle_class_names: list[str] = [str(x).strip().lower() for x in bundle["class_names"]]

    # ---------- Optional preprocessing ----------
    preproc_bundle = None
    if paths.preproc_path is not None:
        pp = Path(paths.preproc_path)
        if not pp.exists():
            raise FileNotFoundError(f"Missing preproc: {pp}")
        preproc_bundle = joblib.load(pp)

        _check_preproc_dim(X_sel, preproc_bundle, feature_mode=feature_mode)
        X_sel = _apply_preproc(X_sel, preproc_bundle)

    # ---------- Split indices ----------
    splits = make_group_splits(
        df,
        group_col=group_col,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    df_val = df.iloc[splits.val_idx].copy()
    df_test = df.iloc[splits.test_idx].copy()

    X_val = X_sel[splits.val_idx]
    X_test = X_sel[splits.test_idx]

    # ---------- Targets ----------
    # Regression target in real units (meters or m² depending on operator)
    y_val_raw = pd.to_numeric(df_val[param_value_col], errors="coerce").to_numpy(dtype=float)
    y_test_raw = pd.to_numeric(df_test[param_value_col], errors="coerce").to_numpy(dtype=float)

    # Classification target: normalize operator strings
    op_all = df[operator_col].astype(str).str.strip().str.lower()
    op_val = df_val[operator_col].astype(str).str.strip().str.lower()
    op_test = df_test[operator_col].astype(str).str.strip().str.lower()

    # Dataset class order for coding: use dataset unique list (normalized)
    ds_class_names: list[str] = sorted(op_all.dropna().unique().tolist())

    y_val = pd.Categorical(op_val, categories=ds_class_names).codes
    y_test = pd.Categorical(op_test, categories=ds_class_names).codes

    # Remap dataset labels -> bundle label space
    y_val_b = remap_to_bundle_order(
        y_val, src_names=ds_class_names, dst_names=bundle_class_names, strict=False
    )
    y_test_b = remap_to_bundle_order(
        y_test, src_names=ds_class_names, dst_names=bundle_class_names, strict=False
    )

    # inverse transform if log1p was used on param_norm during training
    use_log1p = bool(bundle.get("use_log1p", False))
    inv_t: Callable[[float], float] = (lambda x: float(np.expm1(x))) if use_log1p else (lambda x: float(x))

    # ---------- (1) classifier-only ----------
    val_pred = np.asarray(clf.predict(X_val), dtype=int).reshape(-1)
    test_pred = np.asarray(clf.predict(X_test), dtype=int).reshape(-1)

    clf_metrics = {
        "VAL": classification_metrics(
            y_val_b, val_pred, class_names=bundle_class_names, include_report=False
        ),
        "TEST": classification_metrics(
            y_test_b, test_pred, class_names=bundle_class_names, include_report=include_classification_report
        ),
    }

    # ---------- (2) regressors-only (oracle routing using TRUE operator labels) ----------
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

    # ---------- (3) end-to-end pipeline (routing using PREDICTED operator labels) ----------
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
            "preproc_path": str(paths.preproc_path) if paths.preproc_path is not None else None,
        },
        "notes": {
            "feature_mode": feature_mode,
            "split": {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio, "group_col": group_col},
            "dims": {"map_dim": map_dim, "prompt_dim": prompt_dim, "X_dim": int(X_sel.shape[1])},
            "bundle": {"use_log1p": use_log1p},
        },
        "class_names": [str(x) for x in bundle_class_names],
        "classifier": clf_metrics,
        "regressor_oracle": reg_oracle,
        "regressor_pipeline": reg_pipeline,
    }
