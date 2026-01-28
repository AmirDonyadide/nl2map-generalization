# src/eval/routing.py
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def route_and_predict_param_value(
    *,
    X: np.ndarray,
    df: pd.DataFrame,
    op_idx: np.ndarray,
    class_names: Sequence[str],
    regressors_by_class: Mapping[str, Tuple[Any, Any]],
    inv_t: Callable[[float], float] = lambda x: x,
    distance_ops: Sequence[str] = ("aggregate", "displace", "simplify"),
    area_ops: Sequence[str] = ("select",),
    diag_col: str = "extent_diag_m",
    area_col: str = "extent_area_m2",
    clamp_nonneg: bool = True,
) -> np.ndarray:
    """
    Route each sample to the operator-specific regressor and output param_value in real units.

    Expected regressor bundle format per operator:
      regressors_by_class[op] = (regressor, target_scaler)

    Where:
      - regressor.predict(X[i:i+1]) outputs "scaled(target)" (typically scaled param_norm_t)
      - target_scaler.inverse_transform(...) returns the target in transformed space (param_norm_t)
      - inv_t() converts param_norm_t -> param_norm (e.g., expm1 if log1p was used)
      - Finally:
          distance op: param_value = param_norm * extent_diag_m
          area op:     param_value = param_norm * extent_area_m2

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (N, D).
    df : pd.DataFrame
        Must contain diag_col and area_col.
    op_idx : np.ndarray
        Predicted or true operator indices (ints), shape (N,).
    class_names : Sequence[str]
        Class names in the same order as op_idx indices.
    regressors_by_class : Mapping[str, (regressor, target_scaler)]
        Trained per-operator regressors and their target scalers.
    inv_t : callable
        Inverse transform for target (e.g., expm1 if y was log1p).
    distance_ops / area_ops : Sequence[str]
        Operator groups.
    diag_col / area_col : str
        Column names for per-map dynamic extent references.
    clamp_nonneg : bool
        If True, clamp predicted param_norm to >= 0.

    Returns
    -------
    np.ndarray
        Predicted param_value in original units (meters or m²), shape (N,).
        NaN where prediction is not possible (missing regressor or missing refs).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D). Got shape {X.shape}")

    op_idx = np.asarray(op_idx).astype(int, copy=False).reshape(-1)
    if len(op_idx) != X.shape[0]:
        raise ValueError("op_idx length must match number of rows in X")

    if diag_col not in df.columns and area_col not in df.columns:
        raise KeyError(f"df must contain '{diag_col}' and '{area_col}'")

    cn = [str(x) for x in class_names]
    regs_norm = {str(k).strip().lower(): v for k, v in regressors_by_class.items()}

    dist_set = {str(x).strip().lower() for x in distance_ops}
    area_set = {str(x).strip().lower() for x in area_ops}

    # numeric refs
    extent_diag = pd.to_numeric(df[diag_col], errors="coerce").to_numpy(dtype=float)
    extent_area = pd.to_numeric(df[area_col], errors="coerce").to_numpy(dtype=float)

    yhat = np.full(X.shape[0], np.nan, dtype=float)

    for i in range(X.shape[0]):
        k = int(op_idx[i])
        if k < 0 or k >= len(cn):
            continue

        op = str(cn[k]).strip().lower()
        pack = regs_norm.get(op)
        if pack is None:
            continue

        reg, t_scaler = pack

        # model predicts in scaled target space
        pred_scaled = float(np.asarray(reg.predict(X[i:i+1])).reshape(-1)[0])

        # inverse target scaling -> transformed target space
        pred_t = float(t_scaler.inverse_transform([[pred_scaled]])[0, 0])

        # inverse optional transform -> param_norm
        pred_norm = float(inv_t(pred_t))

        if clamp_nonneg:
            pred_norm = max(0.0, pred_norm)

        # unnormalize to real units using per-row refs
        if op in dist_set:
            d = float(extent_diag[i])
            if np.isfinite(d) and d > 0:
                yhat[i] = pred_norm * d
        elif op in area_set:
            a = float(extent_area[i])
            if np.isfinite(a) and a > 0:
                yhat[i] = pred_norm * a
        else:
            # unknown operator group -> NaN
            continue

    return yhat
