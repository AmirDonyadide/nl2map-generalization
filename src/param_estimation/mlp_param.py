# src/param_estimation/mlp_param.py
"""
MLP-based parameter estimation wrapper.

This module wraps your trained per-operator regressors (MLPRegressor + target scaler)
and converts their output into:

  - param_norm (normalized)
  - param_value (original units: meters for distance ops, m² for select)

Assumptions about your saved bundle (cls_plus_regressors.joblib)
---------------------------------------------------------------
bundle = {
  "classifier": <sklearn classifier>,               # optional for this module
  "regressors_by_class": {op: (reg, t_scaler), ...},
  "class_names": [...],                             # list of operator names in class order
  "use_log1p": False,
  "target": "param_norm",                           # optional string marker
  "normalization": {                                # optional (we can ignore it)
      "distance_ops": [...],
      "area_ops": [...],
      ...
  }
}

Note:
- Your regressors were trained to predict *scaled(param_norm_transformed)* where:
    yk_s = StandardScaler().fit_transform(yk.reshape(-1,1)).ravel()
  so the regressor outputs a value in "scaled" space, and we must inverse_transform with t_scaler.
- If you used log1p on param_norm, we also inverse it.

Map refs
--------
At inference time you must provide per-sample references:
  - for distance ops: diag_ref (extent_diag_m or tile_diag_m)
  - for area ops:     area_ref (extent_area_m2 or tile_area_m2)

This module accepts map_refs with keys:
  - extent_diag_m / extent_area_m2 (preferred)
  - tile_diag_m   / tile_area_m2   (fallback)
  - diag_m / area_m2               (aliases)

API
---
- estimate_param_mlp_from_bundle(bundle, X_fused, operator_pred, map_refs_batch, ...)
- estimate_param_mlp(X_fused, operator_pred, regressors_by_class, map_refs_batch, ...)

"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, Callable


DEFAULT_DISTANCE_OPS = {"aggregate", "displace", "simplify"}
DEFAULT_AREA_OPS = {"select"}


def _pick_first_float(d: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                continue
    return None


def _get_refs_for_row(map_ref: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    diag = _pick_first_float(map_ref, ["extent_diag_m", "diag_m", "tile_diag_m"])
    area = _pick_first_float(map_ref, ["extent_area_m2", "area_m2", "tile_area_m2"])
    return diag, area


def _inv_log1p(x: np.ndarray) -> np.ndarray:
    return np.expm1(x)


def estimate_param_mlp(
    X_fused: np.ndarray,
    operator_pred: Sequence[str],
    regressors_by_class: Dict[str, Tuple[Any, Any]],
    map_refs_batch: Sequence[Dict[str, Any]],
    *,
    use_log1p: bool = False,
    distance_ops: Optional[set[str]] = None,
    area_ops: Optional[set[str]] = None,
    allow_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Predict param_norm and param_value from features + operator routing.

    Parameters
    ----------
    X_fused:
      (N, D) feature matrix (map emb | prompt emb), already preprocessed same as training.
    operator_pred:
      length N list/array of operator names (strings).
    regressors_by_class:
      dict: op -> (regressor, target_scaler)
      regressor.predict returns scaled(param_norm_transformed)
      target_scaler.inverse_transform converts back to param_norm_transformed
    map_refs_batch:
      length N list of dicts; each dict contains diag/area references for that map_id.
    use_log1p:
      if True, undo log1p after inverse scaling.
    distance_ops / area_ops:
      operator groups. Default uses STANDARD grouping.
    allow_nan:
      if False, raises if any prediction cannot be produced.

    Returns
    -------
    param_norm: np.ndarray shape (N,)
    param_value: np.ndarray shape (N,)
    debug: dict with counts and warnings
    """
    X = np.asarray(X_fused)
    if X.ndim != 2:
        raise ValueError(f"X_fused must be 2D (N,D). Got shape {X.shape}")

    ops = np.asarray([str(o).strip().lower() for o in operator_pred])
    if len(ops) != X.shape[0]:
        raise ValueError("operator_pred length must match X rows")

    if len(map_refs_batch) != X.shape[0]:
        raise ValueError("map_refs_batch length must match X rows")

    dist_ops = distance_ops or set(DEFAULT_DISTANCE_OPS)
    ar_ops = area_ops or set(DEFAULT_AREA_OPS)

    # outputs
    param_norm = np.full(X.shape[0], np.nan, dtype=float)
    param_value = np.full(X.shape[0], np.nan, dtype=float)

    debug: Dict[str, Any] = {
        "n": int(X.shape[0]),
        "missing_regressor": 0,
        "missing_refs": 0,
        "unknown_operator_group": 0,
        "warnings": [],
    }

    # batch by operator for efficiency
    for op in np.unique(ops):
        idx = np.where(ops == op)[0]
        if idx.size == 0:
            continue

        pack = regressors_by_class.get(op)
        if pack is None:
            debug["missing_regressor"] += int(idx.size)
            continue

        reg, t_scaler = pack

        # predict scaled y in one go
        y_scaled = np.asarray(reg.predict(X[idx])).reshape(-1, 1)

        # inverse target scaling -> param_norm_transformed
        try:
            y_t = t_scaler.inverse_transform(y_scaled).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Failed inverse_transform for op={op}: {e}")

        if use_log1p:
            y_norm = _inv_log1p(y_t)
        else:
            y_norm = y_t

        param_norm[idx] = y_norm

        # convert to real units using per-row refs
        for j, row_i in enumerate(idx):
            diag_ref, area_ref = _get_refs_for_row(map_refs_batch[row_i])

            if op in dist_ops:
                if diag_ref is None:
                    debug["missing_refs"] += 1
                    continue
                param_value[row_i] = float(y_norm[j]) * float(diag_ref)

            elif op in ar_ops:
                if area_ref is None:
                    debug["missing_refs"] += 1
                    continue
                param_value[row_i] = float(y_norm[j]) * float(area_ref)

            else:
                debug["unknown_operator_group"] += 1
                continue

    if not allow_nan:
        if np.isnan(param_norm).any() or np.isnan(param_value).any():
            raise ValueError(
                "Some predictions are NaN. "
                f"missing_regressor={debug['missing_regressor']}, "
                f"missing_refs={debug['missing_refs']}, "
                f"unknown_operator_group={debug['unknown_operator_group']}"
            )

    return param_norm, param_value, debug


def estimate_param_mlp_from_bundle(
    bundle: Dict[str, Any],
    X_fused: np.ndarray,
    operator_pred: Union[Sequence[str], Sequence[int], np.ndarray],
    map_refs_batch: Sequence[Dict[str, Any]],
    *,
    allow_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper if you loaded cls_plus_regressors.joblib.

    operator_pred can be:
      - list of operator strings, OR
      - list/array of class indices (ints). If ints, we map them through bundle["class_names"].
    """
    regs = bundle.get("regressors_by_class")
    if regs is None:
        raise ValueError("bundle missing 'regressors_by_class'")

    class_names = bundle.get("class_names")
    use_log1p = bool(bundle.get("use_log1p", False))

    # operator groups: prefer bundle["normalization"] if present
    norm_meta = bundle.get("normalization") or {}
    dist_ops = set(norm_meta.get("distance_ops", DEFAULT_DISTANCE_OPS))
    ar_ops = set(norm_meta.get("area_ops", DEFAULT_AREA_OPS))

    # map operator_pred to strings if given indices
    op_arr = np.asarray(operator_pred)
    if op_arr.dtype.kind in {"i", "u"}:
        if not class_names:
            raise ValueError("operator_pred are indices but bundle has no class_names")
        cn = [str(x).strip().lower() for x in class_names]
        ops = [cn[int(i)] for i in op_arr.tolist()]
    else:
        ops = [str(x).strip().lower() for x in op_arr.tolist()]

    return estimate_param_mlp(
        X_fused=X_fused,
        operator_pred=ops,
        regressors_by_class=regs,
        map_refs_batch=map_refs_batch,
        use_log1p=use_log1p,
        distance_ops=dist_ops,
        area_ops=ar_ops,
        allow_nan=allow_nan,
    )
