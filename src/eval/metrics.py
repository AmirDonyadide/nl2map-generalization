# src/eval/metrics.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
    include_report: bool = True,
    ignore_label: int = -1,   # NEW: allow ignoring unmapped labels
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    # optionally drop unlabeled rows (e.g., -1 from remapping with strict=False)
    if ignore_label is not None:
        m = (y_true != ignore_label) & (y_pred != ignore_label)
        y_true = y_true[m]
        y_pred = y_pred[m]

    if class_names is None:
        out: Dict[str, Any] = {
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "confusion": confusion_matrix(y_true, y_pred).tolist(),
        }
        if include_report:
            out["report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return out

    labels = np.arange(len(class_names))
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        # IMPORTANT: fix macro-F1 so missing classes count as 0 rather than disappearing
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        # IMPORTANT: fixed-size confusion matrix aligned to class_names
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if include_report:
        out["report"] = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(x) for x in class_names],
            output_dict=True,
            zero_division=0,
        )
    return out



def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE/RMSE on finite pairs only.
    Returns dict: {n, MAE, RMSE}.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))

    return {"n": int(n), "MAE": float(mae), "RMSE": float(rmse)}


def per_operator_regression_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    op_idx: np.ndarray,
    class_names: Sequence[str],
    split_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Per-operator MAE/RMSE.

    op_idx can be:
      - true operator indices (oracle grouping)
      - predicted operator indices (routing bucket grouping)

    Returns list of dict rows:
      {split, operator, n, MAE, RMSE}
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    op_idx = np.asarray(op_idx, dtype=int).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(op_idx)):
        raise ValueError("y_true, y_pred, and op_idx must have the same length")

    rows: List[Dict[str, Any]] = []
    for k, op in enumerate(class_names):
        m = (op_idx == k) & np.isfinite(y_true) & np.isfinite(y_pred)
        n = int(m.sum())
        if n == 0:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        rows.append(
            {
                "split": split_name,
                "operator": str(op),
                "n": n,
                "MAE": float(mae),
                "RMSE": float(rmse),
            }
        )
    return rows
