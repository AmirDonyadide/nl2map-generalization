from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from imgofup.config.constants import CLASS_WEIGHT_MODE_DEFAULT


def normalize_str_series(s: pd.Series) -> pd.Series:
    """Lowercase+strip string labels; keeps NaNs as NaN."""
    return s.astype("string").str.strip().str.lower()


def normalize_fixed_classes(fixed_classes: Sequence[str]) -> List[str]:
    return [str(x).strip().lower() for x in fixed_classes]


def encode_labels(
    df: pd.DataFrame,
    *,
    op_col: str,
    class_names: Sequence[str],
    split_name: str,
) -> np.ndarray:
    if op_col not in df.columns:
        raise KeyError(f"{split_name}: missing column '{op_col}'.")

    ops = normalize_str_series(df[op_col])
    if ops.isna().any():
        bad_n = int(ops.isna().sum())
        raise ValueError(f"{split_name}: '{op_col}' contains {bad_n} missing values.")

    class_arr = np.array(list(class_names), dtype=object)

    codes = pd.Categorical(ops, categories=class_arr).codes
    if not (codes >= 0).all():
        unknown = sorted(set(ops[codes < 0].tolist()))
        raise ValueError(
            f"{split_name}: operator labels not in fixed_classes: {unknown}. "
            f"Allowed: {list(class_names)}"
        )
    return codes.astype(int)


def compute_class_weights_from_train(
    y_train: np.ndarray,
    *,
    class_names: Sequence[str],
    class_weight_mode: str = CLASS_WEIGHT_MODE_DEFAULT,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if y_train.size == 0:
        raise ValueError("TRAIN split is empty; cannot compute class weights.")

    classes = np.arange(len(class_names))
    cls_w = compute_class_weight(class_weight=class_weight_mode, classes=classes, y=y_train)
    cls_w = np.asarray(cls_w, dtype=np.float64)

    class_weight_map = {str(class_names[i]): float(cls_w[i]) for i in range(len(class_names))}
    return cls_w, class_weight_map


def compute_map_weights(df_train: pd.DataFrame, *, map_id_col: str) -> np.ndarray:
    if map_id_col not in df_train.columns:
        raise KeyError(f"TRAIN: missing column '{map_id_col}'.")

    ids = df_train[map_id_col]
    if ids.isna().any():
        bad_n = int(ids.isna().sum())
        raise ValueError(f"TRAIN: '{map_id_col}' contains {bad_n} missing values.")

    counts = ids.value_counts()
    return ids.map(lambda m: 1.0 / float(counts[m])).to_numpy(dtype=np.float64)
