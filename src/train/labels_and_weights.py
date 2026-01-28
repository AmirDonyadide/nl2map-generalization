# src/train/labels_and_weights.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class LabelsAndWeights:
    class_names: List[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    sample_w: np.ndarray
    class_weight_map: Dict[str, float]


def build_labels_and_sample_weights(
    *,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    op_col: str = "operator",
    map_id_col: str = "map_id",
    fixed_classes: Sequence[str] = ("simplify", "select", "aggregate", "displace"),
    use_map_weight: bool = True,
    class_weight_mode: str = "balanced",
) -> LabelsAndWeights:
    """
    Build fixed-order class labels for train/val/test and (optionally) sample weights.

    Sample weights combine:
      (A) class imbalance correction (sklearn compute_class_weight)
      (B) map-level prompt multiplicity correction:
          each map contributes ~1 total weight by scaling each sample by 1/count(map_id)

    Returns:
      - class_names
      - y_train, y_val, y_test (int codes)
      - sample_w (float64, length len(df_train))
      - class_weight_map
    """
    class_names = [str(x).strip().lower() for x in fixed_classes]
    class_arr = np.array(class_names)

    # labels
    y_train = pd.Categorical(df_train[op_col], categories=class_arr).codes
    y_val = pd.Categorical(df_val[op_col], categories=class_arr).codes
    y_test = pd.Categorical(df_test[op_col], categories=class_arr).codes

    # Safety checks
    if not (y_train >= 0).all():
        raise ValueError("TRAIN contains operator labels not in fixed_classes.")
    if not (y_val >= 0).all():
        raise ValueError("VAL contains operator labels not in fixed_classes.")
    if not (y_test >= 0).all():
        raise ValueError("TEST contains operator labels not in fixed_classes.")

    # class weights from training distribution only
    classes = np.arange(len(class_names))
    cls_w = compute_class_weight(class_weight=class_weight_mode, classes=classes, y=y_train)
    cls_w = np.asarray(cls_w, dtype=np.float64)

    class_weight_map = {class_names[i]: float(cls_w[i]) for i in range(len(class_names))}

    # per-sample class weights
    w_class = cls_w[y_train].astype(np.float64)

    # map-level multiplicity correction (train only)
    if use_map_weight:
        map_counts = df_train[map_id_col].value_counts()
        w_map = df_train[map_id_col].map(lambda m: 1.0 / float(map_counts[m])).to_numpy(dtype=np.float64)
    else:
        w_map = np.ones(len(df_train), dtype=np.float64)

    sample_w = (w_class * w_map).astype(np.float64)

    return LabelsAndWeights(
        class_names=class_names,
        y_train=y_train.astype(int),
        y_val=y_val.astype(int),
        y_test=y_test.astype(int),
        sample_w=sample_w,
        class_weight_map=class_weight_map,
    )
