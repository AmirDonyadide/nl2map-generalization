from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.constants import (
    OPERATOR_COL,
    MAPS_ID_COL,
    FIXED_OPERATOR_CLASSES,
    USE_MAP_WEIGHT_DEFAULT,
    CLASS_WEIGHT_MODE_DEFAULT,
)

from .utils._labels_and_weights_utils import (
    compute_class_weights_from_train,
    compute_map_weights,
    encode_labels,
    normalize_fixed_classes,
)


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
    op_col: str = OPERATOR_COL,
    map_id_col: str = MAPS_ID_COL,
    fixed_classes: Sequence[str] = FIXED_OPERATOR_CLASSES,
    use_map_weight: bool = USE_MAP_WEIGHT_DEFAULT,
    class_weight_mode: str = CLASS_WEIGHT_MODE_DEFAULT,
) -> LabelsAndWeights:
    """
    Build fixed-order class labels for train/val/test and (optionally) sample weights.

    Sample weights combine:
      (A) class imbalance correction (sklearn compute_class_weight)
      (B) map-level prompt multiplicity correction:
          each map contributes ~1 total weight by scaling each sample by 1/count(map_id)
    """
    class_names = normalize_fixed_classes(fixed_classes)

    y_train = encode_labels(df_train, op_col=op_col, class_names=class_names, split_name="TRAIN")
    y_val = encode_labels(df_val, op_col=op_col, class_names=class_names, split_name="VAL")
    y_test = encode_labels(df_test, op_col=op_col, class_names=class_names, split_name="TEST")

    cls_w, class_weight_map = compute_class_weights_from_train(
        y_train,
        class_names=class_names,
        class_weight_mode=class_weight_mode,
    )

    w_class = cls_w[y_train].astype(np.float64)

    if use_map_weight:
        w_map = compute_map_weights(df_train, map_id_col=map_id_col)
    else:
        w_map = np.ones(len(df_train), dtype=np.float64)

    sample_w = (w_class * w_map).astype(np.float64)

    return LabelsAndWeights(
        class_names=list(class_names),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_w=sample_w,
        class_weight_map=class_weight_map,
    )
