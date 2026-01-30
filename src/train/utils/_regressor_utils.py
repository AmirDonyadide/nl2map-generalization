# src/train/utils/_regressor_utils.py
from __future__ import annotations

from typing import Any

import numpy as np


def fit_reg_maybe_weighted(reg: Any, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray) -> bool:
    """Fit with sample_weight if supported; fallback otherwise."""
    try:
        reg.fit(X, y, sample_weight=sample_w)
        return True
    except TypeError:
        reg.fit(X, y)
        return False
