#src/train/utils/_classifier_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from imgofup.config.constants import HIDDEN_LAYER_CANDIDATES, BATCH_CANDIDATES


def fit_maybe_weighted(clf: Any, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray) -> bool:
    """Fit with sample_weight if supported; fallback otherwise."""
    try:
        clf.fit(X, y, sample_weight=sample_w)
        return True
    except TypeError:
        clf.fit(X, y)
        return False


def draw_mlp_params(rng: np.random.RandomState, n: int, *, random_state: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _ in range(n):
        out.append(
            {
                "hidden_layer_sizes": HIDDEN_LAYER_CANDIDATES[rng.randint(len(HIDDEN_LAYER_CANDIDATES))],
                "alpha": 10 ** rng.uniform(-5, np.log10(3e-2)),
                "learning_rate_init": 10 ** rng.uniform(-4, np.log10(3e-3)),
                "batch_size": BATCH_CANDIDATES[rng.randint(len(BATCH_CANDIDATES))],
                "activation": "relu",
                "solver": "adam",
                "max_iter": 800,
                "early_stopping": False,
                "random_state": int(random_state),
                "verbose": False,
                "tol": 1e-4,
            }
        )
    return out


def cv_macro_f1(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_w: np.ndarray,
    params: Dict[str, Any],
    cv: StratifiedGroupKFold,
) -> Tuple[float, float]:
    scores: List[float] = []
    for tr_idx, va_idx in cv.split(X, y, groups):
        clf = MLPClassifier(**params)
        fit_maybe_weighted(clf, X[tr_idx], y[tr_idx], sample_w[tr_idx])
        pred = clf.predict(X[va_idx])
        scores.append(float(f1_score(y[va_idx], pred, average="macro")))
    return float(np.mean(scores)), float(np.std(scores))


def candidate_sort_key(c: Dict[str, Any]) -> Tuple[float, float, float]:
    # higher is better
    return (float(c["val_f1"]), float(c["val_acc"]), float(c["cv_mean"]))
