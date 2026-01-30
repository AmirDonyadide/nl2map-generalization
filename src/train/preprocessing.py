#src/train/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from src.types import FeatureMode
from .utils._preprocessing_utils import (
    fit_map_preproc,
    l2_normalize_rows,
    split_map_prompt,
    transform_map_preproc,
)


@dataclass(frozen=True)
class PreprocResult:
    X_train_s: np.ndarray
    X_val_s: np.ndarray
    X_test_s: np.ndarray
    bundle_path: str


def fit_transform_modality_preproc(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_mode: FeatureMode,
    map_dim: int,
    prompt_dim: int,
    eps: float = 1e-12,
    clip_q: Tuple[int, int] = (5, 95),
    impute_strategy: str = "median",
    robust_qrange: Tuple[int, int] = (5, 95),
    var_eps: float = 1e-12,
    save_path: Optional[Path] = None,
) -> PreprocResult:
    """
    Modality-aware preprocessing:
      - prompt_only:
          * L2-normalize prompt rows
      - prompt_plus_map:
          * split map/prompt
          * prompt: L2-normalize
          * map: inf->nan, median impute, clip by train quantiles, drop near-zero variance, robust-scale
          * fuse back to [map | prompt]

    Saves a preprocessing bundle (if save_path provided).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    # shape checks
    if X_val.shape[1] != X_train.shape[1] or X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_train/X_val/X_test must have the same number of columns.")

    # --- prompt_only ---
    if feature_mode == "prompt_only":
        X_train_s = l2_normalize_rows(X_train, eps=eps)
        X_val_s = l2_normalize_rows(X_val, eps=eps)
        X_test_s = l2_normalize_rows(X_test, eps=eps)

        if not (np.isfinite(X_train_s).all() and np.isfinite(X_val_s).all() and np.isfinite(X_test_s).all()):
            raise RuntimeError("Non-finite values after prompt-only preprocessing.")

        bundle = {
            "feature_mode": "prompt_only",
            "prompt_l2_eps": float(eps),
            "map_dim": int(map_dim),
            "prompt_dim": int(prompt_dim),
        }
        return _maybe_save_and_wrap(X_train_s, X_val_s, X_test_s, bundle, save_path)

    # --- prompt_plus_map ---
    if feature_mode != "prompt_plus_map":
        raise ValueError(f"Unsupported feature_mode for this preproc: {feature_mode}")

    if X_train.shape[1] < map_dim + prompt_dim:
        raise ValueError("X does not have enough columns for [map|prompt] split.")

    Xm_tr, Xp_tr = split_map_prompt(X_train, map_dim=map_dim, prompt_dim=prompt_dim)
    Xm_va, Xp_va = split_map_prompt(X_val, map_dim=map_dim, prompt_dim=prompt_dim)
    Xm_te, Xp_te = split_map_prompt(X_test, map_dim=map_dim, prompt_dim=prompt_dim)

    # prompt block
    Xp_tr = l2_normalize_rows(Xp_tr, eps=eps)
    Xp_va = l2_normalize_rows(Xp_va, eps=eps)
    Xp_te = l2_normalize_rows(Xp_te, eps=eps)

    # map block (fit on train only)
    _, fitted = fit_map_preproc(
        Xm_tr,
        clip_q=clip_q,
        impute_strategy=impute_strategy,
        robust_qrange=robust_qrange,
        var_eps=var_eps,
    )

    Xm_tr_s = transform_map_preproc(Xm_tr, fitted, clip_q=clip_q)
    Xm_va_s = transform_map_preproc(Xm_va, fitted, clip_q=clip_q)
    Xm_te_s = transform_map_preproc(Xm_te, fitted, clip_q=clip_q)

    X_train_s = np.concatenate([Xm_tr_s, Xp_tr], axis=1).astype(np.float64)
    X_val_s = np.concatenate([Xm_va_s, Xp_va], axis=1).astype(np.float64)
    X_test_s = np.concatenate([Xm_te_s, Xp_te], axis=1).astype(np.float64)

    if not (np.isfinite(X_train_s).all() and np.isfinite(X_val_s).all() and np.isfinite(X_test_s).all()):
        raise RuntimeError("Non-finite values after prompt+map preprocessing.")

    bundle = {
        "feature_mode": "prompt_plus_map",
        "map_preproc": fitted,
        "clip_quantiles": tuple(map(int, clip_q)),
        "map_dim": int(map_dim),
        "prompt_dim": int(prompt_dim),
        "prompt_l2_eps": float(eps),
        "impute_strategy": str(impute_strategy),
        "robust_qrange": tuple(map(int, robust_qrange)),
        "var_eps": float(var_eps),
    }
    return _maybe_save_and_wrap(X_train_s, X_val_s, X_test_s, bundle, save_path)


def _maybe_save_and_wrap(
    X_train_s: np.ndarray,
    X_val_s: np.ndarray,
    X_test_s: np.ndarray,
    bundle: dict,
    save_path: Optional[Path],
) -> PreprocResult:
    bundle_path = ""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, save_path)
        bundle_path = str(save_path)

    return PreprocResult(X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s, bundle_path=bundle_path)
