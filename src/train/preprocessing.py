# src/train/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


FeatureMode = Literal["prompt_only", "prompt_plus_map"]


@dataclass(frozen=True)
class PreprocResult:
    X_train_s: np.ndarray
    X_val_s: np.ndarray
    X_test_s: np.ndarray
    bundle_path: str


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    nrm = np.sqrt((A * A).sum(axis=1, keepdims=True))
    return A / np.maximum(nrm, eps)


def _clip_to_q(A: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(A, lo, hi)


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
    save_path: Optional[Path] = None,
) -> PreprocResult:
    """
    Modality-aware preprocessing:
      - prompt_only:
          * L2-normalize prompt rows
      - prompt_plus_map:
          * split map/prompt
          * prompt: L2-normalize
          * map: replace inf->nan, median impute, clip by train quantiles, drop zero-variance, robust-scale
          * fuse back to [map | prompt]

    Saves a preprocessing bundle (if save_path provided) containing fitted transformers and masks.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val   = np.asarray(X_val, dtype=np.float64)
    X_test  = np.asarray(X_test, dtype=np.float64)

    # --- feature_mode: prompt_only ---
    if feature_mode == "prompt_only":
        X_train_s = l2_normalize_rows(X_train, eps=eps)
        X_val_s   = l2_normalize_rows(X_val, eps=eps)
        X_test_s  = l2_normalize_rows(X_test, eps=eps)

        if not (np.isfinite(X_train_s).all() and np.isfinite(X_val_s).all() and np.isfinite(X_test_s).all()):
            raise RuntimeError("Non-finite values after prompt-only preprocessing.")

        bundle = {
            "feature_mode": "prompt_only",
            "prompt_l2_eps": float(eps),
            "map_dim": int(map_dim),
            "prompt_dim": int(prompt_dim),
        }

        bundle_path = ""
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, save_path)
            bundle_path = str(save_path)

        return PreprocResult(X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s, bundle_path=bundle_path)

    # --- feature_mode: prompt_plus_map ---
    if feature_mode != "prompt_plus_map":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    if X_train.shape[1] < map_dim + prompt_dim:
        raise ValueError("X does not have enough columns for [map|prompt] split.")

    def split_blocks(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xm = X[:, :map_dim].astype(np.float64, copy=True)
        Xp = X[:, map_dim:map_dim + prompt_dim].astype(np.float64, copy=True)
        return Xm, Xp

    Xm_tr, Xp_tr = split_blocks(X_train)
    Xm_va, Xp_va = split_blocks(X_val)
    Xm_te, Xp_te = split_blocks(X_test)

    # prompts: L2 only
    Xp_tr = l2_normalize_rows(Xp_tr, eps=eps)
    Xp_va = l2_normalize_rows(Xp_va, eps=eps)
    Xp_te = l2_normalize_rows(Xp_te, eps=eps)

    # maps: inf -> NaN
    for A in (Xm_tr, Xm_va, Xm_te):
        A[~np.isfinite(A)] = np.nan

    # impute (train-fit)
    imp = SimpleImputer(strategy=impute_strategy)
    def _to_dense(A):
        # handles ndarray or sparse matrix safely
        if hasattr(A, "toarray"):
            return A.toarray()
        return np.asarray(A)

    Xm_tr_imp = _to_dense(imp.fit_transform(Xm_tr))
    Xm_va_imp = _to_dense(imp.transform(Xm_va))
    Xm_te_imp = _to_dense(imp.transform(Xm_te))

    # clip by train quantiles
    qlo, qhi = clip_q
    q_lo = np.nanpercentile(Xm_tr_imp, qlo, axis=0)
    q_hi = np.nanpercentile(Xm_tr_imp, qhi, axis=0)

    Xm_tr_imp = _clip_to_q(Xm_tr_imp, q_lo, q_hi)
    Xm_va_imp = _clip_to_q(Xm_va_imp, q_lo, q_hi)
    Xm_te_imp = _clip_to_q(Xm_te_imp, q_lo, q_hi)

    # drop zero-variance cols on train
    stds = np.nanstd(Xm_tr_imp, axis=0)
    keep_mask = stds > 1e-12

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=robust_qrange)
    Xm_tr_kept = scaler.fit_transform(Xm_tr_imp[:, keep_mask])
    Xm_va_kept = scaler.transform(Xm_va_imp[:, keep_mask])
    Xm_te_kept = scaler.transform(Xm_te_imp[:, keep_mask])

    # rebuild full map block with dropped cols = 0
    Xm_tr_s = np.zeros_like(Xm_tr_imp, dtype=np.float64)
    Xm_va_s = np.zeros_like(Xm_va_imp, dtype=np.float64)
    Xm_te_s = np.zeros_like(Xm_te_imp, dtype=np.float64)

    Xm_tr_s[:, keep_mask] = Xm_tr_kept
    Xm_va_s[:, keep_mask] = Xm_va_kept
    Xm_te_s[:, keep_mask] = Xm_te_kept

    # fuse back
    X_train_s = np.concatenate([Xm_tr_s, Xp_tr], axis=1).astype(np.float64)
    X_val_s   = np.concatenate([Xm_va_s, Xp_va], axis=1).astype(np.float64)
    X_test_s  = np.concatenate([Xm_te_s, Xp_te], axis=1).astype(np.float64)

    if not (np.isfinite(X_train_s).all() and np.isfinite(X_val_s).all() and np.isfinite(X_test_s).all()):
        raise RuntimeError("Non-finite values after prompt+map preprocessing.")

    bundle = {
        "feature_mode": "prompt_plus_map",
        "imp": imp,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "clip_quantiles": tuple(map(int, clip_q)),
        "keep_mask": keep_mask,
        "scaler": scaler,
        "map_dim": int(map_dim),
        "prompt_dim": int(prompt_dim),
        "prompt_l2_eps": float(eps),
        "impute_strategy": str(impute_strategy),
        "robust_qrange": tuple(map(int, robust_qrange)),
    }

    bundle_path = ""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, save_path)
        bundle_path = str(save_path)

    return PreprocResult(X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s, bundle_path=bundle_path)
