from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    nrm = np.sqrt((A * A).sum(axis=1, keepdims=True))
    return A / np.maximum(nrm, eps)


def clip_to_q(A: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(A, lo, hi)


def to_dense(A):
    # sklearn transformers sometimes return sparse
    if hasattr(A, "toarray"):
        return A.toarray()
    return np.asarray(A)


def split_map_prompt(X: np.ndarray, *, map_dim: int, prompt_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    Xm = X[:, :map_dim].astype(np.float64, copy=True)
    Xp = X[:, map_dim:map_dim + prompt_dim].astype(np.float64, copy=True)
    return Xm, Xp


@dataclass(frozen=True)
class MapPreprocFitted:
    imp: SimpleImputer
    q_lo: np.ndarray
    q_hi: np.ndarray
    keep_mask: np.ndarray
    scaler: RobustScaler


def fit_map_preproc(
    Xm_tr: np.ndarray,
    *,
    clip_q: Tuple[int, int],
    impute_strategy: str,
    robust_qrange: Tuple[int, int],
    var_eps: float,
) -> Tuple[np.ndarray, MapPreprocFitted]:
    # inf -> NaN
    Xm_tr = Xm_tr.copy()
    Xm_tr[~np.isfinite(Xm_tr)] = np.nan

    imp = SimpleImputer(strategy=impute_strategy)
    Xm_tr_imp = to_dense(imp.fit_transform(Xm_tr))

    qlo, qhi = clip_q
    q_lo = np.nanpercentile(Xm_tr_imp, qlo, axis=0)
    q_hi = np.nanpercentile(Xm_tr_imp, qhi, axis=0)
    Xm_tr_imp = clip_to_q(Xm_tr_imp, q_lo, q_hi)

    stds = np.nanstd(Xm_tr_imp, axis=0)
    keep_mask = stds > var_eps

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=robust_qrange)
    Xm_tr_kept = scaler.fit_transform(Xm_tr_imp[:, keep_mask])

    fitted = MapPreprocFitted(imp=imp, q_lo=q_lo, q_hi=q_hi, keep_mask=keep_mask, scaler=scaler)
    return Xm_tr_kept, fitted


def transform_map_preproc(Xm: np.ndarray, fitted: MapPreprocFitted, *, clip_q: Tuple[int, int]) -> np.ndarray:
    Xm = Xm.copy()
    Xm[~np.isfinite(Xm)] = np.nan

    Xm_imp = to_dense(fitted.imp.transform(Xm))
    Xm_imp = clip_to_q(Xm_imp, fitted.q_lo, fitted.q_hi)

    Xm_kept = fitted.scaler.transform(Xm_imp[:, fitted.keep_mask])

    # rebuild full map block (dropped cols = 0)
    Xm_s = np.zeros_like(Xm_imp, dtype=np.float64)
    Xm_s[:, fitted.keep_mask] = Xm_kept
    return Xm_s
