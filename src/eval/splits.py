# src/eval/splits.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays into the original dataframe (no row reordering)."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_group_splits(
    df: pd.DataFrame,
    *,
    group_col: str = "map_id",
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitIndices:
    """
    Create (train, val, test) splits with NO group leakage.

    Splitting strategy:
      1) split off TEST from all rows (GroupShuffleSplit)
      2) split remaining into TRAIN/VAL (GroupShuffleSplit) with adjusted val size

    Parameters
    ----------
    df : pd.DataFrame
        Dataset table (rows align with your feature matrix).
    group_col : str
        Column containing grouping ids (e.g., 'map_id').
    seed : int
        Random seed.
    val_ratio : float
        Fraction of full dataset used for validation.
    test_ratio : float
        Fraction of full dataset used for testing.

    Returns
    -------
    SplitIndices
        train_idx, val_idx, test_idx arrays (dtype=int64).
    """
    if group_col not in df.columns:
        raise KeyError(f"Missing group_col='{group_col}' in dataframe columns: {list(df.columns)}")

    if not (0.0 < val_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in (0,1).")

    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    n = len(df)
    if n < 3:
        raise ValueError(f"Need at least 3 rows to split; got n={n}")

    groups = df[group_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups in '{group_col}' to split; got {n_groups}.")

    idx = np.arange(n, dtype=np.int64)

    # 1) split off test groups
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trv_idx, te_idx = next(gss1.split(idx, groups=groups))

    # 2) split remaining into train/val groups
    # val fraction relative to remaining pool
    val_size_rel = val_ratio / max(1e-12, (1.0 - test_ratio))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=seed + 1)

    tr_rel, va_rel = next(gss2.split(trv_idx, groups=groups[trv_idx]))
    tr_idx = trv_idx[tr_rel]
    va_idx = trv_idx[va_rel]

    # sanity: group disjointness
    g_tr = set(groups[tr_idx].tolist())
    g_va = set(groups[va_idx].tolist())
    g_te = set(groups[te_idx].tolist())
    if (g_tr & g_va) or (g_tr & g_te) or (g_va & g_te):
        raise RuntimeError("Group leakage detected in splits (this should not happen).")

    return SplitIndices(
        train_idx=np.asarray(tr_idx, dtype=np.int64),
        val_idx=np.asarray(va_idx, dtype=np.int64),
        test_idx=np.asarray(te_idx, dtype=np.int64),
    )
