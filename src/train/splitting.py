# src/train/splitting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_maps: List[str]
    val_maps: List[str]
    test_maps: List[str]
    multi_map_ids: List[str]
    single_map_ids: List[str]
    seed_used: int
    use_intensity_for_strat: bool


def _clean_string_col(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    s = s.mask(s.isin(["", "nan"]), pd.NA)
    return s


def _has_all_ops(dfx: pd.DataFrame, op_col: str, fixed_classes: Sequence[str]) -> bool:
    return set(dfx[op_col].dropna().unique()) >= set([str(x).strip().lower() for x in fixed_classes])


def make_splits_multi_prompt_to_train(
    *,
    df: pd.DataFrame,
    X: np.ndarray,
    op_col: str = "operator",
    intensity_col: Optional[str] = "intensity",
    map_id_col: str = "map_id",
    fixed_classes: Sequence[str] = ("simplify", "select", "aggregate", "displace"),
    use_intensity_for_strat: bool = True,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_attempts: int = 500,
    save_splits_json: Optional[Path] = None,
    verbose: bool = True,
) -> SplitResult:
    """
    Create train/val/test splits with the following constraints:
      - No leakage across splits by map_id
      - All multi-prompt maps are forced into TRAIN
      - VAL/TEST contain only single-prompt maps
      - Optionally stratify single-prompt maps by operator×intensity (fallback to operator-only if sparse)
      - Ensure each split contains all operators in fixed_classes

    Returns row indices into df/X.
    """
    df = df.copy()

    if map_id_col not in df.columns:
        raise KeyError(f"Missing '{map_id_col}' in df.")
    if op_col not in df.columns:
        raise KeyError(f"Missing '{op_col}' in df.")

    # clean operator + intensity columns
    df[op_col] = _clean_string_col(df[op_col])
    if intensity_col and intensity_col in df.columns:
        df[intensity_col] = _clean_string_col(df[intensity_col])

    # prompts per map
    prompt_counts = df.groupby(map_id_col).size()
    multi_map_ids = prompt_counts[prompt_counts > 1].index.astype(str).tolist()
    single_map_ids = prompt_counts[prompt_counts == 1].index.astype(str).tolist()

    if verbose:
        print("=== DATASET SUMMARY ===")
        print(f"Total rows (prompts): {len(df)}")
        print(f"Unique maps: {prompt_counts.shape[0]}")
        print(f"Multi-prompt maps (>1 prompt): {len(multi_map_ids)}")
        print(f"Single-prompt maps (=1 prompt): {len(single_map_ids)}")
        print("\nTop 10 maps by prompt count:")
        print(prompt_counts.sort_values(ascending=False).head(10))

    # map-level table for single maps (one row per map_id)
    df_single = df[df[map_id_col].isin(single_map_ids)].copy()
    map_level = df_single.groupby(map_id_col).first().reset_index()
    map_level = map_level.dropna(subset=[op_col]).copy()

    # strat label
    use_int_strat_final = bool(use_intensity_for_strat and intensity_col and intensity_col in map_level.columns)
    if use_int_strat_final:
        map_level["_strat"] = (
            map_level[op_col].astype("string")
            + "__"
            + map_level[intensity_col].astype("string")
        )
        vc = map_level["_strat"].value_counts()
        if (vc < 2).any():
            if verbose:
                print("\n⚠️ Some operator×intensity groups too rare (<2 single-maps). Falling back to operator-only stratification.")
            map_level["_strat"] = map_level[op_col]
            use_int_strat_final = False
    else:
        map_level["_strat"] = map_level[op_col]
        use_int_strat_final = False

    # split ratios
    if not (0.0 < test_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in (0,1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    val_rel = val_ratio / (1.0 - test_ratio)

    X_idx = np.arange(len(map_level))
    y_strat = map_level["_strat"].to_numpy()
    map_ids_arr = map_level[map_id_col].astype(str).to_numpy()

    best = None
    for attempt in range(max_attempts):
        rs = int(seed) + attempt

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=rs)
        trainval_i, test_i = next(sss1.split(X_idx, y_strat))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=rs + 999)
        train_i, val_i = next(sss2.split(trainval_i, y_strat[trainval_i]))

        single_train_maps = set(map_ids_arr[trainval_i[train_i]])
        single_val_maps = set(map_ids_arr[trainval_i[val_i]])
        single_test_maps = set(map_ids_arr[test_i])

        train_maps = set(multi_map_ids) | single_train_maps
        val_maps = single_val_maps
        test_maps = single_test_maps

        # leakage check
        if (train_maps & val_maps) or (train_maps & test_maps) or (val_maps & test_maps):
            continue

        df_train_tmp = df[df[map_id_col].isin(train_maps)]
        df_val_tmp = df[df[map_id_col].isin(val_maps)]
        df_test_tmp = df[df[map_id_col].isin(test_maps)]

        # must contain all operators in each split
        if not (_has_all_ops(df_train_tmp, op_col, fixed_classes)
                and _has_all_ops(df_val_tmp, op_col, fixed_classes)
                and _has_all_ops(df_test_tmp, op_col, fixed_classes)):
            continue

        best = (train_maps, val_maps, test_maps, rs)
        break

    if best is None:
        raise RuntimeError(
            "Could not find a leakage-safe split with operator coverage in all splits "
            "and multi-prompt maps forced to TRAIN. "
            "Try: use_intensity_for_strat=False, adjust VAL/TEST ratios, or reduce constraints."
        )

    train_maps, val_maps, test_maps, used_seed = best

    # row-level indices
    train_idx = df.index[df[map_id_col].isin(train_maps)].to_numpy()
    val_idx = df.index[df[map_id_col].isin(val_maps)].to_numpy()
    test_idx = df.index[df[map_id_col].isin(test_maps)].to_numpy()

    # hard guarantees
    if not set(df.loc[train_idx, map_id_col]).isdisjoint(df.loc[val_idx, map_id_col]):
        raise RuntimeError("Leakage: train and val share map_ids.")
    if not set(df.loc[train_idx, map_id_col]).isdisjoint(df.loc[test_idx, map_id_col]):
        raise RuntimeError("Leakage: train and test share map_ids.")
    if not set(df.loc[val_idx, map_id_col]).isdisjoint(df.loc[test_idx, map_id_col]):
        raise RuntimeError("Leakage: val and test share map_ids.")
    if not set(multi_map_ids).issubset(set(train_maps)):
        raise RuntimeError("Constraint violated: not all multi-prompt maps are in TRAIN.")

    if verbose:
        print("\n=== SPLIT SUMMARY ===")
        print(f"✅ Split found (seed={used_seed})")
        print(f"Train maps: {len(train_maps)}  (includes multi-prompt maps: {len(set(multi_map_ids))})")
        print(f"Val maps:   {len(val_maps)}")
        print(f"Test maps:  {len(test_maps)}")
        print("✅ Verified: no map_id leakage across splits.")
        print("✅ Verified: all multi-prompt maps are in TRAIN.")

        print("\nTRAIN — Operator counts")
        print(df.loc[train_idx, op_col].value_counts())
        print("\nVAL — Operator counts")
        print(df.loc[val_idx, op_col].value_counts())
        print("\nTEST — Operator counts")
        print(df.loc[test_idx, op_col].value_counts())

        if intensity_col and intensity_col in df.columns:
            def _tab(dfx, name):
                print(f"\n{name} — Operator × Intensity table (counts)")
                tab = dfx.groupby([op_col, intensity_col]).size().unstack(fill_value=0).sort_index()
                print(tab)
            _tab(df.loc[train_idx], "TRAIN")
            _tab(df.loc[val_idx], "VAL")
            _tab(df.loc[test_idx], "TEST")

    # save splits json
    if save_splits_json is not None:
        save_splits_json = Path(save_splits_json)
        save_splits_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "seed_used": int(used_seed),
                "use_intensity_for_strat_requested": bool(use_intensity_for_strat),
                "use_intensity_for_strat_final": bool(use_int_strat_final),
                "fixed_classes": [str(x) for x in fixed_classes],
                "ratios": {"val_ratio": float(val_ratio), "test_ratio": float(test_ratio)},
            },
            open(save_splits_json, "w"),
            indent=2,
        )
        if verbose:
            print("\n✅ Saved splits to", str(save_splits_json))

    return SplitResult(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        train_maps=sorted([str(x) for x in train_maps]),
        val_maps=sorted([str(x) for x in val_maps]),
        test_maps=sorted([str(x) for x in test_maps]),
        multi_map_ids=sorted([str(x) for x in multi_map_ids]),
        single_map_ids=sorted([str(x) for x in single_map_ids]),
        seed_used=int(used_seed),
        use_intensity_for_strat=bool(use_int_strat_final),
    )
