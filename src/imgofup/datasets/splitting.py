from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from imgofup.config.constants import (
    OPERATOR_COL,
    INTENSITY_COL,
    MAPS_ID_COL,
    FIXED_OPERATOR_CLASSES,
    MAP_ID_WIDTH,
    NA_TOKENS,
    SPLIT_USE_INTENSITY_FOR_STRAT_DEFAULT,
    SPLIT_SEED_DEFAULT,
    SPLIT_VAL_RATIO_DEFAULT,
    SPLIT_TEST_RATIO_DEFAULT,
    SPLIT_MAX_ATTEMPTS_DEFAULT,
    SPLIT_VERBOSE_DEFAULT,
    SPLIT_TINY_SINGLE_MAP_THRESHOLD,
    SPLIT_STRAT_COL_NAME,
    SPLIT_STRAT_DELIM,
    SPLIT_SSS2_SEED_OFFSET,
    SPLIT_JSON_INDENT,
    SPLIT_PROMPTS_PER_MAP_MULTI_THRESHOLD,
)


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
    s = s.mask(s.isin(NA_TOKENS), pd.NA)
    return s


def _normalize_map_id_col(s: pd.Series) -> pd.Series:
    # robust zfill; keeps strings and handles numeric map ids
    out = s.astype(str).str.strip()
    out = out.mask(out.isin(NA_TOKENS | {"None"}), pd.NA)

    # if numeric-looking -> zfill(MAP_ID_WIDTH)
    m = out.notna() & out.str.fullmatch(r"\d+")
    out.loc[m] = out.loc[m].str.zfill(int(MAP_ID_WIDTH))

    # keep missingness
    out = out.fillna(pd.NA)
    return out


def _has_all_ops(dfx: pd.DataFrame, op_col: str, fixed_classes: Sequence[str]) -> bool:
    want = {str(x).strip().lower() for x in fixed_classes}
    got = set(dfx[op_col].dropna().unique())
    return got >= want


def make_splits_multi_prompt_to_train(
    *,
    df: pd.DataFrame,
    X: np.ndarray,
    op_col: str = OPERATOR_COL,
    intensity_col: Optional[str] = INTENSITY_COL,
    map_id_col: str = MAPS_ID_COL,
    fixed_classes: Sequence[str] = FIXED_OPERATOR_CLASSES,
    use_intensity_for_strat: bool = SPLIT_USE_INTENSITY_FOR_STRAT_DEFAULT,
    seed: int = SPLIT_SEED_DEFAULT,
    val_ratio: float = SPLIT_VAL_RATIO_DEFAULT,
    test_ratio: float = SPLIT_TEST_RATIO_DEFAULT,
    max_attempts: int = SPLIT_MAX_ATTEMPTS_DEFAULT,
    save_splits_json: Optional[Path] = None,
    verbose: bool = SPLIT_VERBOSE_DEFAULT,
) -> SplitResult:
    """
    Create train/val/test splits with constraints:
      - No leakage by map_id
      - All multi-prompt maps forced into TRAIN
      - VAL/TEST contain only single-prompt maps
      - Optionally stratify single-prompt maps by operator×intensity (fallback to operator-only if sparse)
      - Ensure each split contains all operators in fixed_classes

    Note: prompt_id is NOT used here and is assumed stable from Excel.
    """
    df = df.copy()

    if map_id_col not in df.columns:
        raise KeyError(f"Missing '{map_id_col}' in df.")
    if op_col not in df.columns:
        raise KeyError(f"Missing '{op_col}' in df.")

    # Normalize map_id defensively
    df[map_id_col] = _normalize_map_id_col(df[map_id_col])

    # clean operator + intensity columns
    df[op_col] = _clean_string_col(df[op_col])
    if intensity_col and intensity_col in df.columns:
        df[intensity_col] = _clean_string_col(df[intensity_col])

    # prompts per map
    prompt_counts = df.groupby(map_id_col).size()
    multi_map_ids = prompt_counts[prompt_counts > SPLIT_PROMPTS_PER_MAP_MULTI_THRESHOLD].index.astype(str).tolist()
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

    # ---------------------------
    # FALLBACK: tiny datasets
    # ---------------------------
    n_single = len(map_level)
    if n_single < SPLIT_TINY_SINGLE_MAP_THRESHOLD:
        if verbose:
            print(
                "\n⚠️ Fallback split activated: "
                f"only {n_single} single-prompt map(s) available.\n"
                "➡️ All data will be assigned to TRAIN.\n"
                "➡️ VAL and TEST will be empty.\n"
                "This is expected for very small/debug datasets."
            )

        train_maps = set(multi_map_ids) | set(single_map_ids)
        val_maps: set[str] = set()
        test_maps: set[str] = set()

        train_idx = df.index[df[map_id_col].isin(train_maps)].to_numpy()
        val_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)

        if save_splits_json is not None:
            save_splits_json = Path(save_splits_json)
            save_splits_json.parent.mkdir(parents=True, exist_ok=True)
            with open(save_splits_json, "w") as f:
                json.dump(
                    {
                        "train_idx": train_idx.tolist(),
                        "val_idx": [],
                        "test_idx": [],
                        "train_maps": sorted(list(train_maps)),
                        "val_maps": [],
                        "test_maps": [],
                        "seed_used": int(seed),
                        "use_intensity_for_strat_requested": bool(use_intensity_for_strat),
                        "use_intensity_for_strat_final": False,
                        "fixed_classes": [str(x) for x in fixed_classes],
                        "ratios": {"val_ratio": float(val_ratio), "test_ratio": float(test_ratio)},
                        "note": "fallback_all_train_due_to_tiny_dataset",
                    },
                    f,
                    indent=int(SPLIT_JSON_INDENT),
                )
            if verbose:
                print("\n✅ Saved fallback split to", str(save_splits_json))

        return SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_maps=sorted([str(x) for x in train_maps]),
            val_maps=[],
            test_maps=[],
            multi_map_ids=sorted([str(x) for x in multi_map_ids]),
            single_map_ids=sorted([str(x) for x in single_map_ids]),
            seed_used=int(seed),
            use_intensity_for_strat=False,
        )

    # strat label
    use_int_strat_final = bool(use_intensity_for_strat and intensity_col and intensity_col in map_level.columns)
    if use_int_strat_final:
        map_level[SPLIT_STRAT_COL_NAME] = (
            map_level[op_col].astype("string") + SPLIT_STRAT_DELIM + map_level[intensity_col].astype("string")
        )
        vc = map_level[SPLIT_STRAT_COL_NAME].value_counts()
        if (vc < 2).any():
            if verbose:
                print(
                    "\n⚠️ Some operator×intensity groups too rare (<2 single-maps). "
                    "Falling back to operator-only stratification."
                )
            map_level[SPLIT_STRAT_COL_NAME] = map_level[op_col]
            use_int_strat_final = False
    else:
        map_level[SPLIT_STRAT_COL_NAME] = map_level[op_col]
        use_int_strat_final = False

    # split ratios
    if not (0.0 < test_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in (0,1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    val_rel = val_ratio / (1.0 - test_ratio)

    X_idx = np.arange(len(map_level))
    y_strat = map_level[SPLIT_STRAT_COL_NAME].to_numpy()
    map_ids_arr = map_level[map_id_col].astype(str).to_numpy()

    best = None
    for attempt in range(int(max_attempts)):
        rs = int(seed) + attempt

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=float(test_ratio), random_state=rs)
        trainval_i, test_i = next(sss1.split(X_idx, y_strat))

        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=float(val_rel), random_state=rs + int(SPLIT_SSS2_SEED_OFFSET)
        )
        train_i, val_i = next(sss2.split(trainval_i, y_strat[trainval_i]))

        single_train_maps = set(map_ids_arr[trainval_i[train_i]])
        single_val_maps = set(map_ids_arr[trainval_i[val_i]])
        single_test_maps = set(map_ids_arr[test_i])

        train_maps = set(multi_map_ids) | single_train_maps
        val_maps = single_val_maps
        test_maps = single_test_maps

        if (train_maps & val_maps) or (train_maps & test_maps) or (val_maps & test_maps):
            continue

        df_train_tmp = df[df[map_id_col].isin(train_maps)]
        df_val_tmp = df[df[map_id_col].isin(val_maps)]
        df_test_tmp = df[df[map_id_col].isin(test_maps)]

        if not (
            _has_all_ops(df_train_tmp, op_col, fixed_classes)
            and _has_all_ops(df_val_tmp, op_col, fixed_classes)
            and _has_all_ops(df_test_tmp, op_col, fixed_classes)
        ):
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

    train_idx = df.index[df[map_id_col].isin(train_maps)].to_numpy()
    val_idx = df.index[df[map_id_col].isin(val_maps)].to_numpy()
    test_idx = df.index[df[map_id_col].isin(test_maps)].to_numpy()

    # save splits json
    if save_splits_json is not None:
        save_splits_json = Path(save_splits_json)
        save_splits_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_splits_json, "w") as f:
            json.dump(
                {
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                    "train_maps": sorted([str(x) for x in train_maps]),
                    "val_maps": sorted([str(x) for x in val_maps]),
                    "test_maps": sorted([str(x) for x in test_maps]),
                    "seed_used": int(used_seed),
                    "use_intensity_for_strat_requested": bool(use_intensity_for_strat),
                    "use_intensity_for_strat_final": bool(use_int_strat_final),
                    "fixed_classes": [str(x) for x in fixed_classes],
                    "ratios": {"val_ratio": float(val_ratio), "test_ratio": float(test_ratio)},
                },
                f,
                indent=int(SPLIT_JSON_INDENT),
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
