from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from imgofup.config.constants import (
    MAPS_ID_COL,
    PROMPTS_PROMPT_ID_COL,
    PROMPTS_TEXT_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    PARAM_TARGET_NAME,
    TRAIN_REQUIRE_TEXT_DEFAULT,
    PAIRS_REQUIRED_KEY_COLS,
    LABELS_EXCEL_PROMPT_ID_FALLBACK,
    FeatureMode,
)

from imgofup.datasets.utils._load_training_data_utils import (
    build_valid_mask,
    clean_targets,
    compute_param_norm,
    load_and_filter_user_study_labels,
    merge_labels_onto_pairs,
    normalize_map_id,
    normalize_prompt_id,
    require_columns,
    resolve_artifact_paths,
)


@dataclass(frozen=True)
class LoadedTrainingData:
    X: np.ndarray
    df: pd.DataFrame
    exp_name: str
    feature_mode: str
    X_path: str
    pairs_path: str
    n_before_filter: int
    n_after_filter: int


def load_training_data_with_dynamic_param_norm(
    *,
    exp_name: str,
    feature_mode: FeatureMode,
    paths: Any,
    cfg: Any,  # currently unused; keep for API compatibility
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
    require_text: bool = TRAIN_REQUIRE_TEXT_DEFAULT,
    X_path: Optional[Union[str, Path]] = None,
    pairs_path: Optional[Union[str, Path]] = None,
) -> LoadedTrainingData:
    Xp, Pp = resolve_artifact_paths(
        exp_name=exp_name,
        feature_mode=str(feature_mode),
        paths=paths,
        X_path=X_path,
        pairs_path=pairs_path,
    )
    if not Xp.exists():
        raise FileNotFoundError(f"Missing features: {Xp}")
    if not Pp.exists():
        raise FileNotFoundError(f"Missing pairs: {Pp}")

    X = np.load(Xp)
    pairs_df = pd.read_parquet(Pp)

    # Required keys in the pairs parquet
    require_columns(pairs_df, list(PAIRS_REQUIRED_KEY_COLS), where=Pp.name)

    pairs_df = pairs_df.copy()
    pairs_df[PROMPTS_PROMPT_ID_COL] = normalize_prompt_id(pairs_df[PROMPTS_PROMPT_ID_COL])
    pairs_df[MAPS_ID_COL] = normalize_map_id(pairs_df[MAPS_ID_COL])

    if X.shape[0] != len(pairs_df):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but pairs has {len(pairs_df)} rows.")
    n_before = int(len(pairs_df))

    # Load/Filter labels from Excel user study file
    dfu = load_and_filter_user_study_labels(paths)

    prompt_id_src = getattr(paths, "PROMPT_ID_COL", LABELS_EXCEL_PROMPT_ID_FALLBACK)
    if prompt_id_src not in dfu.columns:
        raise KeyError(
            f"Excel sheet is missing required '{prompt_id_src}' column "
            f"(configure paths.PROMPT_ID_COL or rename the column to '{LABELS_EXCEL_PROMPT_ID_FALLBACK}')."
        )

    op_col = paths.OPERATOR_COL
    param_col = paths.PARAM_VALUE_COL

    require_columns(dfu, [prompt_id_src, paths.TILE_ID_COL, op_col, param_col], where="Excel labels")

    labels_cols = [prompt_id_src, paths.TILE_ID_COL, op_col, param_col]
    if getattr(paths, "INTENSITY_COL", None) in dfu.columns:
        labels_cols.append(paths.INTENSITY_COL)

    labels = dfu[labels_cols].copy()

    labels[PROMPTS_PROMPT_ID_COL] = normalize_prompt_id(labels[prompt_id_src])
    labels[MAPS_ID_COL] = normalize_map_id(labels[paths.TILE_ID_COL])

    keep_cols = [MAPS_ID_COL, PROMPTS_PROMPT_ID_COL, op_col, param_col]
    if getattr(paths, "INTENSITY_COL", None) in labels.columns:
        keep_cols.append(paths.INTENSITY_COL)
    labels = labels[keep_cols]

    # Merge labels onto pairs
    df = merge_labels_onto_pairs(pairs_df=pairs_df, labels=labels, op_col=op_col)

    # Require or normalize text
    if require_text:
        require_columns(df, [PROMPTS_TEXT_COL], where=Pp.name)
    if PROMPTS_TEXT_COL in df.columns:
        df[PROMPTS_TEXT_COL] = df[PROMPTS_TEXT_COL].astype("string")

    # Require extent columns and coerce numeric
    require_columns(df, [EXTENT_DIAG_COL, EXTENT_AREA_COL], where="Merged dataframe")
    df[EXTENT_DIAG_COL] = pd.to_numeric(df[EXTENT_DIAG_COL], errors="coerce")
    df[EXTENT_AREA_COL] = pd.to_numeric(df[EXTENT_AREA_COL], errors="coerce")

    # Clean target columns
    df = clean_targets(df, op_col=op_col, param_col=param_col)

    # Filter to valid rows and apply same mask to X
    mask = build_valid_mask(df, op_col=op_col, param_col=param_col)
    df = df.loc[mask].reset_index(drop=True)
    X = np.asarray(X[mask.to_numpy()], dtype=np.float64)

    # Compute dynamic normalized parameter (distance/area dependent)
    df[PARAM_TARGET_NAME] = compute_param_norm(
        df,
        op_col=op_col,
        param_col=param_col,
        distance_ops=distance_ops,
        area_ops=area_ops,
    )

    bad = int(df[PARAM_TARGET_NAME].isna().sum())
    if bad:
        raise RuntimeError(
            f"{PARAM_TARGET_NAME} has {bad} NaNs. Usually an operator is missing from DISTANCE_OPS/AREA_OPS."
        )

    return LoadedTrainingData(
        X=X,
        df=df,
        exp_name=exp_name,
        feature_mode=str(feature_mode),
        X_path=str(Xp),
        pairs_path=str(Pp),
        n_before_filter=n_before,
        n_after_filter=int(len(df)),
    )
