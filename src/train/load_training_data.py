from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .utils._load_training_data_utils import (
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

from src.types import FeatureMode

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
    require_text: bool = True,
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

    require_columns(pairs_df, ["prompt_id", "map_id"], where=Pp.name)

    pairs_df = pairs_df.copy()
    pairs_df["prompt_id"] = normalize_prompt_id(pairs_df["prompt_id"])
    pairs_df["map_id"] = normalize_map_id(pairs_df["map_id"])

    if X.shape[0] != len(pairs_df):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but pairs has {len(pairs_df)} rows.")
    n_before = int(len(pairs_df))

    dfu = load_and_filter_user_study_labels(paths)

    prompt_id_src = getattr(paths, "PROMPT_ID_COL", "prompt_id")
    if prompt_id_src not in dfu.columns:
        raise KeyError(
            f"Excel sheet is missing required '{prompt_id_src}' column "
            "(configure paths.PROMPT_ID_COL or rename the column to 'prompt_id')."
        )

    op_col = paths.OPERATOR_COL
    param_col = paths.PARAM_VALUE_COL

    require_columns(dfu, [prompt_id_src, paths.TILE_ID_COL, op_col, param_col], where="Excel labels")

    labels = dfu[[prompt_id_src, paths.TILE_ID_COL, op_col, param_col] + (
        [paths.INTENSITY_COL] if getattr(paths, "INTENSITY_COL", None) in dfu.columns else []
    )].copy()

    labels["prompt_id"] = normalize_prompt_id(labels[prompt_id_src])
    labels["map_id"] = normalize_map_id(labels[paths.TILE_ID_COL])

    keep_cols = ["map_id", "prompt_id", op_col, param_col]
    if getattr(paths, "INTENSITY_COL", None) in labels.columns:
        keep_cols.append(paths.INTENSITY_COL)
    labels = labels[keep_cols]

    df = merge_labels_onto_pairs(pairs_df=pairs_df, labels=labels, op_col=op_col)

    if require_text:
        require_columns(df, ["text"], where=Pp.name)
    if "text" in df.columns:
        df["text"] = df["text"].astype("string")

    require_columns(df, ["extent_diag_m", "extent_area_m2"], where="Merged dataframe")
    df["extent_diag_m"] = pd.to_numeric(df["extent_diag_m"], errors="coerce")
    df["extent_area_m2"] = pd.to_numeric(df["extent_area_m2"], errors="coerce")

    df = clean_targets(df, op_col=op_col, param_col=param_col)

    mask = build_valid_mask(df, op_col=op_col, param_col=param_col)
    df = df.loc[mask].reset_index(drop=True)
    X = np.asarray(X[mask.to_numpy()], dtype=np.float64)

    df["param_norm"] = compute_param_norm(
        df, op_col=op_col, param_col=param_col, distance_ops=distance_ops, area_ops=area_ops
    )

    bad = int(df["param_norm"].isna().sum())
    if bad:
        raise RuntimeError(
            f"param_norm has {bad} NaNs. Usually an operator is missing from DISTANCE_OPS/AREA_OPS."
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
