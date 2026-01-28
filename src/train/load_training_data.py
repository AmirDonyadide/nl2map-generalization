# src/train/load_training_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, Tuple

import numpy as np
import pandas as pd


FeatureMode = Literal["prompt_only", "prompt_plus_map"]


@dataclass(frozen=True)
class LoadedTrainingData:
    X: np.ndarray               # (N, D), float64
    df: pd.DataFrame            # aligned rows (N)
    exp_name: str
    feature_mode: str
    X_path: str
    pairs_path: str
    n_before_filter: int
    n_after_filter: int


def _make_prompt_ids_for_excel_rows(
    df_excel: pd.DataFrame,
    *,
    prefix: str,
    width: int,
) -> pd.Series:
    """
    Recreate the exact prompt_id scheme used by prompt embedding step.
    Assumes prompt_embeddings created prompt_id from the row index after reset.
    """
    dfu = df_excel.reset_index(drop=False).rename(columns={"index": "_row"})
    return dfu["_row"].apply(lambda r: f"{prefix}{int(r):0{width}d}")


def _normalize_map_id(series: pd.Series) -> pd.Series:
    tile_raw = series
    tile_num = pd.to_numeric(tile_raw, errors="coerce")
    if tile_num.notna().all():
        return tile_num.astype(int).astype(str).str.zfill(4)
    return tile_raw.astype(str).str.strip().str.zfill(4)


def load_training_data_with_dynamic_param_norm(
    *,
    exp_name: str,
    feature_mode: FeatureMode,
    paths: Any,
    cfg: Any,
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
    require_text: bool = True,
) -> LoadedTrainingData:
    """
    Load experiment-specific X + pairs parquet, merge labels from Excel,
    clean, and compute dynamic extent-normalized regression target param_norm.

    Expected inputs produced by your main pipeline:
      - {PATHS.TRAIN_OUT}/X_{exp_name}.npy
      - {PATHS.TRAIN_OUT}/train_pairs_{exp_name}.parquet

    The pairs parquet must contain:
      - map_id, prompt_id, text
      - extent_diag_m, extent_area_m2  (merged from maps.parquet earlier)

    Excel (user study) must contain:
      - operator col, param_value col, intensity col (optional)
      - tile_id col, complete/remove flags (optional depending on filters)
    """
    train_out = Path(paths.TRAIN_OUT)

    X_path = train_out / f"X_{exp_name}.npy"
    pairs_path = train_out / f"train_pairs_{exp_name}.parquet"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing features: {X_path} (run concat step for this experiment)")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs: {pairs_path} (run concat step for this experiment)")

    X = np.load(X_path)
    pairs_df = pd.read_parquet(pairs_path)
    if X.shape[0] != len(pairs_df):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but pairs has {len(pairs_df)} rows.")

    n_before = int(len(pairs_df))

    # Load Excel user study labels
    dfu = pd.read_excel(paths.USER_STUDY_XLSX, sheet_name=paths.RESPONSES_SHEET)

    # Apply same filtering rules as embedding steps (ONLY_COMPLETE / EXCLUDE_REMOVED)
    if getattr(paths, "COMPLETE_COL", None) in dfu.columns:
        dfu[paths.COMPLETE_COL] = dfu[paths.COMPLETE_COL].astype(bool)
    if getattr(paths, "REMOVE_COL", None) in dfu.columns:
        dfu[paths.REMOVE_COL] = dfu[paths.REMOVE_COL].astype(bool)

    mask_excel = pd.Series(True, index=dfu.index)
    if getattr(paths, "ONLY_COMPLETE", False) and (paths.COMPLETE_COL in dfu.columns):
        mask_excel &= (dfu[paths.COMPLETE_COL] == True)
    if getattr(paths, "EXCLUDE_REMOVED", False) and (paths.REMOVE_COL in dfu.columns):
        mask_excel &= (dfu[paths.REMOVE_COL] == False)

    dfu = dfu[mask_excel].copy()

    # Recreate prompt_id and map_id exactly like embedding pipeline
    dfu = dfu.reset_index(drop=False).rename(columns={"index": "_row"})
    prefix = paths.PROMPT_ID_PREFIX
    width = int(paths.PROMPT_ID_WIDTH)
    dfu["prompt_id"] = dfu["_row"].apply(lambda r: f"{prefix}{int(r):0{width}d}")
    dfu["map_id"] = _normalize_map_id(dfu[paths.TILE_ID_COL])

    # Select label columns
    OP_COL = paths.OPERATOR_COL
    PARAM_COL = paths.PARAM_VALUE_COL

    label_cols = ["map_id", "prompt_id", OP_COL, PARAM_COL]
    if getattr(paths, "INTENSITY_COL", None) and (paths.INTENSITY_COL in dfu.columns):
        label_cols.append(paths.INTENSITY_COL)

    labels = dfu[label_cols].copy()

    # Merge labels onto pairs_df
    df = pairs_df.merge(labels, on=["map_id", "prompt_id"], how="left")

    if require_text and "text" not in df.columns:
        raise RuntimeError(
            f"{pairs_path.name} is missing 'text'. "
            "Your concat step must keep 'text' from prompts.parquet."
        )
    if "text" in df.columns:
        df["text"] = df["text"].astype("string")

    # Clean targets
    df[OP_COL] = df[OP_COL].astype("string").str.strip().str.lower()
    df.loc[df[OP_COL].isin(["", "nan"]), OP_COL] = pd.NA
    df[PARAM_COL] = pd.to_numeric(df[PARAM_COL], errors="coerce")

    # Ensure dynamic extents exist
    REQ_EXT = ["extent_diag_m", "extent_area_m2"]
    missing = [c for c in REQ_EXT if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing {missing} in merged dataframe. "
            "Concat step must merge extent_* from maps.parquet."
        )

    df["extent_diag_m"] = pd.to_numeric(df["extent_diag_m"], errors="coerce")
    df["extent_area_m2"] = pd.to_numeric(df["extent_area_m2"], errors="coerce")

    # Keep only valid rows
    mask = (
        df[OP_COL].notna() &
        df[PARAM_COL].notna() &
        df["extent_diag_m"].notna() &
        df["extent_area_m2"].notna() &
        (df["extent_diag_m"] > 0) &
        (df["extent_area_m2"] > 0)
    )

    df = df.loc[mask].reset_index(drop=True)
    X = X[mask.values]

    # Compute param_norm using dynamic extents
    dist_set = set([str(x).strip().lower() for x in distance_ops])
    area_set = set([str(x).strip().lower() for x in area_ops])

    df["param_norm"] = np.nan

    m_dist = df[OP_COL].isin(dist_set)
    m_area = df[OP_COL].isin(area_set)

    df.loc[m_dist, "param_norm"] = df.loc[m_dist, PARAM_COL] / df.loc[m_dist, "extent_diag_m"]
    df.loc[m_area, "param_norm"] = df.loc[m_area, PARAM_COL] / df.loc[m_area, "extent_area_m2"]

    bad = int(df["param_norm"].isna().sum())
    if bad != 0:
        raise RuntimeError(
            f"param_norm has {bad} NaNs. "
            "This usually means an operator is missing from DISTANCE_OPS/AREA_OPS."
        )

    # Cast X to float64 for sklearn stability
    X = np.asarray(X, dtype=np.float64)

    return LoadedTrainingData(
        X=X,
        df=df,
        exp_name=exp_name,
        feature_mode=feature_mode,
        X_path=str(X_path),
        pairs_path=str(pairs_path),
        n_before_filter=n_before,
        n_after_filter=int(len(df)),
    )
