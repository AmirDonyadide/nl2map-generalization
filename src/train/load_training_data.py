# src/train/load_training_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


FeatureMode = Literal[
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
]


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


def _normalize_map_id(series: pd.Series) -> pd.Series:
    tile_raw = series
    tile_num = pd.to_numeric(tile_raw, errors="coerce")
    if tile_num.notna().all():
        return tile_num.astype(int).astype(str).str.zfill(4)
    return tile_raw.astype(str).str.strip().str.zfill(4)


def _normalize_prompt_id(series: pd.Series, width: int = 4) -> pd.Series:
    """
    Normalize prompt_id to a stable string key.
    Handles Excel numeric reads (1.0), whitespace, missing tokens, and pads with zeros.
    Expected canonical format: 4 digits like '0001'.
    """
    s = series.copy()

    # Try numeric route first (handles 1, 1.0)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        out = num.astype(int).astype(str).str.zfill(width)
        return out

    # Otherwise treat as string
    s = s.astype(str).str.strip()
    s = s.mask(s.str.lower().isin(["", "nan", "none"]), pd.NA)

    # If remaining values are numeric-like strings ("1"), normalize them too
    num2 = pd.to_numeric(s, errors="coerce")
    if num2.notna().all():
        return num2.astype(int).astype(str).str.zfill(width)

    return s.astype("string").str.zfill(width)



def resolve_artifact_paths(
    *,
    exp_name: str,
    feature_mode: FeatureMode,
    paths: Any,
    X_path: Optional[Union[str, Path]] = None,
    pairs_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    if X_path is not None and pairs_path is not None:
        return Path(X_path), Path(pairs_path)

    train_out = getattr(paths, "TRAIN_OUT", None)
    if train_out is not None:
        train_out = Path(train_out)
        cand_X = train_out / f"X_{exp_name}.npy"
        cand_P = train_out / f"train_pairs_{exp_name}.parquet"
        if cand_X.exists() and cand_P.exists():
            return cand_X, cand_P

    base = None
    for attr in ["OUTPUT_DIR", "OUT_DIR", "DATA_OUT", "ROOT_OUT", "BASE_OUT"]:
        if getattr(paths, attr, None) is not None:
            base = Path(getattr(paths, attr))
            break

    if base is None and train_out is not None:
        base = train_out.parent

    if base is None:
        raise AttributeError(
            "Cannot resolve artifacts: paths must have TRAIN_OUT or one of "
            "OUTPUT_DIR/OUT_DIR/DATA_OUT/ROOT_OUT/BASE_OUT."
        )

    mode_to_folder = {
        "prompt_only": ("train_out_prompt_only", "prompt_only"),
        "use_map": ("train_out_use", "use_map"),
        "map_only": ("train_out_map_only", "map_only"),
        "openai_map": ("train_out_openai", "openai_map"),
        "prompt_plus_map": ("train_out", "prompt_plus_map"),
    }

    folder_name, stem = mode_to_folder.get(feature_mode, ("train_out", feature_mode))
    cand_dir = base / folder_name

    cand_X = cand_dir / f"X_{stem}.npy"
    cand_P = cand_dir / f"train_pairs_{stem}.parquet"
    if cand_X.exists() and cand_P.exists():
        return cand_X, cand_P

    cand_X2 = cand_dir / f"X_{exp_name}.npy"
    cand_P2 = cand_dir / f"train_pairs_{exp_name}.parquet"
    if cand_X2.exists() and cand_P2.exists():
        return cand_X2, cand_P2

    patterns = [
        (f"X_{exp_name}.npy", f"train_pairs_{exp_name}.parquet"),
        (f"X_{stem}.npy", f"train_pairs_{stem}.parquet"),
    ]
    for X_name, P_name in patterns:
        X_hits = list(base.glob(f"train_out*/{X_name}"))
        P_hits = list(base.glob(f"train_out*/{P_name}"))
        if X_hits and P_hits:
            return X_hits[0], P_hits[0]

    raise FileNotFoundError(
        "Could not resolve experiment artifacts.\n"
        f"exp_name={exp_name}, feature_mode={feature_mode}\n"
        f"Tried canonical: {train_out}/X_{exp_name}.npy (if TRAIN_OUT exists)\n"
        f"Tried folder: {cand_dir} with X_{stem}.npy and train_pairs_{stem}.parquet\n"
        f"Also searched under: {base}/train_out*/\n"
    )


def load_training_data_with_dynamic_param_norm(
    *,
    exp_name: str,
    feature_mode: FeatureMode,
    paths: Any,
    cfg: Any,
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
    require_text: bool = True,
    X_path: Optional[Union[str, Path]] = None,
    pairs_path: Optional[Union[str, Path]] = None,
) -> LoadedTrainingData:
    Xp, Pp = resolve_artifact_paths(
        exp_name=exp_name,
        feature_mode=feature_mode,
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
    # --- normalize merge keys on pairs_df side (CRITICAL) ---
    if "prompt_id" not in pairs_df.columns:
        raise KeyError(f"{Pp.name} is missing 'prompt_id'. Concat step must keep prompt_id from prompts.parquet.")
    if "map_id" not in pairs_df.columns:
        raise KeyError(f"{Pp.name} is missing 'map_id'. Concat step must keep map_id (tile_id) from maps/prompts.")

    pairs_df["prompt_id"] = _normalize_prompt_id(pairs_df["prompt_id"])
    pairs_df["map_id"] = _normalize_map_id(pairs_df["map_id"])

    if X.shape[0] != len(pairs_df):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but pairs has {len(pairs_df)} rows.")

    n_before = int(len(pairs_df))

    # Load Excel user study labels
    dfu = pd.read_excel(paths.USER_STUDY_XLSX, sheet_name=paths.RESPONSES_SHEET)

    # Apply filtering rules (ONLY_COMPLETE / EXCLUDE_REMOVED)
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

    # -------------------------------
    # NEW: use prompt_id directly from Excel
    # -------------------------------
    PROMPT_ID_COL = getattr(paths, "PROMPT_ID_COL", "prompt_id")
    if PROMPT_ID_COL not in dfu.columns:
        raise KeyError(
            f"Excel sheet is missing required '{PROMPT_ID_COL}' column. "
            "You said you added prompt_id — ensure the column name matches paths.PROMPT_ID_COL "
            "(or rename it to 'prompt_id')."
        )

    dfu["prompt_id"] = _normalize_prompt_id(dfu[PROMPT_ID_COL])
    dfu["map_id"] = _normalize_map_id(dfu[paths.TILE_ID_COL])

    OP_COL = paths.OPERATOR_COL
    PARAM_COL = paths.PARAM_VALUE_COL

    label_cols = ["map_id", "prompt_id", OP_COL, PARAM_COL]
    if getattr(paths, "INTENSITY_COL", None) and (paths.INTENSITY_COL in dfu.columns):
        label_cols.append(paths.INTENSITY_COL)

    labels = dfu[label_cols].copy()

    # Merge labels onto pairs_df
    df = pairs_df.merge(labels, on=["map_id", "prompt_id"], how="left")

    # Optional but very helpful diagnostics if merge fails badly
    hit_rate = float(df[OP_COL].notna().mean()) if OP_COL in df.columns else 0.0
    if hit_rate < 0.5:
        # show overlap counts to guide debugging
        pairs_pids = set(pairs_df["prompt_id"].astype(str).tolist()) if "prompt_id" in pairs_df.columns else set()
        excel_pids = set(labels["prompt_id"].dropna().astype(str).tolist())
        inter = len(pairs_pids & excel_pids)
        raise RuntimeError(
            "Label merge hit-rate is very low "
            f"({hit_rate*100:.1f}% rows received an operator label).\n"
            f"prompt_id overlap: intersection={inter} | pairs_unique={len(pairs_pids)} | excel_unique={len(excel_pids)}.\n"
            "This means pairs_df.prompt_id does not match Excel prompt_id. "
            "Check that prompt_id is preserved through prompt_embeddings/concat and not regenerated."
        )

    if require_text and "text" not in df.columns:
        raise RuntimeError(
            f"{Pp.name} is missing 'text'. "
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

    X = np.asarray(X, dtype=np.float64)

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
