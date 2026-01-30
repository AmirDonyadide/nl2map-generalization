from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _to_zfilled_numeric_str(series: pd.Series, width: int) -> pd.Series:
    """If all values are numeric-like, convert to int->str->zfill."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all():
        return num.astype(int).astype(str).str.zfill(width)
    return pd.Series([pd.NA] * len(series), index=series.index, dtype="string")


def normalize_map_id(series: pd.Series, width: int = 4) -> pd.Series:
    # try numeric
    out = _to_zfilled_numeric_str(series, width)
    if out.notna().all():
        return out

    s = series.astype("string").str.strip()
    # don't destroy missingness
    s = s.mask(s.str.lower().isin(["", "nan", "none"]), pd.NA)
    return s.str.zfill(width)


def normalize_prompt_id(series: pd.Series, width: int = 4) -> pd.Series:
    out = _to_zfilled_numeric_str(series, width)
    if out.notna().all():
        return out

    s = series.astype("string").str.strip()
    s = s.mask(s.str.lower().isin(["", "nan", "none"]), pd.NA)

    # second numeric attempt (strings like "1")
    out2 = _to_zfilled_numeric_str(s, width)
    if out2.notna().all():
        return out2

    return s.str.zfill(width)


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing columns: {missing}")


def resolve_artifact_paths(
    *,
    exp_name: str,
    feature_mode: str,
    paths: Any,
    X_path: Optional[Union[str, Path]] = None,
    pairs_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    # same logic as your current function, but moved here (unchanged is ok)
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
        v = getattr(paths, attr, None)
        if v is not None:
            base = Path(v)
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

    for X_name, P_name in [
        (f"X_{stem}.npy", f"train_pairs_{stem}.parquet"),
        (f"X_{exp_name}.npy", f"train_pairs_{exp_name}.parquet"),
    ]:
        Xp, Pp = cand_dir / X_name, cand_dir / P_name
        if Xp.exists() and Pp.exists():
            return Xp, Pp

    for X_name, P_name in [
        (f"X_{exp_name}.npy", f"train_pairs_{exp_name}.parquet"),
        (f"X_{stem}.npy", f"train_pairs_{stem}.parquet"),
    ]:
        X_hits = list(base.glob(f"train_out*/{X_name}"))
        P_hits = list(base.glob(f"train_out*/{P_name}"))
        if X_hits and P_hits:
            return X_hits[0], P_hits[0]

    raise FileNotFoundError(
        "Could not resolve experiment artifacts.\n"
        f"exp_name={exp_name}, feature_mode={feature_mode}\n"
        f"Also searched under: {base}/train_out*/\n"
    )


def load_and_filter_user_study_labels(paths: Any) -> pd.DataFrame:
    dfu = pd.read_excel(paths.USER_STUDY_XLSX, sheet_name=paths.RESPONSES_SHEET)

    complete_col = getattr(paths, "COMPLETE_COL", None)
    remove_col = getattr(paths, "REMOVE_COL", None)

    mask = pd.Series(True, index=dfu.index)

    if getattr(paths, "ONLY_COMPLETE", False) and complete_col and complete_col in dfu.columns:
        mask &= dfu[complete_col].astype(bool)

    if getattr(paths, "EXCLUDE_REMOVED", False) and remove_col and remove_col in dfu.columns:
        mask &= ~dfu[remove_col].astype(bool)

    return dfu.loc[mask].copy()


def merge_labels_onto_pairs(
    *,
    pairs_df: pd.DataFrame,
    labels: pd.DataFrame,
    op_col: str,
    key_cols: Sequence[str] = ("map_id", "prompt_id"),
    min_hit_rate: float = 0.5,
) -> pd.DataFrame:
    df = pairs_df.merge(labels, on=list(key_cols), how="left")

    hit_rate = float(df[op_col].notna().mean()) if op_col in df.columns else 0.0
    if hit_rate < min_hit_rate:
        pairs_keys = set(map(tuple, pairs_df[list(key_cols)].astype(str).to_numpy()))
        label_keys = set(map(tuple, labels[list(key_cols)].astype(str).to_numpy()))
        inter = len(pairs_keys & label_keys)

        raise RuntimeError(
            "Label merge hit-rate is very low "
            f"({hit_rate*100:.1f}% rows received an operator label).\n"
            f"key intersection count: {inter} | pairs_unique={len(pairs_keys)} | excel_unique={len(label_keys)}.\n"
            "This suggests (map_id, prompt_id) keys do not match between pairs_df and Excel."
        )
    return df


def clean_targets(df: pd.DataFrame, *, op_col: str, param_col: str) -> pd.DataFrame:
    df = df.copy()
    df[op_col] = df[op_col].astype("string").str.strip().str.lower()
    df.loc[df[op_col].isin(["", "nan"]), op_col] = pd.NA
    df[param_col] = pd.to_numeric(df[param_col], errors="coerce")
    return df


def build_valid_mask(df: pd.DataFrame, *, op_col: str, param_col: str) -> pd.Series:
    return (
        df[op_col].notna()
        & df[param_col].notna()
        & df["extent_diag_m"].notna()
        & df["extent_area_m2"].notna()
        & (df["extent_diag_m"] > 0)
        & (df["extent_area_m2"] > 0)
    )


def compute_param_norm(
    df: pd.DataFrame,
    *,
    op_col: str,
    param_col: str,
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
) -> pd.Series:
    dist_set = {str(x).strip().lower() for x in distance_ops}
    area_set = {str(x).strip().lower() for x in area_ops}

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    m_dist = df[op_col].isin(dist_set)
    m_area = df[op_col].isin(area_set)

    out.loc[m_dist] = df.loc[m_dist, param_col] / df.loc[m_dist, "extent_diag_m"]
    out.loc[m_area] = df.loc[m_area, param_col] / df.loc[m_area, "extent_area_m2"]
    return out
