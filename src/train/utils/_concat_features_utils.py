#src/train/utils/_concat_features_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import json
import numpy as np
import pandas as pd

from src.mapvec.concat import concat_embeddings as ce
from src.types import FeatureMode

_NA_TOKENS = {"", "nan", "none", "null"}

def _to_string_series(s: pd.Series) -> pd.Series:
    # pandas "string" dtype keeps NA nicely and avoids object quirks
    return s.astype("string")


def normalize_id_str(x: Any, *, width: int = 4) -> Optional[str]:
    """
    Normalize a single id (map_id / prompt_id) into a stable string key.

    - trims whitespace
    - treats '', 'nan', 'None', etc. as missing -> returns None
    - numeric-like values (including 1.0) become zero-padded strings, e.g. '0001'
    - non-numeric strings are returned stripped, and if they are digit-only, zfilled
    """
    if x is None:
        return None

    s = str(x).strip()
    if s.lower() in _NA_TOKENS:
        return None

    # numeric-like strings -> int -> zfill
    if s.isdigit():
        return s.zfill(int(width))

    # handle cases like "1.0" from Excel exports
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f)).zfill(int(width))
    except Exception:
        pass

    return s

def normalize_map_id_series(s: pd.Series, *, width: int = 4) -> pd.Series:
    """
    Normalize map_id column to stable string keys (zero-padded width, default 4).
    Handles numeric Excel reads and keeps NA.
    """
    out = _to_string_series(s).str.strip()
    out = out.mask(out.str.lower().isin(_NA_TOKENS), pd.NA)

    # if all numeric-like -> fast path
    num = pd.to_numeric(out, errors="coerce")
    if num.notna().all():
        return num.astype(int).astype(str).str.zfill(int(width)).astype("string")

    # otherwise: zfill only digit-only rows
    m = out.notna() & out.str.fullmatch(r"\d+")
    out.loc[m] = out.loc[m].str.zfill(int(width))
    return out.astype("string")


def normalize_prompt_id_series(s: pd.Series, *, width: int = 4) -> pd.Series:
    """
    Normalize prompt_id column to stable string keys (zero-padded width, default 4).
    Handles numeric Excel reads and keeps NA.
    """
    out = _to_string_series(s).str.strip()
    out = out.mask(out.str.lower().isin(_NA_TOKENS), pd.NA)

    num = pd.to_numeric(out, errors="coerce")
    if num.notna().all():
        return num.astype(int).astype(str).str.zfill(int(width)).astype("string")

    m = out.notna() & out.str.fullmatch(r"\d+")
    out.loc[m] = out.loc[m].str.zfill(int(width))
    return out.astype("string")

# --- mode helpers ---
def uses_prompt(mode: FeatureMode) -> bool:
    return mode in ("prompt_only", "prompt_plus_map", "use_map", "openai_map")


def concat_map_and_prompt(mode: FeatureMode) -> bool:
    return mode in ("prompt_plus_map", "use_map", "openai_map")

# --- loading ---
def load_npz_E_and_ids(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    E, ids = ce.load_npz(npz_path)
    ids = [str(x).strip() for x in ids]
    return np.asarray(E), ids


def build_pairs_from_prompts(prompts_pq: Path, *, prompt_id_width: int) -> pd.DataFrame:
    pairs = pd.read_parquet(prompts_pq)
    need_cols = ["tile_id", "prompt_id", "text"]
    missing = [c for c in need_cols if c not in pairs.columns]
    if missing:
        raise RuntimeError(f"prompts.parquet missing required columns: {missing}")

    pairs = pairs.rename(columns={"tile_id": "map_id"})[["map_id", "prompt_id", "text"]].copy()
    pairs["map_id"] = normalize_map_id_series(pairs["map_id"], width=4)
    pairs["prompt_id"] = normalize_prompt_id_series(pairs["prompt_id"], width=prompt_id_width)

    pairs = pairs.dropna(subset=["map_id", "prompt_id"])
    pairs = pairs[(pairs["map_id"] != "") & (pairs["prompt_id"] != "")]
    pairs = pairs.drop_duplicates(subset=["map_id", "prompt_id"]).reset_index(drop=True)
    return pairs


def merge_extents_from_maps(
    pairs: pd.DataFrame,
    maps_pq: Path,
    *,
    extent_cols_preferred: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], int]:
    maps_df = pd.read_parquet(maps_pq)
    if "map_id" not in maps_df.columns:
        raise RuntimeError("maps.parquet must contain 'map_id'.")

    maps_df = maps_df.copy()
    maps_df["map_id"] = normalize_map_id_series(maps_df["map_id"], width=4)

    required = ["map_id", "extent_diag_m", "extent_area_m2"]
    missing = [c for c in required if c not in maps_df.columns]
    if missing:
        raise RuntimeError(f"maps.parquet is missing required extent columns: {missing}")

    extent_cols = [c for c in extent_cols_preferred if c in maps_df.columns]
    if "map_id" not in extent_cols:
        extent_cols = ["map_id"] + extent_cols

    out = pairs.merge(maps_df[extent_cols], on="map_id", how="left")

    n_missing = int(out["extent_diag_m"].isna().sum())
    if n_missing:
        out = out.dropna(subset=["extent_diag_m", "extent_area_m2"]).reset_index(drop=True)

    saved = [c for c in extent_cols if c != "map_id"]
    return out, saved, n_missing


def match_pairs_to_embedding_indices(
    pairs: pd.DataFrame,
    *,
    feature_mode: FeatureMode,
    map_ids: List[str],
    prm_ids: Optional[List[str]],
    prompt_id_width: int,
) -> Tuple[List[int], List[int], List[int], int]:
    map_ids_n = [str(x).strip().zfill(4) for x in map_ids]
    idx_map = {k: i for i, k in enumerate(map_ids_n)}

    idx_prm: Dict[str, int] = {}
    if uses_prompt(feature_mode):
        assert prm_ids is not None
        prm_ids_n = [str(x).strip() for x in prm_ids]
        prm_ids_n = [pid.zfill(prompt_id_width) if pid.isdigit() else pid for pid in prm_ids_n]
        idx_prm = {k: i for i, k in enumerate(prm_ids_n)}

    chosen_rows: List[int] = []
    im_list: List[int] = []
    ip_list: List[int] = []
    missing_ids = 0

    for i, row in enumerate(pairs.itertuples(index=False), start=0):
        mid = str(row.map_id).strip().zfill(4)
        im = idx_map.get(mid)

        if feature_mode == "map_only":
            if im is None:
                missing_ids += 1
                continue
            chosen_rows.append(i)
            im_list.append(im)
            continue

        # prompt-based
        pid_raw = str(row.prompt_id).strip()
        pid = pid_raw.zfill(prompt_id_width) if pid_raw.isdigit() else pid_raw
        ip = idx_prm.get(pid)

        if im is None or ip is None:
            missing_ids += 1
            continue

        chosen_rows.append(i)
        im_list.append(im)
        ip_list.append(ip)

    return chosen_rows, im_list, ip_list, missing_ids


def build_X(
    *,
    feature_mode: FeatureMode,
    E_map: np.ndarray,
    E_prm: Optional[np.ndarray],
    im_list: List[int],
    ip_list: List[int],
) -> Tuple[np.ndarray, int, int]:
    map_dim = int(E_map.shape[1])

    if feature_mode == "map_only":
        X = E_map[np.asarray(im_list, dtype=int)].astype(np.float32, copy=False)
        return X, map_dim, 0

    assert E_prm is not None
    X_prm = E_prm[np.asarray(ip_list, dtype=int)].astype(np.float32, copy=False)
    prompt_dim = int(E_prm.shape[1])

    if feature_mode == "prompt_only":
        return X_prm, map_dim, prompt_dim

    if concat_map_and_prompt(feature_mode):
        X_map = E_map[np.asarray(im_list, dtype=int)].astype(np.float32, copy=False)
        X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)
        return X, map_dim, prompt_dim

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def write_concat_outputs(
    *,
    out_dir: Path,
    exp_name: str,
    feature_mode: FeatureMode,
    X: np.ndarray,
    join_df: pd.DataFrame,
    map_dim: int,
    prompt_dim: int,
    missing_ids: int,
    extent_cols_saved: List[str],
    sources: Dict[str, str],
    prompt_id_width: int,
    save_pairs_name: Optional[str],
    save_X_name: Optional[str],
    save_meta_name: Optional[str],
) -> Tuple[str, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_name = save_X_name or f"X_{exp_name}.npy"
    pairs_name = save_pairs_name or f"train_pairs_{exp_name}.parquet"
    meta_name = save_meta_name or f"meta_{exp_name}.json"

    X_path = out_dir / X_name
    pairs_path = out_dir / pairs_name
    meta_path = out_dir / meta_name

    np.save(X_path, X)
    join_df.to_parquet(pairs_path, index=False)

    meta = {
        "exp_name": exp_name,
        "feature_mode": str(feature_mode),
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "map_dim": int(map_dim),
        "prompt_dim": int(prompt_dim),
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "skipped_pairs_missing_ids": int(missing_ids),
        "sources": sources,
        "extent_cols_saved": list(extent_cols_saved),
        "prompt_id_width": int(prompt_id_width),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return str(X_path), str(pairs_path), str(meta_path)
