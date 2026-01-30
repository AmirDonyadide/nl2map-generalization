# src/train/run_concat_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import numpy as np
import pandas as pd

from src.mapvec.concat import concat_embeddings as ce


# Updated to reflect actual experiment families (even if this function only uses 2 modes for X construction)
FeatureMode = Literal[
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
]


@dataclass(frozen=True)
class ConcatRunMeta:
    exp_name: str
    feature_mode: str
    X_path: str
    pairs_path: str
    meta_path: str
    rows: int
    cols: int
    map_dim: int
    prompt_dim: int
    skipped_pairs_missing_ids: int
    extent_cols_saved: List[str]
    sources: Dict[str, str]


def _load_npz_E_and_ids(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    E, ids = ce.load_npz(npz_path)
    ids = [str(x).strip() for x in ids]
    return np.asarray(E), ids


def _normalize_prompt_id_series(s: pd.Series, *, width: int = 8) -> pd.Series:
    """
    Ensure prompt_id is a clean string and preserve leading zeros.
    If prompt_id is purely numeric, pad to `width`.
    """
    out = s.astype(str).str.strip()
    out = out.mask(out.isin(["", "nan", "None"]), pd.NA)

    m = out.notna() & out.str.fullmatch(r"\d+")
    out.loc[m] = out.loc[m].str.zfill(int(width))
    return out


def run_concat_features_from_dirs(
    *,
    prompt_out_dir: Path,
    map_out_dir: Path,
    out_dir: Path,
    exp_name: str,
    feature_mode: FeatureMode,
    verbosity: int = 1,
    save_pairs_name: Optional[str] = None,
    save_X_name: Optional[str] = None,
    save_meta_name: Optional[str] = None,
    prompt_id_width: int = 8,  # ✅ NEW: keep consistent with your Excel format
) -> ConcatRunMeta:
    prompt_out_dir = Path(prompt_out_dir)
    map_out_dir = Path(map_out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ce.setup_logging(verbosity=verbosity)

    prm_npz_path = prompt_out_dir / "prompts_embeddings.npz"
    prompts_pq = prompt_out_dir / "prompts.parquet"

    map_npz_path = map_out_dir / "maps_embeddings.npz"
    maps_pq = map_out_dir / "maps.parquet"

    # Always required for all modes: we always need pairs (from prompts.parquet) + maps + extents
    required = [prompts_pq, map_npz_path, maps_pq]

    # Prompt embeddings are required only if prompts are part of X
    if feature_mode in ("prompt_only", "prompt_plus_map", "use_map", "openai_map"):
        required.append(prm_npz_path)

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")


    # ---- build pairs from prompts.parquet (authoritative) ----
    pairs = pd.read_parquet(prompts_pq)
    if "prompt_id" not in pairs.columns or "tile_id" not in pairs.columns:
        raise RuntimeError("prompts.parquet must contain columns: prompt_id, tile_id")

    need_cols = ["tile_id", "prompt_id", "text"]
    missing_cols = [c for c in need_cols if c not in pairs.columns]
    if missing_cols:
        raise RuntimeError(f"prompts.parquet missing required columns: {missing_cols}")

    pairs = pairs.rename(columns={"tile_id": "map_id"})[["map_id", "prompt_id", "text"]].copy()
    pairs["map_id"] = pairs["map_id"].astype(str).str.strip().str.zfill(4)
    pairs["prompt_id"] = _normalize_prompt_id_series(pairs["prompt_id"], width=prompt_id_width)

    pairs = pairs.dropna(subset=["map_id", "prompt_id"])
    pairs = pairs[(pairs["map_id"] != "") & (pairs["prompt_id"] != "")]
    pairs = pairs.drop_duplicates(subset=["map_id", "prompt_id"]).reset_index(drop=True)

    # ---- load map extent refs from maps.parquet and merge into pairs ----
    maps_df = pd.read_parquet(maps_pq)
    maps_df["map_id"] = maps_df["map_id"].astype(str).str.strip().str.zfill(4)

    required_extent_cols = ["map_id", "extent_diag_m", "extent_area_m2"]
    missing = [c for c in required_extent_cols if c not in maps_df.columns]
    if missing:
        raise RuntimeError(f"maps.parquet is missing required extent columns: {missing}")

    extent_cols = [
        "map_id",
        "extent_diag_m",
        "extent_area_m2",
        "extent_width_m",
        "extent_height_m",
        "extent_minx",
        "extent_miny",
        "extent_maxx",
        "extent_maxy",
    ]
    extent_cols = [c for c in extent_cols if c in maps_df.columns]

    pairs = pairs.merge(maps_df[extent_cols], on="map_id", how="left")

    n_missing_extent = int(pairs["extent_diag_m"].isna().sum())
    if n_missing_extent:
        pairs = pairs.dropna(subset=["extent_diag_m", "extent_area_m2"]).reset_index(drop=True)

    # ---- load embeddings ----
    E_map, map_ids = _load_npz_E_and_ids(map_npz_path)

    E_prm = None
    prm_ids: List[str] = []
    if feature_mode in ("prompt_only", "prompt_plus_map", "use_map", "openai_map"):
        E_prm, prm_ids = _load_npz_E_and_ids(prm_npz_path)

    # Normalize ids the same way as pairs (important if ids in NPZ are numeric-looking strings)
    map_ids = [str(x).strip().zfill(4) for x in map_ids]
    if feature_mode in ("prompt_only", "prompt_plus_map", "use_map", "openai_map"):
        prm_ids = [str(x).strip() for x in prm_ids]
        prm_ids = [pid.zfill(prompt_id_width) if pid.isdigit() else pid for pid in prm_ids]

    idx_map = {k: i for i, k in enumerate(map_ids)}
    idx_prm = {k: i for i, k in enumerate(prm_ids)}

    # ---- match & build X ----
    chosen_rows: List[int] = []
    im_list: List[int] = []
    ip_list: List[int] = []
    missing_ids = 0

    idx_map = {k: i for i, k in enumerate(map_ids)}

    if feature_mode == "map_only":
        # map-only: only map_id must exist in map embeddings
        for i, row in enumerate(pairs.itertuples(index=False), start=0):
            mid = str(row.map_id).strip().zfill(4)
            im = idx_map.get(mid)
            if im is None:
                missing_ids += 1
                continue
            chosen_rows.append(i)
            im_list.append(im)
    else:
        # prompt-based: need both map_id and prompt_id
        assert E_prm is not None
        idx_prm = {k: i for i, k in enumerate(prm_ids)}

        for i, row in enumerate(pairs.itertuples(index=False), start=0):
            mid = str(row.map_id).strip().zfill(4)
            pid_raw = str(row.prompt_id).strip()
            pid = pid_raw.zfill(prompt_id_width) if pid_raw.isdigit() else pid_raw

            im = idx_map.get(mid)
            ip = idx_prm.get(pid)

            if im is None or ip is None:
                missing_ids += 1
                continue

            chosen_rows.append(i)
            im_list.append(im)
            ip_list.append(ip)

    if not im_list:
        raise RuntimeError("No valid pairs after ID matching.")


    map_dim = int(E_map.shape[1])

    if feature_mode == "map_only":
        X = E_map[np.asarray(im_list, dtype=int)].astype(np.float32, copy=False)
        prompt_dim = 0

    else:
        assert E_prm is not None
        X_prm = E_prm[np.asarray(ip_list, dtype=int)].astype(np.float32, copy=False)
        prompt_dim = int(E_prm.shape[1])

        if feature_mode == "prompt_only":
            X = X_prm
        elif feature_mode in ("prompt_plus_map", "use_map", "openai_map"):
            X_map = E_map[np.asarray(im_list, dtype=int)].astype(np.float32, copy=False)
            X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    assert X.shape[0] == len(join_df), "Row count mismatch between X and join_df."

    # ---- save outputs ----
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
        "feature_mode": feature_mode,
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "map_dim": map_dim,
        "prompt_dim": prompt_dim,
        "rows": int(X.shape[0]),
        "skipped_pairs_missing_ids": int(missing_ids),
        "sources": {
            "prompts_parquet": str(prompts_pq),
            "maps_parquet": str(maps_pq),
            "map_npz": str(map_npz_path),
            "prompt_npz": str(prm_npz_path),
        },
        "extent_cols_saved": [c for c in extent_cols if c != "map_id"],
        "prompt_id_width": int(prompt_id_width),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return ConcatRunMeta(
        exp_name=str(exp_name),
        feature_mode=str(feature_mode),
        X_path=str(X_path),
        pairs_path=str(pairs_path),
        meta_path=str(meta_path),
        rows=int(X.shape[0]),
        cols=int(X.shape[1]),
        map_dim=map_dim,
        prompt_dim=prompt_dim,
        skipped_pairs_missing_ids=int(missing_ids),
        extent_cols_saved=[c for c in extent_cols if c != "map_id"],
        sources=meta["sources"],
    )
