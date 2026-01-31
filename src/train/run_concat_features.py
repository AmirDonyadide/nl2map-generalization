# src/train/run_concat_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.mapvec.concat import concat_embeddings as ce
from src.constants import (
    CONCAT_VERBOSITY_DEFAULT,
    PROMPT_ID_WIDTH_DEFAULT,
    PROMPT_EMBEDDINGS_NPZ_NAME,
    PROMPTS_PARQUET_NAME,
    MAP_EMBEDDINGS_NPZ_NAME,
    MAPS_PARQUET_NAME,
    EXTENT_COLS_PREFERRED,
    FeatureMode,
)

from .utils._concat_features_utils import (
    build_pairs_from_prompts,
    build_X,
    load_npz_E_and_ids,
    match_pairs_to_embedding_indices,
    merge_extents_from_maps,
    uses_prompt,
    write_concat_outputs,
)


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


def run_concat_features_from_dirs(
    *,
    prompt_out_dir: Path,
    map_out_dir: Path,
    out_dir: Path,
    exp_name: str,
    feature_mode: FeatureMode,
    verbosity: int = CONCAT_VERBOSITY_DEFAULT,
    save_pairs_name: Optional[str] = None,
    save_X_name: Optional[str] = None,
    save_meta_name: Optional[str] = None,
    prompt_id_width: int = PROMPT_ID_WIDTH_DEFAULT,
    pairs_parquet: Optional[Path] = None,
) -> ConcatRunMeta:
    prompt_out_dir = Path(prompt_out_dir)
    map_out_dir = Path(map_out_dir)
    out_dir = Path(out_dir)

    ce.setup_logging(verbosity=int(verbosity))

    prm_npz_path = prompt_out_dir / PROMPT_EMBEDDINGS_NPZ_NAME
    prompts_pq = Path(pairs_parquet) if pairs_parquet is not None else (prompt_out_dir / PROMPTS_PARQUET_NAME)

    map_npz_path = map_out_dir / MAP_EMBEDDINGS_NPZ_NAME
    maps_pq = map_out_dir / MAPS_PARQUET_NAME

    required = [prompts_pq, map_npz_path, maps_pq]
    if uses_prompt(feature_mode):
        required.append(prm_npz_path)

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    pairs = build_pairs_from_prompts(prompts_pq, prompt_id_width=int(prompt_id_width))

    pairs, extent_cols_saved, _n_dropped_missing_extent = merge_extents_from_maps(
        pairs,
        maps_pq,
        extent_cols_preferred=EXTENT_COLS_PREFERRED,
    )

    E_map, map_ids = load_npz_E_and_ids(map_npz_path)

    E_prm = None
    prm_ids: List[str] | None = None
    if uses_prompt(feature_mode):
        E_prm, prm_ids = load_npz_E_and_ids(prm_npz_path)

    chosen_rows, im_list, ip_list, missing_ids = match_pairs_to_embedding_indices(
        pairs,
        feature_mode=feature_mode,
        map_ids=map_ids,
        prm_ids=prm_ids,
        prompt_id_width=int(prompt_id_width),
    )
    if not im_list:
        raise RuntimeError("No valid pairs after ID matching.")

    X, map_dim, prompt_dim = build_X(
        feature_mode=feature_mode,
        E_map=E_map,
        E_prm=E_prm,
        im_list=im_list,
        ip_list=ip_list,
    )

    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    if X.shape[0] != len(join_df):
        raise RuntimeError("Row count mismatch between X and joined pairs dataframe.")

    sources = {
        "prompts_parquet": str(prompts_pq),
        "maps_parquet": str(maps_pq),
        "map_npz": str(map_npz_path),
    }
    if uses_prompt(feature_mode):
        sources["prompt_npz"] = str(prm_npz_path)

    X_path, pairs_path, meta_path = write_concat_outputs(
        out_dir=out_dir,
        exp_name=exp_name,
        feature_mode=feature_mode,
        X=X,
        join_df=join_df,
        map_dim=map_dim,
        prompt_dim=prompt_dim,
        missing_ids=missing_ids,
        extent_cols_saved=extent_cols_saved,
        sources=sources,
        prompt_id_width=int(prompt_id_width),
        save_pairs_name=save_pairs_name,
        save_X_name=save_X_name,
        save_meta_name=save_meta_name,
    )

    return ConcatRunMeta(
        exp_name=str(exp_name),
        feature_mode=str(feature_mode),
        X_path=X_path,
        pairs_path=pairs_path,
        meta_path=meta_path,
        rows=int(X.shape[0]),
        cols=int(X.shape[1]),
        map_dim=int(map_dim),
        prompt_dim=int(prompt_dim),
        skipped_pairs_missing_ids=int(missing_ids),
        extent_cols_saved=list(extent_cols_saved),
        sources=dict(sources),
    )
