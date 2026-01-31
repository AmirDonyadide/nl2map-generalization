# src/train/utils/_map_embeddings_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.mapvec.maps import map_embeddings as me
from src.constants import (
    MAP_ID_WIDTH,
    MAPS_ID_COL,
    MAP_GEOJSON_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
)


def normalize_tile_id(x: Any, width: int = MAP_ID_WIDTH) -> str:
    return str(x).strip().zfill(int(width))


def allowed_tiles_from_excel(
    *,
    user_study_xlsx: Path,
    responses_sheet: str,
    tile_id_col: str,
    complete_col: str,
    remove_col: str,
    only_complete: bool,
    exclude_removed: bool,
    width: int = MAP_ID_WIDTH,
) -> set[str]:
    dfu = pd.read_excel(user_study_xlsx, sheet_name=responses_sheet)

    # boolean columns if present
    if complete_col in dfu.columns:
        dfu[complete_col] = dfu[complete_col].astype(bool)
    if remove_col in dfu.columns:
        dfu[remove_col] = dfu[remove_col].astype(bool)

    mask = pd.Series(True, index=dfu.index)
    if only_complete and complete_col in dfu.columns:
        mask &= dfu[complete_col]
    if exclude_removed and remove_col in dfu.columns:
        mask &= ~dfu[remove_col]

    dfu = dfu.loc[mask].copy()
    if tile_id_col not in dfu.columns:
        raise KeyError(f"Excel sheet missing tile_id_col='{tile_id_col}'")

    tile_raw = dfu[tile_id_col]
    tile_num = pd.to_numeric(tile_raw, errors="coerce")

    if tile_num.notna().all():
        return set(tile_num.astype(int).astype(str).str.zfill(int(width)).tolist())

    return set(tile_raw.astype(str).str.strip().str.zfill(int(width)).tolist())


def safe_count_valid_polygons(path: Path) -> int:
    try:
        # isolate private API usage here
        return int(me._count_valid_polygons(path))
    except Exception:
        return 0


@dataclass(frozen=True)
class EmbedStats:
    n_found: int
    n_allowed: int
    n_after_allowed_filter: int
    n_skipped_dim_mismatch: int
    n_skipped_bad_extent: int
    n_failed_embed: int


def embed_maps_with_extents(
    *,
    pairs: List[Tuple[str, Path]],
    norm: str,
    max_polygons: int,
) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]], List[str], EmbedStats]:
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    feat_names: List[str] = []
    first_dim: int | None = None

    n_skipped_dim_mismatch = 0
    n_skipped_bad_extent = 0
    n_failed_embed = 0

    for map_id_str, path in pairs:
        try:
            vec, names = me.embed_one_map(
                path,
                max_polygons=max_polygons,
                norm=norm,
                norm_wh=None,
            )
        except Exception:
            n_failed_embed += 1
            continue

        if first_dim is None:
            first_dim = int(vec.shape[0])
            feat_names = list(names)
        elif int(vec.shape[0]) != first_dim:
            n_skipped_dim_mismatch += 1
            continue

        extent = me.compute_extent_refs(path)
        diag = float(extent.get(EXTENT_DIAG_COL, np.nan))
        area = float(extent.get(EXTENT_AREA_COL, np.nan))

        if not (np.isfinite(diag) and diag > 0 and np.isfinite(area) and area > 0):
            n_skipped_bad_extent += 1
            continue

        ids.append(map_id_str)
        vecs.append(vec)
        rows.append({MAPS_ID_COL: map_id_str, MAP_GEOJSON_COL: str(path), **extent})

    if not vecs:
        raise RuntimeError("No valid map embeddings produced (all filtered out).")

    E = np.vstack(vecs).astype(np.float32)
    stats = EmbedStats(
        n_found=len(pairs),
        n_allowed=0,
        n_after_allowed_filter=len(pairs),
        n_skipped_dim_mismatch=n_skipped_dim_mismatch,
        n_skipped_bad_extent=n_skipped_bad_extent,
        n_failed_embed=n_failed_embed,
    )
    return ids, E, rows, feat_names, stats
