# src/train/run_map_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.mapvec.maps import map_embeddings as me


@dataclass(frozen=True)
class MapEmbeddingRunMeta:
    n_tiles_allowed: int
    n_maps_found: int
    n_maps_used: int
    max_polygons: int
    out_dir: str
    embeddings_path: str
    maps_parquet_path: str


def _allowed_tiles_from_excel(
    *,
    user_study_xlsx: Path,
    responses_sheet: str,
    tile_id_col: str,
    complete_col: str,
    remove_col: str,
    only_complete: bool,
    exclude_removed: bool,
) -> set[str]:
    dfu = pd.read_excel(user_study_xlsx, sheet_name=responses_sheet)

    # booleans
    if complete_col in dfu.columns:
        dfu[complete_col] = dfu[complete_col].astype(bool)
    if remove_col in dfu.columns:
        dfu[remove_col] = dfu[remove_col].astype(bool)

    mask = pd.Series(True, index=dfu.index)
    if only_complete and complete_col in dfu.columns:
        mask &= (dfu[complete_col] == True)
    if exclude_removed and remove_col in dfu.columns:
        mask &= (dfu[remove_col] == False)

    dfu = dfu[mask].copy()

    tile_raw = dfu[tile_id_col]
    tile_num = pd.to_numeric(tile_raw, errors="coerce")

    if tile_num.notna().all():
        allowed = set(tile_num.astype(int).astype(str).str.zfill(4).tolist())
    else:
        allowed = set(tile_raw.astype(str).str.strip().str.zfill(4).tolist())

    return allowed


def run_map_embeddings_from_config(
    *,
    maps_root: Path,
    input_pattern: str,
    user_study_xlsx: Path,
    responses_sheet: str,
    tile_id_col: str,
    complete_col: str,
    remove_col: str,
    only_complete: bool,
    exclude_removed: bool,
    out_dir: Path,
    verbosity: int = 1,
    norm: str = "extent",  # dynamic per-map normalization
) -> MapEmbeddingRunMeta:
    """
    End-to-end map embedding pipeline.

    - Loads allowed tile_ids from Excel user study (with filtering)
    - Finds GeoJSON maps under maps_root matching input_pattern
    - Filters maps to allowed tile_ids
    - First pass: counts valid polygons to set max_polygons
    - Second pass: embeds each map using norm='extent'
    - Computes dynamic per-map extent refs (extent_diag_m, extent_area_m2)
    - Saves outputs using map_embeddings.save_outputs()

    Output files are written into `out_dir` (experiment-scoped).
    """
    maps_root = Path(maps_root)
    user_study_xlsx = Path(user_study_xlsx)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    me.setup_logging(verbosity=verbosity)

    # 1) allowed tiles
    allowed_tile_ids = _allowed_tiles_from_excel(
        user_study_xlsx=user_study_xlsx,
        responses_sheet=responses_sheet,
        tile_id_col=tile_id_col,
        complete_col=complete_col,
        remove_col=remove_col,
        only_complete=only_complete,
        exclude_removed=exclude_removed,
    )

    # 2) discover geojsons and filter
    pairs = list(me.find_geojsons(maps_root, input_pattern))
    n_found = len(pairs)

    pairs = [(map_id, path) for (map_id, path) in pairs if str(map_id).strip().zfill(4) in allowed_tile_ids]
    if not pairs:
        raise RuntimeError("No maps left after Excel filtering.")

    # 3) first pass polygon count
    counts: Dict[str, int] = {}
    for map_id, path in pairs:
        try:
            counts[str(map_id)] = int(me._count_valid_polygons(path))
        except Exception:
            counts[str(map_id)] = 0

    max_polygons = max(max(counts.values()), 1)

    # 4) second pass embedding + extent refs
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    feat_names: List[str] = []
    first_dim: int | None = None

    for map_id, path in pairs:
        map_id_str = str(map_id).strip().zfill(4)

        vec, names = me.embed_one_map(
            path,
            max_polygons=max_polygons,
            norm=norm,      # "extent" for dynamic per-map normalization
            norm_wh=None,
        )

        if first_dim is None:
            first_dim = int(vec.shape[0])
            feat_names = list(names)
        elif int(vec.shape[0]) != first_dim:
            # skip inconsistent feature dims
            continue

        extent = me.compute_extent_refs(path)
        diag = float(extent.get("extent_diag_m", np.nan))
        area = float(extent.get("extent_area_m2", np.nan))

        # skip degenerate extents
        if not (np.isfinite(diag) and diag > 0):
            continue
        if not (np.isfinite(area) and area > 0):
            continue

        ids.append(map_id_str)
        vecs.append(vec)

        rows.append(
            {
                "map_id": map_id_str,
                "geojson": str(path),
                "n_polygons": int(counts.get(str(map_id), 0)),
                **extent,
            }
        )

    if not vecs:
        raise RuntimeError("No valid map embeddings produced (all filtered out).")

    E = np.vstack(vecs).astype(np.float32)

    # 5) save
    me.save_outputs(
        out_dir=out_dir,
        rows=rows,
        E=E,
        ids=ids,
        feat_names=feat_names,
        save_csv=False,
    )

    embeddings_path = out_dir / "maps_embeddings.npz"
    maps_parquet_path = out_dir / "maps.parquet"

    return MapEmbeddingRunMeta(
        n_tiles_allowed=int(len(allowed_tile_ids)),
        n_maps_found=int(n_found),
        n_maps_used=int(len(ids)),
        max_polygons=int(max_polygons),
        out_dir=str(out_dir),
        embeddings_path=str(embeddings_path),
        maps_parquet_path=str(maps_parquet_path),
    )
