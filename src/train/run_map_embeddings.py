from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.mapvec.maps import map_embeddings as me
from .utils._map_embeddings_utils import (
    allowed_tiles_from_excel,
    embed_maps_with_extents,
    normalize_tile_id,
    safe_count_valid_polygons,
)


@dataclass(frozen=True)
class MapEmbeddingRunMeta:
    n_tiles_allowed: int
    n_maps_found: int
    n_maps_used: int
    max_polygons: int
    out_dir: str
    embeddings_path: str
    maps_parquet_path: str
    n_skipped_bad_extent: int
    n_skipped_dim_mismatch: int
    n_failed_embed: int


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
    norm: str = "extent",
) -> MapEmbeddingRunMeta:
    maps_root = Path(maps_root)
    user_study_xlsx = Path(user_study_xlsx)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    me.setup_logging(verbosity=verbosity)

    allowed_tile_ids = allowed_tiles_from_excel(
        user_study_xlsx=user_study_xlsx,
        responses_sheet=responses_sheet,
        tile_id_col=tile_id_col,
        complete_col=complete_col,
        remove_col=remove_col,
        only_complete=only_complete,
        exclude_removed=exclude_removed,
    )

    found_pairs = list(me.find_geojsons(maps_root, input_pattern))
    n_found = len(found_pairs)

    pairs = [(normalize_tile_id(map_id), Path(path)) for (map_id, path) in found_pairs]
    pairs = [(mid, p) for (mid, p) in pairs if mid in allowed_tile_ids]
    if not pairs:
        raise RuntimeError("No maps left after Excel filtering.")

    # first pass polygon counts to set max_polygons
    counts: Dict[str, int] = {mid: safe_count_valid_polygons(p) for (mid, p) in pairs}
    max_polygons = max(max(counts.values()), 1)

    # second pass embed
    ids, E, rows, feat_names, stats = embed_maps_with_extents(
        pairs=pairs,
        norm=norm,
        max_polygons=max_polygons,
    )

    # Add polygon counts into rows (optional but useful)
    for r in rows:
        r["n_polygons"] = int(counts.get(r["map_id"], 0))

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
        n_skipped_bad_extent=int(stats.n_skipped_bad_extent),
        n_skipped_dim_mismatch=int(stats.n_skipped_dim_mismatch),
        n_failed_embed=int(stats.n_failed_embed),
    )
