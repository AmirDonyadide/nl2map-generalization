# src/imgofup/webapp/services/embedding_map.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Import the existing embedding logic
from imgofup.embeddings.maps import compute_extent_refs, embed_one_map


def _write_geojson_tmp(geojson: Dict[str, Any]) -> Path:
    """
    Write an in-memory GeoJSON dict to a temporary file and return its path.

    Why: your existing map embedding functions work on file paths (GeoJSON on disk).
    """
    if not isinstance(geojson, dict):
        raise ValueError("geojson must be a dict.")
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson.type must be 'FeatureCollection'.")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False, encoding="utf-8")
    tmp_path = Path(tmp.name)
    try:
        json.dump(geojson, tmp)
        tmp.flush()
    finally:
        tmp.close()
    return tmp_path


def embed_geojson_map(
    geojson: Dict[str, Any],
    *,
    norm: str = "extent",
    norm_wh: str | None = None,
    max_polygons: int | float | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute a fixed-length MapVec embedding vector and extent references from an
    uploaded GeoJSON FeatureCollection.

    Returns:
      map_vec: np.ndarray shape (D,) float32
      extent_refs: dict with keys like extent_diag, extent_area, bounds, etc.
                  (exact keys come from your constants used in compute_extent_refs)

    Notes:
      - Uses your existing embedding implementation (embed_one_map + compute_extent_refs).
      - The webapp receives GeoJSON as a dict, so we write it to a temp file first.
      - Temporary file is removed after embedding.
    """
    tmp_path = _write_geojson_tmp(geojson)
    try:
        # extent refs in the same way training computed them
        extent_refs = compute_extent_refs(tmp_path)

        # fixed-length map embedding
        vec, _names = embed_one_map(
            tmp_path,
            max_polygons=max_polygons,
            norm=norm,
            norm_wh=norm_wh,
        )

        # ensure 1D float32 vector
        vec = np.asarray(vec).reshape(-1).astype(np.float32, copy=False)

        if vec.size == 0:
            raise ValueError("Map embedding vector is empty.")

        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return vec, extent_refs
    finally:
        # best-effort cleanup
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
