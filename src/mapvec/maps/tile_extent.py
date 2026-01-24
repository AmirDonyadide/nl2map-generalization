# src/mapvec/maps/tile_extent.py
"""
Utility to compute per-map extent reference scales from a GeoJSON (or any vector file
readable by GeoPandas).

This is used for Solution-1 normalization:
- distance ops:  param_norm = param_value / extent_diag_m
- area ops:      param_norm = param_value / extent_area_m2

Then during inference:
- distance ops:  param_value = pred_param_norm * extent_diag_m
- area ops:      param_value = pred_param_norm * extent_area_m2

Notes:
- If CRS is missing, we assume EPSG:4326.
- If CRS is geographic (lat/lon), we project to EPSG:3857 (meters) before computing bounds.
- Extents are computed from gdf.total_bounds (minx, miny, maxx, maxy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import geopandas as gpd


def _read_geo(path: Union[str, Path]) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Geo file not found: {path}")

    gdf = gpd.read_file(path)

    # If CRS missing, assume WGS84 (common for GeoJSON)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if gdf.empty or "geometry" not in gdf.columns:
        raise ValueError(f"Invalid/empty GeoDataFrame for: {path}")

    gdf = gdf[gdf.geometry.notnull()].copy()
    # robust empty-geometry filter
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        gdf = gdf[[not geom.is_empty for geom in gdf.geometry]].copy()

    if gdf.empty:
        raise ValueError(f"All geometries empty/invalid for: {path}")

    return gdf


def compute_extent_refs(
    geo_path: Union[str, Path],
    *,
    project_if_geographic: bool = True,
    metric_epsg: int = 3857,
    include_bounds: bool = True,
    include_crs: bool = True,
    allow_nan: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute extent metrics in meters / m².

    Returns dict with keys:
      extent_width_m, extent_height_m, extent_diag_m, extent_area_m2
    and optionally:
      extent_minx, extent_miny, extent_maxx, extent_maxy, extent_crs

    If extents are degenerate (w<=0 or h<=0), diag/area will be NaN (if allow_nan=True),
    otherwise an exception is raised.

    Parameters
    ----------
    geo_path : str | Path
        Path to GeoJSON (or other vector).
    project_if_geographic : bool
        If CRS is geographic (lat/lon), project to EPSG:metric_epsg before measuring.
    metric_epsg : int
        Target EPSG for metric projection (default 3857).
    include_bounds : bool
        Include raw bounds (minx/miny/maxx/maxy) in output.
    include_crs : bool
        Include extent_crs string in output.
    allow_nan : bool
        If False, raise on degenerate/missing bounds.
    """
    gdf = _read_geo(geo_path)

    # Project to meters when needed
    if project_if_geographic:
        try:
            if gdf.crs is not None and getattr(gdf.crs, "is_geographic", False):
                gdf = gdf.to_crs(metric_epsg)
        except Exception:
            # If projection fails, proceed with original CRS (but units might be wrong)
            pass

    minx, miny, maxx, maxy = map(float, gdf.total_bounds)

    w = maxx - minx
    h = maxy - miny

    # Validate
    finite_bounds = all(np.isfinite([minx, miny, maxx, maxy]))
    finite_wh = np.isfinite(w) and np.isfinite(h)
    if (not finite_bounds) or (not finite_wh):
        if allow_nan:
            out = {
                "extent_width_m": np.nan,
                "extent_height_m": np.nan,
                "extent_diag_m": np.nan,
                "extent_area_m2": np.nan,
            }
            if include_bounds:
                out.update(
                    {
                        "extent_minx": np.nan,
                        "extent_miny": np.nan,
                        "extent_maxx": np.nan,
                        "extent_maxy": np.nan,
                    }
                )
            return out
        raise ValueError(f"Non-finite bounds for: {geo_path}")

    if w <= 0 or h <= 0:
        if allow_nan:
            diag = np.nan
            area = np.nan
        else:
            raise ValueError(f"Degenerate extent (w={w}, h={h}) for: {geo_path}")
    else:
        diag = float(np.sqrt(w * w + h * h))
        area = float(w * h)

    out: Dict[str, Optional[float]] = {
        "extent_width_m": float(w),
        "extent_height_m": float(h),
        "extent_diag_m": float(diag) if np.isfinite(diag) else np.nan,
        "extent_area_m2": float(area) if np.isfinite(area) else np.nan,
    }

    if include_bounds:
        out.update(
            {
                "extent_minx": float(minx),
                "extent_miny": float(miny),
                "extent_maxx": float(maxx),
                "extent_maxy": float(maxy),
            }
        )

    return out


def safe_extent_scale(
    refs: Dict[str, Optional[float]],
    *,
    kind: str,
    default_diag_m: float = 565.685424949238,   # sqrt(400^2 + 400^2)
    default_area_m2: float = 160000.0,          # 400*400
) -> float:
    """
    Convenience: pick a scale (diag for distance, area for select) with fallback.
    kind: "distance" or "area"
    """
    kind = kind.lower().strip()
    if kind == "distance":
        v = refs.get("extent_diag_m")
        v = float(v) if v is not None else np.nan
        return v if np.isfinite(v) and v > 0 else float(default_diag_m)
    if kind == "area":
        v = refs.get("extent_area_m2")
        v = float(v) if v is not None else np.nan
        return v if np.isfinite(v) and v > 0 else float(default_area_m2)
    raise ValueError("kind must be 'distance' or 'area'")
