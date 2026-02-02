# src/imgofup/webapp/services/operators/aggregate.py
from __future__ import annotations

from typing import List, Tuple, Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def op_aggregate(
    gdf_metric: gpd.GeoDataFrame,
    distance: float,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Aggregate (merge) nearby geometries using a buffer-union-buffer trick.

    This is a practical baseline for "aggregation":
      1) buffer each geometry by distance/2
      2) unary_union all buffered geometries
      3) buffer back by -distance/2

    Parameters
    ----------
    gdf_metric:
        Input GeoDataFrame in a *metric* CRS (meters). If CRS is geographic (degrees),
        the distance will be interpreted in degrees (not recommended).
    distance:
        Aggregation distance in meters (metric CRS). distance <= 0 => union only (no buffering).

    Returns
    -------
    (gdf_out, warnings):
        Returns a GeoDataFrame with a single merged geometry (properties cannot be preserved per feature).
    """
    warnings: List[str] = []

    try:
        d = float(distance)
    except Exception:
        warnings.append(f"Aggregate: invalid distance '{distance}'. Returning input unchanged.")
        return gdf_metric, warnings

    if gdf_metric is None or len(gdf_metric) == 0:
        warnings.append("Aggregate: empty input; returning input unchanged.")
        return gdf_metric, warnings

    # Filter valid geometries
    geoms = [g for g in gdf_metric.geometry if g is not None and not g.is_empty]
    if not geoms:
        warnings.append("Aggregate: no valid geometries found; returning input unchanged.")
        return gdf_metric, warnings

    merged: Optional[BaseGeometry] = None

    try:
        if d <= 0:
            # No-distance union baseline
            merged = unary_union(geoms)
        else:
            grown = [g.buffer(d / 2.0) for g in geoms]
            merged_buf = unary_union(grown)
            merged = merged_buf.buffer(-d / 2.0)
    except Exception as e:
        warnings.append(f"Aggregate failed: {e}. Returning input unchanged.")
        return gdf_metric, warnings

    if merged is None or merged.is_empty:
        warnings.append("Aggregate produced empty geometry; returning empty output.")
        out_gdf = gpd.GeoDataFrame([], geometry="geometry", crs=gdf_metric.crs)
        return out_gdf, warnings

    warnings.append("Aggregation merges features; output properties are not preserved per-feature.")

    out_gdf = gpd.GeoDataFrame([{"geometry": merged}], geometry="geometry", crs=gdf_metric.crs)
    return out_gdf, warnings
