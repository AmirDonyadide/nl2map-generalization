# src/imgofup/webapp/services/operators/simplify.py
from __future__ import annotations

from typing import List, Tuple

import geopandas as gpd


def op_simplify(
    gdf_metric: gpd.GeoDataFrame,
    tolerance: float,
    *,
    preserve_topology: bool = True,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Simplify geometries using Shapely's Douglas-Peucker implementation via GeoPandas.

    Parameters
    ----------
    gdf_metric:
        Input GeoDataFrame in a *metric* CRS (meters). If CRS is geographic (degrees),
        the tolerance will be interpreted in degrees (not recommended).
    tolerance:
        Simplification tolerance. For metric CRS this is meters.
        - tolerance <= 0 => no-op.
    preserve_topology:
        Whether to preserve topology (recommended for polygons).

    Returns
    -------
    (gdf_out, warnings):
        gdf_out keeps original attributes/properties. Invalid/empty geometries after
        simplification are dropped.
    """
    warnings: List[str] = []

    try:
        tol = float(tolerance)
    except Exception:
        warnings.append(f"Simplify: invalid tolerance '{tolerance}'. Returning input unchanged.")
        return gdf_metric, warnings

    if tol <= 0:
        warnings.append("Simplify: tolerance <= 0; returning input unchanged.")
        return gdf_metric, warnings

    if gdf_metric is None or len(gdf_metric) == 0:
        warnings.append("Simplify: empty input; returning input unchanged.")
        return gdf_metric, warnings

    gdf_out = gdf_metric.copy()

    # Apply simplify (GeoPandas uses Shapely)
    try:
        gdf_out["geometry"] = gdf_out.geometry.simplify(tol, preserve_topology=preserve_topology)
    except Exception as e:
        warnings.append(f"Simplify failed: {e}. Returning input unchanged.")
        return gdf_metric, warnings

    # Drop null/empty geometries (keep old behavior explicitly)
    try:
        gdf_out = gdf_out[gdf_out.geometry.notna() & ~gdf_out.geometry.is_empty].copy()
    except Exception:
        # If geometry column weirdly fails, fall back to returning what we have
        pass

    if gdf_out.empty:
        warnings.append("Simplify removed all geometries (became empty/null). Returning empty output.")
        return gdf_out, warnings

    # Optional: warn if any geometries became invalid (we don't auto-fix here)
    try:
        if hasattr(gdf_out.geometry, "is_valid"):
            invalid_count = int((~gdf_out.geometry.is_valid).sum())
            if invalid_count > 0:
                warnings.append(f"Simplify produced {invalid_count} invalid geometries.")
    except Exception:
        pass

    return gdf_out, warnings
