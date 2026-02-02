# src/imgofup/webapp/services/operators/select.py
from __future__ import annotations

from typing import List, Tuple

import geopandas as gpd
import numpy as np


def op_select(
    gdf_metric: gpd.GeoDataFrame,
    min_area: float,
    *,
    polygons_only: bool = True,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Select (filter) features by a minimum area threshold.

    Parameters
    ----------
    gdf_metric:
        Input GeoDataFrame in a *metric* CRS (meters), so geometry.area is in m².
        If CRS is geographic (degrees), area values will be meaningless.
    min_area:
        Minimum area threshold (m² for metric CRS). min_area < 0 is treated as invalid.
    polygons_only:
        If True, we only consider Polygon/MultiPolygon geometries.
        If False, we compute area for all geometries; non-area types usually yield 0.

    Returns
    -------
    (gdf_out, warnings):
        A filtered GeoDataFrame preserving properties.
        If everything is filtered out, returns an empty GeoDataFrame (valid output).
    """
    warnings: List[str] = []

    try:
        thr = float(min_area)
    except Exception:
        warnings.append(f"Select: invalid min_area '{min_area}'. Returning input unchanged.")
        return gdf_metric, warnings

    if thr < 0:
        warnings.append("Select: min_area < 0 is invalid. Returning input unchanged.")
        return gdf_metric, warnings

    if gdf_metric is None or len(gdf_metric) == 0:
        warnings.append("Select: empty input; returning input unchanged.")
        return gdf_metric, warnings

    gdf_out = gdf_metric.copy()

    # Optionally restrict to polygons (recommended)
    if polygons_only:
        try:
            geom_types = gdf_out.geometry.geom_type
            poly_mask = geom_types.isin(["Polygon", "MultiPolygon"])
            if not bool(poly_mask.any()):
                warnings.append("Select: polygons_only=True but there are no polygon geometries. Returning input unchanged.")
                return gdf_metric, warnings
            gdf_out = gdf_out.loc[poly_mask].copy()
        except Exception as e:
            warnings.append(f"Select: failed to check geometry types ({e}); continuing without filtering by type.")

    # Compute area robustly
    try:
        areas = np.asarray(gdf_out.geometry.area, dtype=float)
    except Exception as e:
        warnings.append(f"Select: area computation failed ({e}); treating all areas as 0.")
        areas = np.zeros(len(gdf_out), dtype=float)

    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)

    keep = (areas >= thr) & gdf_out.geometry.notna() & ~gdf_out.geometry.is_empty
    gdf_out = gdf_out.loc[keep].copy()

    if gdf_out.empty:
        warnings.append("Select removed all features (threshold too high or areas too small).")
        # return an empty GeoDataFrame (valid)
        empty = gpd.GeoDataFrame([], geometry="geometry", crs=gdf_metric.crs)
        return empty, warnings

    return gdf_out, warnings
