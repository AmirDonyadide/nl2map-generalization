# src/imgofup/webapp/services/operators/displace.py
from __future__ import annotations

from typing import List, Tuple

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point


def op_displace(
    gdf_metric: gpd.GeoDataFrame,
    distance: float,
    *,
    influence_radius: float | None = None,
    max_shift_factor: float = 1.0,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Displace geometries to reduce overlap / congestion.

    This operator moves each feature by a small vector derived from
    repulsion forces between nearby feature centroids.

    Parameters
    ----------
    gdf_metric:
        Input GeoDataFrame in a *metric CRS* (meters).
    distance:
        Maximum displacement distance (meters).
    influence_radius:
        Radius within which other features exert repulsion (meters).
        Defaults to 3 * distance.
    max_shift_factor:
        Multiplier controlling how strong displacement is (default 1.0).

    Returns
    -------
    (gdf_out, warnings):
        Displaced GeoDataFrame and warnings list.
    """
    warnings: List[str] = []

    if gdf_metric is None or gdf_metric.empty:
        warnings.append("Displace: empty input; returning input unchanged.")
        return gdf_metric, warnings

    try:
        max_dist = float(distance)
    except Exception:
        warnings.append("Displace: invalid distance value; returning input unchanged.")
        return gdf_metric, warnings

    if max_dist <= 0:
        warnings.append("Displace: distance <= 0; returning input unchanged.")
        return gdf_metric, warnings

    if influence_radius is None:
        influence_radius = 3.0 * max_dist

    # Ensure centroids are safe
    try:
        centroids = gdf_metric.geometry.centroid
    except Exception as e:
        warnings.append(f"Displace: centroid computation failed ({e}); returning input unchanged.")
        return gdf_metric, warnings

    coords = np.array([[p.x, p.y] if isinstance(p, Point) else [np.nan, np.nan] for p in centroids])

    if np.isnan(coords).any():
        warnings.append("Displace: some centroids invalid; returning input unchanged.")
        return gdf_metric, warnings

    n = len(coords)
    shifts = np.zeros((n, 2), dtype=float)

    # Compute repulsion vectors
    for i in range(n):
        xi, yi = coords[i]
        vec = np.zeros(2, dtype=float)

        for j in range(n):
            if i == j:
                continue

            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.hypot(dx, dy)

            if dist == 0 or dist > influence_radius:
                continue

            # Repulsion strength decays with distance
            strength = (influence_radius - dist) / influence_radius
            vec += strength * np.array([dx, dy]) / (dist + 1e-9)

        shifts[i] = vec

    # Normalize and clamp shifts
    norms = np.linalg.norm(shifts, axis=1)
    nonzero = norms > 0

    shifts[nonzero] = (
        shifts[nonzero]
        / norms[nonzero][:, None]
        * np.minimum(norms[nonzero], max_dist)
        * max_shift_factor
    )

    # Apply displacement
    gdf_out = gdf_metric.copy()
    new_geoms = []

    for geom, (dx, dy) in zip(gdf_out.geometry, shifts):
        try:
            new_geoms.append(translate(geom, xoff=dx, yoff=dy))
        except Exception:
            new_geoms.append(geom)

    gdf_out["geometry"] = new_geoms

    warnings.append(
        "Displace applied using centroid-based repulsion; topology preserved but exact positions changed."
    )

    return gdf_out, warnings
