# src/imgofup/webapp/services/geom_clean.py
from __future__ import annotations

from typing import List

import geopandas as gpd


def clean_geometries(gdf: gpd.GeoDataFrame, warnings: List[str]) -> gpd.GeoDataFrame:
    """
    Clean/repair geometries in a GeoDataFrame (best-effort).

    Steps:
      1) Drop missing/empty geometries
      2) Try Shapely 2.x `make_valid` (if available)
      3) Fallback to `buffer(0)` trick (often fixes self-intersections for polygons)
      4) Drop remaining empty/invalid geometries

    Parameters
    ----------
    gdf:
        Input GeoDataFrame.
    warnings:
        A list that we append human-readable warnings to.

    Returns
    -------
    gpd.GeoDataFrame:
        Cleaned GeoDataFrame (may be empty).
    """
    out = gdf.copy()

    # Drop missing/empty
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        warnings.append("Geometry cleaning: input contained no non-empty geometries.")
        return out

    fixed_any = False

    # Try Shapely 2.x make_valid
    try:
        from shapely.make_valid import make_valid  # type: ignore

        out["geometry"] = out.geometry.apply(make_valid)
        fixed_any = True
        warnings.append("Geometry cleaning: applied shapely.make_valid.")
    except Exception:
        pass

    # Fallback: buffer(0)
    if not fixed_any:
        try:
            out["geometry"] = out.geometry.buffer(0)
            fixed_any = True
            warnings.append("Geometry cleaning: applied buffer(0) fallback.")
        except Exception:
            pass

    # Drop remaining null/empty
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        warnings.append("Geometry cleaning: all geometries became empty after repair.")
        return out

    # Drop invalid if possible
    try:
        if hasattr(out.geometry, "is_valid"):
            invalid_mask = ~out.geometry.is_valid
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                warnings.append(f"Geometry cleaning: dropping {invalid_count} invalid geometries.")
                out = out.loc[~invalid_mask].copy()
    except Exception:
        pass

    if out.empty:
        warnings.append("Geometry cleaning: all geometries were invalid after repair.")
        return out

    return out
