# src/imgofup/webapp/services/generalize_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_generalization(
    geojson: Dict[str, Any],
    operator: str,
    param_name: Optional[str],
    param_value: Optional[float],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply a generalization operator to an input GeoJSON FeatureCollection.

    Implemented (practical baseline):
      - simplify: shapely.simplify(tolerance, preserve_topology=True)
      - aggregate: buffer-union-buffer trick to merge nearby polygons/lines (distance-based)
      - select: filter features by minimum area threshold (area-based)
      - displace: not implemented (returns unchanged with warning)

    Notes:
      - If input CRS is geographic (typical GeoJSON), we project to EPSG:3857
        so distance/area parameters are interpreted in meters / m².
      - Properties are preserved for simplify/select.
      - Aggregation merges geometries → properties cannot be preserved 1:1; we return
        a single merged feature and warn about attribute loss.
    """
    warnings: List[str] = []
    validate_geojson_featurecollection(geojson)

    op = (operator or "").strip().lower()
    if op in {"unknown", ""}:
        warnings.append("Operator could not be inferred reliably; returning input unchanged.")
        return geojson, warnings

    # param_value is required for these operators (simplify/aggregate/select)
    if param_value is None or not np.isfinite(float(param_value)) or float(param_value) < 0:
        warnings.append("Parameter value missing/invalid; returning input unchanged.")
        return geojson, warnings

    # Load to GeoDataFrame (preserve properties)
    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        warnings.append("Input has no valid geometries; returning input unchanged.")
        return geojson, warnings

    # Project to metric CRS if geographic → interpret params as meters / m²
    gdf_metric, back_to = _ensure_metric(gdf)

    try:
        # -----------------------------
        # SIMPLIFY
        # -----------------------------
        if op in {"simplify", "simplification"}:
            tol = float(param_value)
            if tol == 0:
                return geojson, warnings

            gdf_out = gdf_metric.copy()
            gdf_out["geometry"] = gdf_out.geometry.simplify(tol, preserve_topology=True)
            gdf_out = gdf_out[gdf_out.geometry.notnull() & ~gdf_out.geometry.is_empty].copy()

            out = _gdf_to_geojson(gdf_out, back_to)
            return out, warnings

        # -----------------------------
        # AGGREGATE
        # -----------------------------
        if op in {"aggregate", "aggregation"}:
            dist = float(param_value)
            if dist == 0:
                return geojson, warnings

            merged = _aggregate_geoms(gdf_metric, dist)
            if merged is None or merged.is_empty:
                warnings.append("Aggregation produced empty geometry; returning input unchanged.")
                return geojson, warnings

            warnings.append("Aggregation merges features; output properties are not preserved per-feature.")

            out_gdf = gpd.GeoDataFrame([{"geometry": merged}], geometry="geometry", crs=gdf_metric.crs)
            out = _gdf_to_geojson(out_gdf, back_to)
            return out, warnings

        # -----------------------------
        # SELECT (area threshold)
        # -----------------------------
        if op in {"select", "selection"}:
            # Interpret param_value as minimum area threshold (m² in metric CRS)
            min_area = float(param_value)

            gdf_out = gdf_metric.copy()

            # compute area; if geometry type doesn't support area well, area=0
            try:
                areas = gdf_out.geometry.area
            except Exception:
                areas = np.zeros(len(gdf_out), dtype=float)

            keep = (areas >= min_area) & gdf_out.geometry.notnull() & ~gdf_out.geometry.is_empty
            gdf_out = gdf_out.loc[keep].copy()

            if gdf_out.empty:
                warnings.append("Select removed all features (area threshold too high). Returning empty FeatureCollection.")
                # Return an empty FeatureCollection (valid output)
                return {"type": "FeatureCollection", "features": []}, warnings

            out = _gdf_to_geojson(gdf_out, back_to)
            return out, warnings

        # -----------------------------
        # DISPLACE
        # -----------------------------
        if op in {"displace", "displacement"}:
            warnings.append("Displacement is not implemented yet; returning input unchanged.")
            return geojson, warnings

        warnings.append(f"Unsupported operator '{operator}'; returning input unchanged.")
        return geojson, warnings

    except Exception as e:
        warnings.append(f"Generalization failed ({op}): {e}; returning input unchanged.")
        return geojson, warnings


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def validate_geojson_featurecollection(geojson: Dict[str, Any]) -> None:
    if not isinstance(geojson, dict):
        raise ValueError("geojson must be a dict.")
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson.type must be 'FeatureCollection'.")
    feats = geojson.get("features")
    if not isinstance(feats, list):
        raise ValueError("geojson.features must be a list.")


def operator_supported(operator: str) -> bool:
    op = (operator or "").strip().lower()
    return op in {
        "simplify",
        "simplification",
        "aggregate",
        "aggregation",
        "select",
        "selection",
        "displace",
        "displacement",
    }


def _geojson_to_gdf(fc: Dict[str, Any]) -> gpd.GeoDataFrame:
    feats = fc.get("features", [])
    rows = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        geom = f.get("geometry")
        if geom is None:
            continue
        try:
            shp = shape(geom)
        except Exception:
            continue
        if shp.is_empty:
            continue
        props = f.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        row = dict(props)
        row["geometry"] = shp
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    # Assume WGS84 for GeoJSON dict inputs (matches your embedding pipeline assumption)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    return gdf


def _ensure_metric(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int]:
    back_to = 4326
    try:
        if gdf.crs is not None:
            epsg = gdf.crs.to_epsg()
            if epsg is not None:
                back_to = int(epsg)
    except Exception:
        back_to = 4326

    try:
        if gdf.crs is not None and getattr(gdf.crs, "is_geographic", False):
            return gdf.to_crs(3857), back_to
    except Exception:
        pass

    return gdf, back_to


def _gdf_to_geojson(gdf: gpd.GeoDataFrame, back_to_epsg: int) -> Dict[str, Any]:
    out = gdf
    try:
        if out.crs is not None and out.crs.to_epsg() != back_to_epsg:
            out = out.to_crs(back_to_epsg)
    except Exception:
        pass

    features = []
    geom_col = out.geometry.name if out.geometry is not None else "geometry"
    for _, row in out.iterrows():
        geom = row.get(geom_col)
        if geom is None or geom.is_empty:
            continue
        props = {k: row[k] for k in out.columns if k != geom_col}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})

    return {"type": "FeatureCollection", "features": features}


def _aggregate_geoms(gdf_metric: gpd.GeoDataFrame, distance: float):
    d = float(distance)
    if d <= 0:
        return unary_union(list(gdf_metric.geometry))

    geoms = [g for g in gdf_metric.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None

    grown = [g.buffer(d / 2.0) for g in geoms]
    merged = unary_union(grown)
    shrunk = merged.buffer(-d / 2.0)
    return shrunk
