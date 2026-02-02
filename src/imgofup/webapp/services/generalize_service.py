# src/imgofup/webapp/services/generalize_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from imgofup.webapp.services.geoio import geojson_to_gdf, gdf_to_geojson, ensure_metric
from imgofup.webapp.services.geom_clean import clean_geometries
from imgofup.webapp.services.operators.simplify import op_simplify
from imgofup.webapp.services.operators.aggregate import op_aggregate
from imgofup.webapp.services.operators.select import op_select
from imgofup.webapp.services.operators.displace import op_displace


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

    Operators (strict set):
      - simplify  (distance)
      - aggregate (distance)
      - select    (area)
      - displace  (distance)

    CRS policy:
      - If input GeoJSON has a 'crs' member (common in GIS exports), we respect it.
      - If CRS is geographic (lon/lat), we project to EPSG:3857 for metric ops.
      - Output is always returned in the SAME CRS as the input:
        - we reproject back to original EPSG (best-effort)
        - we copy the exact input 'crs' JSON object into the output if provided.

    Parameter policy:
      - simplify/aggregate/displace require non-negative finite distance
      - select requires non-negative finite area threshold

    Robustness:
      - We clean invalid geometries (make_valid or buffer(0)).
      - If cleaning removes everything, we return input unchanged with warnings.
    """
    warnings: List[str] = []
    validate_geojson_featurecollection(geojson)

    input_crs_obj = geojson.get("crs")  # preserve exact CRS JSON if present

    op = (operator or "").strip().lower()
    if op in {"unknown", ""}:
        warnings.append("Operator could not be inferred reliably; returning input unchanged.")
        return geojson, warnings
    if not operator_supported(op):
        warnings.append(f"Unsupported operator '{operator}'; returning input unchanged.")
        return geojson, warnings

    # Parameter validation
    needs_param = op in {"simplify", "aggregate", "select", "displace"}
    pv: Optional[float] = None
    if needs_param:
        if param_value is None or not np.isfinite(float(param_value)) or float(param_value) < 0:
            warnings.append("Parameter value missing/invalid; returning input unchanged.")
            return geojson, warnings
        pv = float(param_value)

    # Optional param kind consistency check (soft warning only)
    param_kind = (param_name or "").strip().lower()
    expected_kind = "area" if op == "select" else "distance"
    if param_kind and param_kind != expected_kind:
        warnings.append(
            f"Parameter type mismatch: operator '{op}' expects '{expected_kind}' but got '{param_kind}'. Proceeding."
        )

    # Read input
    gdf = geojson_to_gdf(geojson)
    if gdf.empty:
        warnings.append("Input has no valid geometries; returning input unchanged.")
        return geojson, warnings

    # Ensure metric when needed
    gdf_metric, back_to_epsg = ensure_metric(gdf)

    # Clean invalid geometries
    gdf_metric = clean_geometries(gdf_metric, warnings)
    if gdf_metric.empty:
        warnings.append("All geometries became invalid/empty after cleaning; returning input unchanged.")
        return geojson, warnings

    # Dispatch to operator
    try:
        if op == "simplify":
            assert pv is not None
            gdf_out, w = op_simplify(gdf_metric, pv)
            warnings.extend(w)
            return gdf_to_geojson(gdf_out, back_to_epsg, input_crs_obj=input_crs_obj), warnings

        if op == "aggregate":
            assert pv is not None
            gdf_out, w = op_aggregate(gdf_metric, pv)
            warnings.extend(w)
            return gdf_to_geojson(gdf_out, back_to_epsg, input_crs_obj=input_crs_obj), warnings

        if op == "select":
            assert pv is not None
            gdf_out, w = op_select(gdf_metric, pv, polygons_only=True)
            warnings.extend(w)

            # If everything removed, still return a valid empty FC with preserved CRS.
            if gdf_out.empty:
                empty_fc: Dict[str, Any] = {"type": "FeatureCollection", "features": []}
                if isinstance(input_crs_obj, dict):
                    empty_fc["crs"] = input_crs_obj
                else:
                    empty_fc = gdf_to_geojson(gdf_out, back_to_epsg, input_crs_obj=input_crs_obj)
                return empty_fc, warnings

            return gdf_to_geojson(gdf_out, back_to_epsg, input_crs_obj=input_crs_obj), warnings

        if op == "displace":
            assert pv is not None
            gdf_out, w = op_displace(gdf_metric, pv)
            warnings.extend(w)
            return gdf_to_geojson(gdf_out, back_to_epsg, input_crs_obj=input_crs_obj), warnings

        # Should not happen due to operator_supported()
        warnings.append(f"Unsupported operator '{operator}'; returning input unchanged.")
        return geojson, warnings

    except Exception as e:
        warnings.append(f"Generalization failed ({op}): {e}; returning input unchanged.")
        return geojson, warnings


# -----------------------------------------------------------------------------
# Validation
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
    return op in {"simplify", "aggregate", "select", "displace"}
