# src/imgofup/webapp/services/geoio.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
from shapely.geometry import shape, mapping


# -----------------------------------------------------------------------------
# CRS helpers
# -----------------------------------------------------------------------------
def extract_epsg_from_geojson(fc: Dict[str, Any]) -> Optional[int]:
    """
    Extract EPSG code from a GeoJSON 'crs' member like:
      {"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::25832"}}
    or:
      {"type":"name","properties":{"name":"EPSG:25832"}}

    Returns None if not found.
    """
    crs = fc.get("crs")
    if not isinstance(crs, dict):
        return None
    props = crs.get("properties")
    if not isinstance(props, dict):
        return None
    name = props.get("name")
    if not isinstance(name, str):
        return None

    import re

    m = re.search(r"EPSG(?:::+|:)(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def ensure_metric(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int]:
    """
    Ensure a GeoDataFrame is in a metric CRS.

    Returns (gdf_metric, back_to_epsg):
      - back_to_epsg is the original EPSG of gdf if known, otherwise 4326.
      - if gdf is geographic (lon/lat), reproject to EPSG:3857 for meter/m² operations.
      - if gdf is already projected/metric, return unchanged.

    Notes:
      - For best results, callers should ensure gdf.crs is set correctly.
    """
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


# -----------------------------------------------------------------------------
# GeoJSON <-> GeoDataFrame
# -----------------------------------------------------------------------------
def geojson_to_gdf(fc: Dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Parse a GeoJSON FeatureCollection dict into a GeoDataFrame.

    - Preserves properties into columns.
    - Respects fc["crs"] if present (non-RFC7946 but common in GIS tools).
    - Falls back to EPSG:4326 if no CRS is provided.
    - Drops invalid/unparseable/empty geometries.

    Returns an empty GeoDataFrame if nothing valid is found.
    """
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

    # Set CRS based on input
    if gdf.crs is None:
        epsg = extract_epsg_from_geojson(fc)
        if epsg is not None:
            gdf = gdf.set_crs(epsg)
        else:
            gdf = gdf.set_crs(4326)

    return gdf


def gdf_to_geojson(
    gdf: gpd.GeoDataFrame,
    back_to_epsg: int,
    *,
    input_crs_obj: Any = None,
) -> Dict[str, Any]:
    """
    Convert a GeoDataFrame into a GeoJSON FeatureCollection dict.

    CRS behavior:
      1) Reproject to back_to_epsg (best effort)
      2) Write CRS into GeoJSON:
         - if input_crs_obj is a dict => copy EXACTLY
         - else write EPSG URN "urn:ogc:def:crs:EPSG::<epsg>"

    Notes:
      - This keeps your "input CRS == output CRS" requirement.
      - GeoJSON CRS is non-standard in RFC 7946, but many GIS tools support it.
    """
    out = gdf

    # Reproject back
    try:
        if out.crs is not None:
            current = out.crs.to_epsg()
            if current is None or int(current) != int(back_to_epsg):
                out = out.to_crs(back_to_epsg)
    except Exception:
        # Keep best-effort output even if reprojection fails
        pass

    features = []
    geom_col = out.geometry.name if out.geometry is not None else "geometry"

    for _, row in out.iterrows():
        geom = row.get(geom_col)
        if geom is None or geom.is_empty:
            continue
        props = {k: row[k] for k in out.columns if k != geom_col}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})

    fc: Dict[str, Any] = {"type": "FeatureCollection", "features": features}

    # Preserve CRS like input
    if isinstance(input_crs_obj, dict):
        fc["crs"] = input_crs_obj
    elif back_to_epsg is not None:
        fc["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(back_to_epsg)}"},
        }

    return fc
