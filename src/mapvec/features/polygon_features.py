# src/mapvec/features/polygon_features.py
"""
polygon_features.py
Hand-crafted polygon feature extraction (vector embeddings) using Shapely + Pandas.

As a module:
    from polygon_features import embed_polygons_handcrafted
    df = embed_polygons_handcrafted(list_of_polygons_or_multipolygons)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Union
from contextlib import contextmanager
import warnings

import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.errors import TopologicalError
from shapely.prepared import prep

from src.constants import (
    # output schema
    POLY_FEATURE_ID_COL,
    POLY_FEATURE_ORDER,
    POLY_BY_DIAG_FEATURES,
    # normalization modes / defaults
    POLY_NORM_MODE_DEFAULT,
    POLY_NORM_FIXED_WH_DEFAULT,
    # numeric stability / eps
    EPS_POSITIVE,
    POLY_ECC_EPS,
    POLY_ECC_MAX,
    # density radii (fractions of map diagonal)
    POLY_DENSITY_R05_FRAC,
    POLY_DENSITY_R10_FRAC,
    # clipping / cleanup policy
    POLY_CLIP_QHI,
    # warnings policy
    SHAPELY_WARN_MODULE_PREDICATES,
    SHAPELY_WARN_MODULE_SETOPS,
)

GeometryLike = Union[Polygon, MultiPolygon]


# ---------- normalization helper ----------
def _stabilize_polygon_feats(feats: dict, bbox) -> dict:
    minx, miny, maxx, maxy = bbox
    bw = max(maxx - minx, float(EPS_POSITIVE))
    bh = max(maxy - miny, float(EPS_POSITIVE))
    bbox_area = bw * bh
    diag = (bw * bw + bh * bh) ** 0.5

    # divide lengths/distances by diag
    for k in POLY_BY_DIAG_FEATURES:
        if k in feats and np.isfinite(feats[k]):
            feats[k] = float(feats[k]) / diag

    # divide area by bbox area
    if "area" in feats and np.isfinite(feats["area"]):
        feats["area"] = float(feats["area"]) / bbox_area

    return feats


# ---------- warning suppressor used around GEOS calls ----------
@contextmanager
def _suppress_shapely_runtime_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=SHAPELY_WARN_MODULE_PREDICATES)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=SHAPELY_WARN_MODULE_SETOPS)
        yield


def _finite_coords(g: BaseGeometry) -> bool:
    if g.is_empty:
        return False
    xs, ys = zip(*g.exterior.coords)  # type: ignore
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return False
    for r in g.interiors:  # type: ignore
        xs, ys = zip(*r.coords)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            return False
    return True


def _fix_and_filter(geoms: Iterable[GeometryLike]) -> List[BaseGeometry]:
    """
    Make geometries valid with buffer(0) and drop empties/invalids.
    This prevents STRtree predicate warnings.
    """
    fixed: List[BaseGeometry] = []
    for g in geoms:
        if not isinstance(g, (Polygon, MultiPolygon)):
            raise TypeError(f"Unsupported geometry type: {type(g)}")
        try:
            gg = g.buffer(0)  # fixes many minor invalidities
        except Exception:
            continue
        if gg is None or gg.is_empty or not gg.is_valid or not _finite_coords(gg):
            continue
        fixed.append(gg)
    return fixed


def _min_rotated_rect_axes(poly: Polygon) -> Tuple[float, float]:
    """Return the two side lengths (a >= b) of the polygon's minimum rotated rectangle."""
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)  # type: ignore
    edges = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        edges.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    uniq = sorted({round(l, 12) for l in edges}, reverse=True)  # opposite sides repeat
    if len(uniq) < 2:
        return uniq[0], uniq[0]
    return uniq[0], uniq[1]


def _polygon_features_single(poly: Polygon, bbox=None) -> dict:
    """Compute compact features for a single Polygon. Centroid can be bbox-normalized."""
    if poly is None or poly.is_empty or not poly.is_valid:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "circularity": 0.0,
            "axis_ratio": 0.0,
            "convexity": 0.0,
            "rectangularity": 0.0,
        }

    area = poly.area
    perimeter = poly.length

    # centroid (optionally normalized)
    cx, cy = poly.centroid.x, poly.centroid.y
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        w = max(maxx - minx, float(EPS_POSITIVE))
        h = max(maxy - miny, float(EPS_POSITIVE))
        cx_n = (cx - minx) / w
        cy_n = (cy - miny) / h
    else:
        cx_n, cy_n = cx, cy

    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    a, b = _min_rotated_rect_axes(poly)
    axis_ratio = float(min(1.0, max(0.0, (b / a) if a > 0 else 0.0)))

    hull = poly.convex_hull
    convexity = (area / hull.area) if hull.area > 0 else 0.0

    mrr = poly.minimum_rotated_rectangle
    rectangularity = (area / mrr.area) if mrr.area > 0 else 0.0

    bw = max(poly.bounds[2] - poly.bounds[0], float(EPS_POSITIVE))
    bh = max(poly.bounds[3] - poly.bounds[1], float(EPS_POSITIVE))
    extent_fill = min(1.0, float(area / (bw * bh)))

    # orientation from the long edge of the MRR
    mrr_coords = list(mrr.exterior.coords)  # type: ignore
    best_len, best_ang = 0.0, 0.0
    for k in range(len(mrr_coords) - 1):
        x1, y1 = mrr_coords[k]
        x2, y2 = mrr_coords[k + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        L = (dx * dx + dy * dy) ** 0.5
        if L > best_len:
            best_len = L
            best_ang = math.degrees(math.atan2(dy, dx)) % 180.0
    theta = math.radians(best_ang)  # 0..π

    # moments / eccentricity
    xs, ys = zip(*list(poly.exterior.coords))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dxs = [x - mx for x in xs]
    dys = [y - my for y in ys]
    cxx = sum(v * v for v in dxs) / max(len(xs) - 1, 1)
    cyy = sum(v * v for v in dys) / max(len(xs) - 1, 1)
    cxy = sum(a0 * b0 for a0, b0 in zip(dxs, dys)) / max(len(xs) - 1, 1)
    vals = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]], dtype=float))
    lam2, lam1 = float(vals[0]), float(vals[1])

    if lam1 <= 0 or lam2 <= 0:
        eccentricity = 0.0 if lam1 <= 0 else float(POLY_ECC_MAX)
    else:
        r = math.sqrt(lam1 / lam2)
        eccentricity = (
            0.0
            if r < 1.0 + float(POLY_ECC_EPS)
            else min(math.sqrt(max(0.0, 1.0 - 1.0 / (r * r))), float(POLY_ECC_MAX))
        )

    has_hole = 1.0 if len(poly.interiors) > 0 else 0.0

    # reflex ratio (exterior only)
    ext = list(poly.exterior.coords)[:-1]
    reflex = 0
    n_vertices = len(ext)
    if n_vertices >= 3:
        for k in range(n_vertices):
            a0 = ext[(k - 1) % n_vertices]
            b0 = ext[k]
            c0 = ext[(k + 1) % n_vertices]
            cross = (b0[0] - a0[0]) * (c0[1] - b0[1]) - (b0[1] - a0[1]) * (c0[0] - b0[0])
            if cross < 0:
                reflex += 1
    reflex_ratio = reflex / max(n_vertices, 1)

    return {
        "area": area,
        "perimeter": perimeter,
        "centroid_x": cx_n,
        "centroid_y": cy_n,
        "circularity": circularity,
        "axis_ratio": axis_ratio,
        "convexity": convexity,
        "rectangularity": rectangularity,
        "bbox_width": bw,
        "bbox_height": bh,
        "orient_sin": math.sin(2 * theta),
        "orient_cos": math.cos(2 * theta),
        "eq_diameter": (2.0 * (area / math.pi) ** 0.5) if area > 0 else 0.0,
        "eccentricity": eccentricity,
        "has_hole": has_hole,
        "reflex_ratio": reflex_ratio,
        "extent_fill": extent_fill,
    }


def embed_polygons_handcrafted(
    geoms: Iterable[GeometryLike],
    norm_mode: str = POLY_NORM_MODE_DEFAULT,                     # "extent" | "fixed"
    fixed_wh: tuple[float, float] | None = None,                 # overrides POLY_NORM_FIXED_WH_DEFAULT
) -> pd.DataFrame:
    normalized: List[BaseGeometry] = _fix_and_filter(geoms)
    if not normalized:
        return pd.DataFrame(columns=[POLY_FEATURE_ID_COL] + list(POLY_FEATURE_ORDER))

    # extent (computed once)
    minx = min(g.bounds[0] for g in normalized)
    miny = min(g.bounds[1] for g in normalized)
    maxx = max(g.bounds[2] for g in normalized)
    maxy = max(g.bounds[3] for g in normalized)
    bbox_extent = (minx, miny, maxx, maxy)

    if norm_mode == "fixed":
        W_desired, H_desired = fixed_wh if fixed_wh is not None else POLY_NORM_FIXED_WH_DEFAULT
        W_desired = float(W_desired)
        H_desired = float(H_desired)

        ext_w = maxx - minx
        ext_h = maxy - miny

        pad_w_total = max(0.0, W_desired - ext_w)
        pad_h_total = max(0.0, H_desired - ext_h)

        pad_left = pad_w_total * 0.5
        pad_bottom = pad_h_total * 0.5

        origin_x = minx - pad_left
        origin_y = miny - pad_bottom

        scale_bbox = (0.0, 0.0, W_desired, H_desired)
        centroid_bbox = (origin_x, origin_y, origin_x + W_desired, origin_y + H_desired)
    else:
        scale_bbox = bbox_extent
        centroid_bbox = bbox_extent

    centroids = [g.centroid for g in normalized]

    dx = (scale_bbox[2] - scale_bbox[0])
    dy = (scale_bbox[3] - scale_bbox[1])
    map_diag = math.hypot(dx, dy)

    r05 = float(POLY_DENSITY_R05_FRAC) * map_diag
    r10 = float(POLY_DENSITY_R10_FRAC) * map_diag

    tree = STRtree(normalized)
    prepared = [prep(g) if (not g.is_empty and g.is_valid) else None for g in normalized]
    id_map = {id(g): k for k, g in enumerate(normalized)}
    wkb_map = {g.wkb: k for k, g in enumerate(normalized)}

    rows = []
    for i, geom in enumerate(normalized):
        base = (
            max(geom.geoms, key=lambda p: p.area).buffer(0)
            if isinstance(geom, MultiPolygon)
            else geom.buffer(0)
        )

        feats = _polygon_features_single(base, bbox=centroid_bbox)

        touch_candidates = list(tree.query(geom)) if (not geom.is_empty and geom.is_valid) else []
        inter_candidates = touch_candidates
        prep_geom = prepared[i] if (not geom.is_empty and geom.is_valid) else None

        def _safe_intersects(g1, g2, _prep=None) -> bool:
            if (g1.is_empty or not g1.is_valid) or (g2.is_empty or not g2.is_valid):
                return False
            try:
                with _suppress_shapely_runtime_warnings():
                    return (_prep or prep(g1)).intersects(g2)
            except (TopologicalError, ValueError):
                return False

        def _safe_touches(g1, g2) -> bool:
            if (g1.is_empty or not g1.is_valid) or (g2.is_empty or not g2.is_valid):
                return False
            try:
                with _suppress_shapely_runtime_warnings():
                    return g1.touches(g2)
            except (TopologicalError, ValueError):
                return False

        def _to_indices(cands, pred: str) -> List[int]:
            if cands is None:
                return []
            cands = list(cands)
            if len(cands) == 0:
                return []

            idxs: List[int] = []

            def _accept(j: int) -> bool:
                gj = normalized[j]
                if gj.is_empty or not gj.is_valid:
                    return False
                if pred == "touches":
                    return _safe_touches(geom, gj)
                if pred in ("overlap", "intersects"):
                    hits = _safe_intersects(geom, gj, _prep=prep_geom)
                    if not hits:
                        return False
                    if pred == "intersects":
                        return True
                    return (
                        not _safe_touches(geom, gj)
                        and not geom.contains(gj)
                        and not geom.within(gj)
                    )
                return False

            if isinstance(cands[0], (int, np.integer)):
                for j in map(int, cands):
                    if j != i and _accept(j):
                        idxs.append(j)
            else:
                for g2 in cands:
                    j = id_map.get(id(g2))
                    if j is None:
                        try:
                            wkb_key = g2.wkb if hasattr(g2, "wkb") else None
                        except Exception:
                            wkb_key = None
                        if wkb_key is not None:
                            j = wkb_map.get(wkb_key)
                    if j is not None and j != i and _accept(j):
                        idxs.append(j)
            return idxs

        nbr_touch = _to_indices(touch_candidates, "touches")
        nbr_inter = _to_indices(inter_candidates, "overlap")
        nbr = list(set(nbr_touch + nbr_inter))

        dists_union = [centroids[i].distance(centroids[j]) for j in nbr] if nbr else []
        if dists_union:
            s = sorted(dists_union)
            n = len(s)
            nn_median = 0.5 * (s[n // 2 - 1] + s[n // 2]) if n % 2 == 0 else s[n // 2]
        else:
            nn_median = 0.0
        feats["nn_dist_median"] = nn_median

        all_d = sorted(centroids[i].distance(centroids[j]) for j in range(len(normalized)) if j != i)
        feats["knn1"] = all_d[0] if len(all_d) >= 1 else 0.0
        feats["knn3"] = all_d[2] if len(all_d) >= 3 else (all_d[-1] if all_d else 0.0)

        feats["density_r05"] = sum(
            1 for j in range(len(normalized))
            if j != i and centroids[i].distance(centroids[j]) <= r05
        )
        feats["density_r10"] = sum(
            1 for j in range(len(normalized))
            if j != i and centroids[i].distance(centroids[j]) <= r10
        )

        k_any = len(nbr)
        if len(normalized) > 1:
            denom = float(len(normalized) - 1)
            feats["neighbor_count"] = math.log1p(k_any) / math.log1p(denom)
        else:
            feats["neighbor_count"] = 0.0
            denom = 1.0

        feats["density_r05"] /= denom
        feats["density_r10"] /= denom

        feats[POLY_FEATURE_ID_COL] = i + 1

        feats = _stabilize_polygon_feats(feats, bbox=scale_bbox)
        rows.append(feats)

    df = pd.DataFrame(rows)

    ordered = [POLY_FEATURE_ID_COL] + [c for c in POLY_FEATURE_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    # Optional: clip upper tail per feature within this map
    feat_cols = [c for c in df.columns if c != POLY_FEATURE_ID_COL]
    if len(df) > 0 and len(feat_cols) > 0:
        q_hi = df[feat_cols].quantile(float(POLY_CLIP_QHI), axis=0, numeric_only=True)
        for c in feat_cols:
            hi = q_hi.get(c, None)
            if hi is not None and np.isfinite(hi):
                df[c] = np.minimum(df[c], float(hi))

    # Final safety: replace NaN/Inf with per-column median, fallback to 0
    df = df.replace([np.inf, -np.inf], np.nan)
    feat_cols = [c for c in df.columns if c != POLY_FEATURE_ID_COL]
    for c in feat_cols:
        med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
        df[c] = df[c].fillna(float(med) if pd.notna(med) else 0.0)

    return df


# --------- Minimal CLI (optional) ----------
def _load_geojson_polygons(path: str) -> list:
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return list(gdf.geometry)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Extract stabilized polygon features to CSV.")
    parser.add_argument("input", help="Input GeoJSON")
    parser.add_argument("output", help="Output CSV")
    args = parser.parse_args(argv)

    geoms = _load_geojson_polygons(args.input)
    df = embed_polygons_handcrafted(geoms)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} polygon embeddings to {args.output}")


if __name__ == "__main__":
    main()
