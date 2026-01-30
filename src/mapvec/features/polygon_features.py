#src/mapvec/features/polygon_features.py
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

GeometryLike = Union[Polygon, MultiPolygon]

# Stable feature order for output columns (id is added separately)
_NUMERIC_ORDER = [
    "area",
    "perimeter",
    "centroid_x",
    "centroid_y",
    "circularity",
    "axis_ratio",
    "convexity",
    "rectangularity",
    "neighbor_count",
    "bbox_width",
    "bbox_height",
    "orient_sin",
    "orient_cos",
    "eq_diameter",
    "eccentricity",
    "has_hole",
    "reflex_ratio",
    "nn_dist_median",  
    "knn1",            
    "knn3",            
    "density_r05",
    "density_r10",
    "extent_fill",
]


_BY_DIAG = [
    "perimeter", "eq_diameter",
    "nn_dist_median",
    "knn1", "knn3",
    "bbox_width", "bbox_height",
]

def _stabilize_polygon_feats(feats: dict, bbox) -> dict:
    minx, miny, maxx, maxy = bbox
    bw = max(maxx - minx, 1e-12)
    bh = max(maxy - miny, 1e-12)
    bbox_area = bw * bh
    diag = (bw * bw + bh * bh) ** 0.5

    # divide lengths/distances by diag
    for k in _BY_DIAG:
        if k in feats and np.isfinite(feats[k]):
            feats[k] = float(feats[k]) / diag

    # divide area by bbox area
    if "area" in feats and np.isfinite(feats["area"]):
        feats["area"] = float(feats["area"]) / bbox_area

    # map orientation components from [-1,1] to [0,1]
    #for k in ("orient_sin", "orient_cos"):
    #    if k in feats and np.isfinite(feats[k]):
    #        feats[k] = 0.5 * (feats[k] + 1.0)

    return feats


# ---------- warning suppressor used around GEOS calls ----------
@contextmanager
def _suppress_shapely_runtime_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\.predicates")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\.set_operations")
        yield


def _finite_coords(g: BaseGeometry) -> bool:
    if g.is_empty:
        return False
    # exterior
    xs, ys = zip(*g.exterior.coords) #type: ignore
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return False
    # holes
    for r in g.interiors: #type: ignore
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
    coords = list(mrr.exterior.coords) #type: ignore
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
            "area": 0.0, "perimeter": 0.0,
            "centroid_x": 0.0, "centroid_y": 0.0,
            "circularity": 0.0, "axis_ratio": 0.0,
            "convexity": 0.0, "rectangularity": 0.0,
        }

    area = poly.area
    perimeter = poly.length

    # centroid (optionally normalized)
    cx, cy = poly.centroid.x, poly.centroid.y
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        w = max(maxx - minx, 1e-12)
        h = max(maxy - miny, 1e-12)
        cx_n = (cx - minx) / w
        cy_n = (cy - miny) / h
    else:
        cx_n, cy_n = cx, cy

    # shape descriptors
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    a, b = _min_rotated_rect_axes(poly)        # from min rotated rectangle
    axis_ratio = float(min(1.0, max(0.0, (b / a) if a > 0 else 0.0)))

    hull = poly.convex_hull
    convexity = (area / hull.area) if hull.area > 0 else 0.0

    mrr = poly.minimum_rotated_rectangle       # cache once
    rectangularity = (area / mrr.area) if mrr.area > 0 else 0.0

    # bbox width/height (no need for min/max vars)
    bw = max(poly.bounds[2] - poly.bounds[0], 1e-12)
    bh = max(poly.bounds[3] - poly.bounds[1], 1e-12)

    extent_fill = min(1.0, float(area / (bw * bh)))
    
    # orientation from the long edge of the MRR
    mrr_coords = list(mrr.exterior.coords)     # type: ignore
    best_len, best_ang = 0.0, 0.0
    for k in range(len(mrr_coords) - 1):
        x1, y1 = mrr_coords[k]
        x2, y2 = mrr_coords[k + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        L = (dx*dx + dy*dy) ** 0.5
        if L > best_len:
            best_len = L
            best_ang = math.degrees(math.atan2(dy, dx)) % 180.0
    theta = math.radians(best_ang)  # 0..π

    # moments / eccentricity
    xs, ys = zip(*list(poly.exterior.coords))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dxs = [x - mx for x in xs]; dys = [y - my for y in ys]
    cxx = sum(v*v for v in dxs) / max(len(xs)-1, 1)
    cyy = sum(v*v for v in dys) / max(len(xs)-1, 1)
    cxy = sum(a*b for a, b in zip(dxs, dys)) / max(len(xs)-1, 1)
    vals = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]], dtype=float))
    lam2, lam1 = float(vals[0]), float(vals[1])

    if lam1 <= 0 or lam2 <= 0:
        eccentricity = 0.0 if lam1 <= 0 else 0.999999
    else:
        r = math.sqrt(lam1 / lam2)
        eccentricity = 0.0 if r < 1.0 + 1e-12 else min(math.sqrt(max(0.0, 1.0 - 1.0/(r*r))), 0.999999)

    # holes
    has_hole = 1.0 if len(poly.interiors) > 0 else 0.0

    # reflex ratio (exterior only) — angle values not needed
    ext = list(poly.exterior.coords)[:-1]
    reflex = 0
    n_vertices = len(ext)
    if n_vertices >= 3:
        for k in range(n_vertices):
            a0 = ext[(k - 1) % n_vertices]; b0 = ext[k]; c0 = ext[(k + 1) % n_vertices]
            cross = (b0[0]-a0[0])*(c0[1]-b0[1]) - (b0[1]-a0[1])*(c0[0]-b0[0])
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
        "orient_sin": math.sin(2*theta),
        "orient_cos": math.cos(2*theta),
        "eq_diameter": (2.0 * (area / math.pi) ** 0.5) if area > 0 else 0.0,
        "eccentricity": eccentricity,
        "has_hole": has_hole,
        "reflex_ratio": reflex_ratio,
        "extent_fill": extent_fill,
    }

def embed_polygons_handcrafted(
    geoms: Iterable[GeometryLike],
    norm_mode: str = "extent",                      # "extent" | "fixed"
    fixed_wh: tuple[float, float] | None = None,    # (width, height) when norm_mode="fixed"
) -> pd.DataFrame:
    # 1) Clean/validate
    normalized: List[BaseGeometry] = _fix_and_filter(geoms)
    if not normalized:
        return pd.DataFrame(columns=["id"] + _NUMERIC_ORDER)

    # 2) Real polygon extent (always computed once)
    minx = min(g.bounds[0] for g in normalized)
    miny = min(g.bounds[1] for g in normalized)
    maxx = max(g.bounds[2] for g in normalized)
    maxy = max(g.bounds[3] for g in normalized)
    bbox_extent = (minx, miny, maxx, maxy)

    # Decide the normalization scales (ONE mode drives both diag/area AND centroid normalization)
    if norm_mode == "fixed":
        # desired fixed frame (defaults to 400x400 if not given)
        W_desired, H_desired = (fixed_wh if fixed_wh is not None else (400.0, 400.0))
        W_desired = float(W_desired); H_desired = float(H_desired)

        # current polygon extent
        ext_w = maxx - minx
        ext_h = maxy - miny

        # how much we are short of the desired size
        pad_w_total = max(0.0, W_desired - ext_w)
        pad_h_total = max(0.0, H_desired - ext_h)

        # split padding evenly on both sides
        pad_left   = pad_w_total * 0.5
        pad_right  = pad_w_total * 0.5
        pad_bottom = pad_h_total * 0.5
        pad_top    = pad_h_total * 0.5

        # anchor the centroid frame at the padded origin and make it exactly W_desired x H_desired
        origin_x = minx - pad_left
        origin_y = miny - pad_bottom

        # scale used for diag/area + density radii is the fixed size
        scale_bbox    = (0.0, 0.0, W_desired, H_desired)

        # centroid_x/centroid_y are normalized to the padded fixed frame
        centroid_bbox = (origin_x, origin_y, origin_x + W_desired, origin_y + H_desired)

    else:
        # extent mode (original behavior)
        scale_bbox    = bbox_extent
        centroid_bbox = bbox_extent


    centroids = [g.centroid for g in normalized]
    
    # Precompute map-scale distances once
    dx = (scale_bbox[2] - scale_bbox[0])
    dy = (scale_bbox[3] - scale_bbox[1])
    _map_diag = math.hypot(dx, dy)
    _r05 = 0.05 * _map_diag
    _r10 = 0.10 * _map_diag

    # 3) Indexes …
    tree = STRtree(normalized)
    prepared = [prep(g) if (not g.is_empty and g.is_valid) else None for g in normalized]
    id_map  = {id(g): k for k, g in enumerate(normalized)}
    wkb_map = {g.wkb: k for k, g in enumerate(normalized)}

    rows = []
    for i, geom in enumerate(normalized):
        base = (max(geom.geoms, key=lambda p: p.area).buffer(0) if isinstance(geom, MultiPolygon) else geom.buffer(0))

        # 👉 centroids normalized by the SAME mode (extent or fixed width/height)
        feats = _polygon_features_single(base, bbox=centroid_bbox)

        # neighbors …
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
            # normalize to list to avoid NumPy truthiness errors
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
                if pred in ("overlap","intersects"):
                    hits = _safe_intersects(geom, gj, _prep=prep_geom)
                    if not hits: 
                        return False
                    if pred == "intersects": 
                        return True
                    return (not _safe_touches(geom, gj)
                            and not geom.contains(gj)
                            and not geom.within(gj))
                return False

            # cands now a list; safe to look at first element
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
        nbr = list(set(nbr_touch + nbr_inter))  # union
        
        # Distances to union of neighbors (touch ∪ intersect)
        dists_union = [centroids[i].distance(centroids[j]) for j in nbr] if nbr else []
        if dists_union:
            s = sorted(dists_union)
            n = len(s)
            nn_median = 0.5 * (s[n//2 - 1] + s[n//2]) if n % 2 == 0 else s[n//2]
        else:
            nn_median = 0.0
        feats["nn_dist_median"] = nn_median


        # KNN over all others
        all_d = sorted(centroids[i].distance(centroids[j]) for j in range(len(normalized)) if j != i)
        feats["knn1"] = all_d[0] if len(all_d) >= 1 else 0.0
        feats["knn3"] = all_d[2] if len(all_d) >= 3 else (all_d[-1] if all_d else 0.0)
        
        # densities (same mode), then normalize by denom
        feats["density_r05"] = sum(
            1 for j in range(len(normalized))
            if j != i and centroids[i].distance(centroids[j]) <= _r05
        )
        feats["density_r10"] = sum(
            1 for j in range(len(normalized))
            if j != i and centroids[i].distance(centroids[j]) <= _r10
        )


        # merged neighbor count → log-fraction in [0,1]
        k_any = len(nbr)
        if len(normalized) > 1:
            denom = float(len(normalized) - 1)
            feats["neighbor_count"] = math.log1p(k_any) / math.log1p(denom)
        else:
            feats["neighbor_count"] = 0.0
            denom = 1.0

        feats["density_r05"] /= denom
        feats["density_r10"] /= denom
        
        feats["id"] = i + 1

        feats = _stabilize_polygon_feats(feats, bbox=scale_bbox)
        rows.append(feats)

    # 4) DataFrame + stable column order
    df = pd.DataFrame(rows)
    ordered = ["id"] + [c for c in _NUMERIC_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    # Optional: clip upper 0.5% per feature within this map to reduce sliver spikes
    feat_cols = [c for c in df.columns if c != "id"]
    if len(df) > 0 and len(feat_cols) > 0:
        q_hi = df[feat_cols].quantile(0.995, axis=0, numeric_only=True)
        for c in feat_cols:
            hi = q_hi.get(c, None)
            if hi is not None and np.isfinite(hi):
                df[c] = np.minimum(df[c], float(hi))

    # Final safety: replace any residual NaN/Inf with Median
    df = df.replace([np.inf, -np.inf], np.nan)
    feat_cols = [c for c in df.columns if c != "id"]
    for c in feat_cols:
        med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
        if pd.notna(med):
            df[c] = df[c].fillna(float(med))
        else:
            # column is all-NaN → harmless fallback
            df[c] = df[c].fillna(0.0)
    return df

# --------- Minimal CLI (optional) ----------
def _load_geojson_polygons(path: str) -> list:
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # default for GeoJSON; fine since we use bbox-normalization
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