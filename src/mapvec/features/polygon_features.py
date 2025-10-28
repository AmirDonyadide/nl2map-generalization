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
    "vertex_count",
    "centroid_x",
    "centroid_y",
    "circularity",
    "elongation",
    "convexity",
    "rectangularity",
    "neighbor_count_touches",
    "mean_neighbor_distance_touches",
    "neighbor_count_intersects",
    "mean_neighbor_distance_intersects",
    "bbox_width",
    "bbox_height",
    "orient_sin",
    "orient_cos",
    "eq_diameter",
    "eccentricity",
    "has_hole",
    "reflex_count",
    "reflex_ratio",
    "nn_dist_min",
    "nn_dist_median",
    "nn_dist_max",
    "knn1",
    "knn2",
    "knn3",
    "density_r05",
    "density_r10",
]

# --- numeric stabilizer for per-polygon features (stateless; no leakage) ---
def _stabilize_polygon_feats(feats: dict, bbox) -> dict:
    """
    Normalize length/area/distance features by the map scale, then log1p heavy-tailed positives.
    """
    minx, miny, maxx, maxy = bbox
    bw = max(maxx - minx, 1e-12)
    bh = max(maxy - miny, 1e-12)
    bbox_area = bw * bh
    diag = (bw * bw + bh * bh) ** 0.5

    # ---- 1) Normalize to remove units ------------------------------
    # lengths / distances -> divide by diag
    _by_diag = [
        "perimeter",
        "mean_neighbor_distance_touches", "mean_neighbor_distance_intersects",
        "eq_diameter",
        "nn_dist_min", "nn_dist_median", "nn_dist_max",
        "knn1", "knn2", "knn3",
        "bbox_width", "bbox_height",
    ]
    for k in _by_diag:
        if k in feats and np.isfinite(feats[k]):
            feats[k] = float(feats[k]) / diag

    # areas -> divide by bbox area
    if "area" in feats and np.isfinite(feats["area"]):
        feats["area"] = float(feats["area"]) / bbox_area
        
    # ---- 2) log1p heavy-tailed positives (after normalization) ----
    _log_keys = {
        "vertex_count",
        "reflex_count",
        # DO NOT add area/perimeter/distances/bbox sizes here since they are already scale-normalized.
        # neighbor_count_* and density_* are already handled (N-normalize + log1p) outside.
    }
    for k in _log_keys:
        if k in feats:
            v = feats[k]
            if v is not None and np.isfinite(v) and v >= 0:
                feats[k] = float(np.log1p(v))
            elif v is None or not np.isfinite(v):
                feats[k] = 0.0  # make it finite

    # ---- 3) Shape ratio compression ----
    # Elongation (>=1): compress tail so 1 -> 0, 2 -> 0.69, 5 -> 1.61, 20 -> 3.00
    if "elongation" in feats and np.isfinite(feats["elongation"]):
        e = max(1.0, float(feats["elongation"]))     # ensure at least 1
        feats["elongation"] = float(np.log1p(e - 1.0))

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
    """Compute a compact set of features for a single Polygon. Centroid can be bbox-normalized."""
    # guard invalid / empty shapes
    if poly is None or poly.is_empty or not poly.is_valid:
        return {
            "area": 0.0, "perimeter": 0.0, "vertex_count": 0,
            "centroid_x": 0.0, "centroid_y": 0.0,
            "circularity": 0.0, "elongation": 0.0,
            "convexity": 0.0, "rectangularity": 0.0,
        }

    area = poly.area
    perimeter = poly.length
    vcount = max(len(poly.exterior.coords) - 1, 0)  # exclude closing point

    # centroid (optionally normalized by dataset bbox)
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

    a, b = _min_rotated_rect_axes(poly)
    elongation = (a / b) if b > 0 else 0.0

    hull = poly.convex_hull
    convexity = (area / hull.area) if hull.area > 0 else 0.0

    mrr = poly.minimum_rotated_rectangle
    rectangularity = (area / mrr.area) if mrr.area > 0 else 0.0

    # --- bounding box & orientation ---
    bxmin, bymin, bxmax, bymax = poly.bounds
    bw = max(bxmax - bxmin, 1e-12)
    bh = max(bymax - bymin, 1e-12)

    # orientation from minimum rotated rectangle long axis
    mrr = poly.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords) #type: ignore
    best_len, best_ang = 0.0, 0.0
    for k in range(len(mrr_coords) - 1):
        x1, y1 = mrr_coords[k]
        x2, y2 = mrr_coords[k + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        L = (dx*dx + dy*dy) ** 0.5
        if L > best_len:
            best_len = L
            best_ang = math.degrees(math.atan2(dy, dx)) % 180.0  # 0..180
    theta = math.radians(best_ang)  # 0..π

    # --- moments / eccentricity (covariance of exterior vertices) ---
    xs, ys = zip(*list(poly.exterior.coords))
    # simple (unweighted) covariance of boundary vertices
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dxs = [x - mx for x in xs]; dys = [y - my for y in ys]
    cxx = sum(v*v for v in dxs) / max(len(xs)-1, 1)
    cyy = sum(v*v for v in dys) / max(len(xs)-1, 1)
    cxy = sum(a*b for a,b in zip(dxs, dys)) / max(len(xs)-1, 1)

    # eigenvalues of 2x2 covariance (sorted: lam1 >= lam2 >= 0)
    vals = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]], dtype=float))
    lam2, lam1 = float(vals[0]), float(vals[1])

    if lam1 <= 0 or lam2 <= 0:
        # degenerate (line-like or point-like) → treat as almost 1
        eccentricity = 0.0 if lam1 <= 0 else 0.999999
    else:
        # axis ratio r = a/b = sqrt(lam1/lam2); true ellipse eccentricity e in [0,1)
        r = math.sqrt(lam1 / lam2)
        # numerical guard for near-isotropic shapes
        if r < 1.0 + 1e-12:
            eccentricity = 0.0
        else:
            eccentricity = math.sqrt(max(0.0, 1.0 - 1.0 / (r * r)))
            # hard clip to avoid rare >1 due to fp
            eccentricity = min(eccentricity, 0.999999)

    # --- holes ---
    has_hole = 1.0 if len(poly.interiors) > 0 else 0.0


    # --- vertex angle / reflex stats (exterior only) ---
    def _angle(a, b, c):
        # angle ABC in radians
        bax = a[0] - b[0]; bay = a[1] - b[1]
        bcx = c[0] - b[0]; bcy = c[1] - b[1]
        dot = bax*bcx + bay*bcy
        la = (bax*bax + bay*bay) ** 0.5
        lb = (bcx*bcx + bcy*bcy) ** 0.5
        if la == 0 or lb == 0: return 0.0
        cosv = max(-1.0, min(1.0, dot / (la*lb)))
        return math.acos(cosv)

    ext = list(poly.exterior.coords)[:-1]  # drop closing dup
    angles = []
    reflex = 0
    n_vertices = len(ext)
    if n_vertices >= 3:
        for k in range(n_vertices):
            a = ext[(k - 1) % n_vertices]; b = ext[k]; c = ext[(k + 1) % n_vertices]
            ang = _angle(a, b, c)
            angles.append(ang)
            cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if cross < 0:
                reflex += 1
                
    reflex_ratio = reflex / max(n_vertices, 1)

    feats = {
        "area": area,
        "perimeter": perimeter,
        "vertex_count": vcount,
        "centroid_x": cx_n,
        "centroid_y": cy_n,
        "circularity": circularity,
        "elongation": elongation,
        "convexity": convexity,
        "rectangularity": rectangularity,
        "bbox_width": bw,
        "bbox_height": bh,
        "orient_sin": math.sin(2*theta),
        "orient_cos": math.cos(2*theta),
        "eq_diameter": (2.0 * (area / math.pi) ** 0.5) if area > 0 else 0.0,
        "eccentricity": eccentricity,
        "has_hole": has_hole,
        "reflex_count": reflex,
        "reflex_ratio": reflex_ratio,
    }
    # Return raw feats; we stabilize later after adding neighbor/density fields.
    return feats

def embed_polygons_handcrafted(
    geoms: Iterable[GeometryLike],
) -> pd.DataFrame:
    """
    Compute hand-crafted feature vectors for Polygon/MultiPolygon geometries.
    Returns a DataFrame with one row per polygon and stabilized numeric features.
    """
    # 1) Clean/validate and fix minor invalidities
    normalized: List[BaseGeometry] = _fix_and_filter(geoms)
    if not normalized:
        return pd.DataFrame(columns=["id"] + _NUMERIC_ORDER)

    # 2) Dataset bbox for centroid normalization and centroids for distances
    minx = min(g.bounds[0] for g in normalized)
    miny = min(g.bounds[1] for g in normalized)
    maxx = max(g.bounds[2] for g in normalized)
    maxy = max(g.bounds[3] for g in normalized)
    bbox = (minx, miny, maxx, maxy)

    centroids = [g.centroid for g in normalized]

    # 3) Spatial index and prepared geoms (works for Shapely 1.x and 2.x)
    tree = STRtree(normalized)
    prepared = [prep(g) if (not g.is_empty and g.is_valid) else None for g in normalized]

    # one-time geometry → index maps (work for geometry-object returns)
    id_map  = {id(g): k for k, g in enumerate(normalized)}
    wkb_map = {g.wkb: k for k, g in enumerate(normalized)}

    rows = []
    for i, geom in enumerate(normalized):
        # Use largest part for MultiPolygon then buffer(0) to iron small defects.
        if isinstance(geom, MultiPolygon):
            try:
                largest = max(geom.geoms, key=lambda p: p.area)
            except ValueError:  # empty
                largest = geom
            base = largest.buffer(0)
        else:
            base = geom.buffer(0)

        feats = _polygon_features_single(base, bbox=bbox)

        if geom.is_empty or not geom.is_valid:
            touch_candidates = []
            inter_candidates = []
            prep_geom = None
        else:
            # bbox hits (no predicate) + prepared geometry for fast pairwise checks
            touch_candidates = tree.query(geom)
            inter_candidates = touch_candidates  # reuse bbox hits
            prep_geom = prepared[i]

        # ----- robust scalar predicates -----
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
            """Convert STRtree results (indices or geoms) to indices; apply manual predicate if needed."""
            idxs: List[int] = []
            size = getattr(cands, "size", None)
            if (size == 0) or (size is None and len(cands) == 0):
                return idxs
            if geom.is_empty or not geom.is_valid:
                return idxs

            first = cands[0]

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
                    return (not _safe_touches(geom, gj)
                            and not geom.contains(gj)
                            and not geom.within(gj))
                return False

            if isinstance(first, (int, np.integer)):
                for j in cands:
                    j = int(j)
                    if j != i and _accept(j):
                        idxs.append(j)
            else:
                for g2 in cands:
                    j = id_map.get(id(g2))
                    if j is None:
                        try:
                            j = wkb_map.get(g2.wkb)
                        except Exception:
                            j = None
                    if j is not None and j != i and _accept(j):
                        idxs.append(j)
            return idxs

        nbr_touch = _to_indices(touch_candidates, "touches")
        nbr_inter = _to_indices(inter_candidates, "overlap")  # exclude pure-touch cases

        # distances to neighbors (centroid to centroid)
        dists = [centroids[i].distance(centroids[j]) for j in nbr_touch] if nbr_touch else []

        feats["nn_dist_min"]     = min(dists) if dists else 0.0
        feats["nn_dist_median"]  = (sorted(dists)[len(dists)//2] if dists else 0.0)
        feats["nn_dist_max"]     = max(dists) if dists else 0.0

        # kNN (within all polygons, not just touches)
        all_d = sorted(centroids[i].distance(centroids[j]) for j in range(len(normalized)) if j != i)
        feats["knn1"] = all_d[0] if len(all_d) >= 1 else 0.0
        feats["knn2"] = all_d[1] if len(all_d) >= 2 else 0.0
        feats["knn3"] = all_d[2] if len(all_d) >= 3 else 0.0

        # local density in radii proportional to map diagonal
        dx = (bbox[2] - bbox[0]); dy = (bbox[3] - bbox[1])
        diag = (dx*dx + dy*dy) ** 0.5
        r05 = 0.05 * diag; r10 = 0.10 * diag
        feats["density_r05"] = sum(1 for j in range(len(normalized)) if j != i and centroids[i].distance(centroids[j]) <= r05)
        feats["density_r10"] = sum(1 for j in range(len(normalized)) if j != i and centroids[i].distance(centroids[j]) <= r10)

        feats["neighbor_count_touches"] = len(nbr_touch)
        feats["mean_neighbor_distance_touches"] = (
            sum(centroids[i].distance(centroids[j]) for j in nbr_touch) / len(nbr_touch)
            if nbr_touch else 0.0
        )

        feats["neighbor_count_intersects"] = len(nbr_inter)
        feats["mean_neighbor_distance_intersects"] = (
            sum(centroids[i].distance(centroids[j]) for j in nbr_inter) / len(nbr_inter)
            if nbr_inter else 0.0
        )

        # Size-normalize counts/densities by (N-1)
        N = float(len(normalized))
        feats["neighbor_count_touches"]    = feats["neighbor_count_touches"] / max(N-1, 1.0)
        feats["neighbor_count_intersects"] = feats["neighbor_count_intersects"] / max(N-1, 1.0)
        feats["density_r05"]               = feats["density_r05"] / max(N-1, 1.0)
        feats["density_r10"]               = feats["density_r10"] / max(N-1, 1.0)

        feats["id"] = i + 1

        # Final stabilization AFTER all features are present
        feats = _stabilize_polygon_feats(feats, bbox=bbox)
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