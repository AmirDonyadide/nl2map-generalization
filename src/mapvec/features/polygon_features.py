"""
polygon_embeddings.py
Hand-crafted polygon feature extraction (vector embeddings) using Shapely + Pandas.

Usage (CLI):
    python polygon_embeddings.py input.geojson output.csv [normalize] [--utm]

    normalize:
        none    (default)
        zscore  (mean=0, std=1)
        minmax  ([0,1] per column)

As a module:
    from polygon_embeddings import embed_polygons_handcrafted
"""

from __future__ import annotations

import math
import sys
from numbers import Integral
from typing import Iterable, List, Tuple, Union, Optional
from contextlib import contextmanager
import warnings

import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.strtree import STRtree
from shapely.errors import TopologicalError
from shapely.prepared import prep

# If you prefer no external deps, remove sklearn and use _zscore/_minmax below.
from sklearn.preprocessing import StandardScaler, MinMaxScaler

GeometryLike = Union[Polygon, MultiPolygon]

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
    xs, ys = zip(*g.exterior.coords)
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return False
    # holes
    for r in g.interiors:
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
    coords = list(mrr.exterior.coords)
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
            "compactness": 0.0, "circularity": 0.0, "elongation": 0.0,
            "convexity": 0.0, "rectangularity": 0.0, "straightness": 0.0,
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
    compactness = (perimeter ** 2) / (4 * math.pi * area) if area > 0 else 0.0
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    a, b = _min_rotated_rect_axes(poly)
    elongation = (a / b) if b > 0 else 0.0

    hull = poly.convex_hull
    convexity = (area / hull.area) if hull.area > 0 else 0.0

    mrr = poly.minimum_rotated_rectangle
    rectangularity = (area / mrr.area) if mrr.area > 0 else 0.0

    # >= 1.0; closer to 1 means boundary close to convex hull
    straightness = (perimeter / hull.length) if hull.length > 0 else 0.0

    # --- bounding box & orientation ---
    bxmin, bymin, bxmax, bymax = poly.bounds
    bw = max(bxmax - bxmin, 1e-12)
    bh = max(bymax - bymin, 1e-12)
    bbox_area = bw * bh
    bbox_aspect = bw / bh if bh > 0 else 0.0
    extent = area / bbox_area if bbox_area > 0 else 0.0

    # orientation from minimum rotated rectangle long axis
    mrr = poly.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    best_len, best_ang = 0.0, 0.0
    for k in range(len(mrr_coords) - 1):
        x1, y1 = mrr_coords[k]
        x2, y2 = mrr_coords[k + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        L = (dx*dx + dy*dy) ** 0.5
        if L > best_len:
            best_len = L
            best_ang = math.degrees(math.atan2(dy, dx)) % 180.0  # 0..180
    orientation = best_ang

    # --- moments / eccentricity (covariance of exterior vertices) ---
    xs, ys = zip(*list(poly.exterior.coords))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dxs = [x - mx for x in xs]; dys = [y - my for y in ys]
    cxx = sum(v*v for v in dxs) / max(len(xs)-1, 1)
    cyy = sum(v*v for v in dys) / max(len(xs)-1, 1)
    cxy = sum(a*b for a,b in zip(dxs, dys)) / max(len(xs)-1, 1)
    tr = cxx + cyy
    det = cxx*cyy - cxy*cxy
    delta = max(0.0, tr*tr/4 - det)
    lam1 = tr/2 + delta**0.5
    lam2 = tr/2 - delta**0.5
    eccentricity = (lam1 / max(lam2, 1e-12)) if lam1 >= lam2 and lam2 > 0 else 0.0

    # --- holes ---
    hole_count = len(poly.interiors)
    hole_area = sum(Polygon(ring).area for ring in poly.interiors)
    hole_area_ratio = (hole_area / area) if area > 0 else 0.0

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
    n = len(ext)
    if n >= 3:
        for k in range(n):
            a = ext[(k - 1) % n]; b = ext[k]; c = ext[(k + 1) % n]
            ang = _angle(a, b, c)
            angles.append(ang)
            cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if cross < 0:
                reflex += 1
    angle_std = (sum((v - (sum(angles)/max(len(angles),1)))**2 for v in angles)/max(len(angles),1))**0.5 if angles else 0.0
    reflex_ratio = reflex / max(n, 1)

    return {
        "area": area,
        "perimeter": perimeter,
        "vertex_count": vcount,
        "centroid_x": cx_n,
        "centroid_y": cy_n,
        "compactness": compactness,
        "circularity": circularity,
        "elongation": elongation,
        "convexity": convexity,
        "rectangularity": rectangularity,
        "straightness": straightness,
        "bbox_width": bw,
        "bbox_height": bh,
        "bbox_aspect": bbox_aspect,
        "extent": extent,
        "orientation": orientation,
        "eq_diameter": (2.0 * (area / math.pi) ** 0.5) if area > 0 else 0.0,
        "eccentricity": eccentricity,
        "hole_count": hole_count,
        "hole_area_ratio": hole_area_ratio,
        "angle_std": angle_std,
        "reflex_count": reflex,
        "reflex_ratio": reflex_ratio,
    }

# ---- CRS helpers (use only if your inputs are in degrees) ----

def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Pick a local UTM EPSG based on a lon/lat centroid."""
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return f"EPSG:{32700 + zone if south else 32600 + zone}"

try:
    from pyproj import Transformer
    def reproject_polys_to_local_utm(polys: List[GeometryLike], src_epsg: str = "EPSG:4326"):
        """
        Reproject a list of Shapely (Multi)Polygons from src_epsg to a local UTM zone
        chosen by the dataset centroid (computed in WGS84).
        Returns (projected_polys, target_epsg).
        """
        # if src is not WGS84, convert to WGS84 just for centroid calculation
        if src_epsg != "EPSG:4326":
            to_wgs84 = Transformer.from_crs(src_epsg, "EPSG:4326", always_xy=True).transform
            polys_wgs84 = [transform(to_wgs84, g) for g in polys]
        else:
            polys_wgs84 = polys

        # union centroid
        union = polys_wgs84[0]
        for g in polys_wgs84[1:]:
            union = union.union(g)
        lon, lat = union.centroid.x, union.centroid.y

        target_epsg = utm_epsg_from_lonlat(lon, lat)
        tfm = Transformer.from_crs(src_epsg, target_epsg, always_xy=True).transform
        polys_proj = [transform(tfm, g) for g in polys]
        return polys_proj, target_epsg
except Exception:
    def reproject_polys_to_local_utm(polys: List[GeometryLike], src_epsg: str = "EPSG:4326"):
        raise ImportError("pyproj is required for CRS reprojection. Install 'pyproj' to use this helper.")

# Stable feature order for output columns (id is added separately)
_NUMERIC_ORDER = [
    "area",
    "perimeter",
    "vertex_count",
    "centroid_x",
    "centroid_y",
    "compactness",
    "circularity",
    "elongation",
    "convexity",
    "rectangularity",
    "straightness",
    "neighbor_count_touches",
    "mean_neighbor_distance_touches",
    "neighbor_count_intersects",
    "mean_neighbor_distance_intersects",
    "bbox_width",
    "bbox_height",
    "bbox_aspect",
    "extent",
    "orientation",
    "eq_diameter",
    "eccentricity",
    "hole_count",
    "hole_area_ratio",
    "angle_std",
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

# Optional: pure-Pandas normalizers if you don't want sklearn
def _zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        mu = d[c].mean()
        sd = d[c].std(ddof=0)
        d[c] = 0.0 if sd == 0 else (d[c] - mu) / sd
    return d

def _minmax(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        mn = d[c].min()
        mx = d[c].max()
        rng = mx - mn
        d[c] = 0.0 if rng == 0 else (d[c] - mn) / rng
    return d

def embed_polygons_handcrafted(
    geoms: Iterable[GeometryLike],
    normalize: Union[bool, str] = False,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, object]]:
    """
    Compute hand-crafted feature vectors for Polygon/MultiPolygon geometries.

    Parameters
    ----------
    geoms : iterable of shapely Polygon or MultiPolygon
    normalize : bool or str
        - False (default): return raw features
        - True or "zscore": standardize (mean=0, std=1)
        - "minmax": scale to [0,1]
    return_scaler : bool
        If True, also return the fitted scaler (for reuse at inference).
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
            # robust emptiness check for list/tuple/ndarray
            size = getattr(cands, "size", None)
            if (size == 0) or (size is None and len(cands) == 0):
                return idxs
            if geom.is_empty or not geom.is_valid:
                return idxs

            # First element tells us if results are indices or geoms
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
                    # pred == "overlap": intersects but not touches, no containment
                    return (not _safe_touches(geom, gj)
                            and not geom.contains(gj)
                            and not geom.within(gj))
                return False

            if isinstance(first, Integral):
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

        feats["id"] = i + 1
        rows.append(feats)

    # 4) DataFrame + stable column order
    df = pd.DataFrame(rows)
    ordered = ["id"] + [c for c in _NUMERIC_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    # 5) Optional normalization
    scaler_obj: Optional[object] = None
    if normalize:
        scaler_obj = StandardScaler() if normalize in (True, "zscore") else MinMaxScaler()
        feat_cols = [c for c in df.columns if c != "id"]
        df[feat_cols] = scaler_obj.fit_transform(df[feat_cols])

    return (df, scaler_obj) if return_scaler else df

# ---------------- CLI with GeoJSON + auto-CRS ----------------

# Optional CLI: requires GeoPandas. If you don't need CLI, you can delete this block.
def _load_geojson_polygons(project_to_local_utm: bool, path: str) -> List[GeometryLike]:
    """
    Load (Multi)Polygons from GeoJSON.
    If project_to_local_utm=True, reproject to a local UTM chosen by the dataset centroid.
    """
    try:
        import geopandas as gpd
    except Exception as e:
        raise ImportError("GeoPandas is required for CLI loading.") from e

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # GeoJSON default

    if not project_to_local_utm:
        return list(gdf.geometry)

    # choose UTM by dataset centroid in WGS84
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    lon = gdf_wgs84.unary_union.centroid.x
    lat = gdf_wgs84.unary_union.centroid.y
    target = utm_epsg_from_lonlat(lon, lat)
    gdf_proj = gdf.to_crs(target)
    return list(gdf_proj.geometry)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def main(argv=None):
    """
    CLI:
        python polygon_embeddings.py input.geojson output.csv [normalize] [--utm]
    where:
        normalize ∈ {none, zscore, minmax}  (default: none)
        --utm: reproject input to local UTM before feature extraction.
    """
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print("Usage: python polygon_embeddings.py <input.geojson> <output.csv> [normalize] [--utm]")
        print(" normalize: none (default) | zscore | minmax")
        print(" --utm: reproject to local UTM for metric area/length/distance")
        sys.exit(1)

    in_path = argv[0]
    out_path = argv[1]
    norm = "none"
    project_to_utm = False

    for a in argv[2:]:
        if a.lower() in ("none", "zscore", "minmax"):
            norm = a.lower()
        elif a == "--utm":
            project_to_utm = True

    norm_arg: Optional[Union[bool, str]] = (False if norm == "none"
                                            else ("zscore" if norm == "zscore" else "minmax"))

    geoms = _load_geojson_polygons(project_to_utm, in_path)
    df = embed_polygons_handcrafted(geoms, normalize=norm_arg, return_scaler=False)
    _save_csv(df, out_path)
    print(f"Saved {len(df)} polygon embeddings to {out_path} (normalize={norm}, utm={project_to_utm})")


if __name__ == "__main__":
    main()
