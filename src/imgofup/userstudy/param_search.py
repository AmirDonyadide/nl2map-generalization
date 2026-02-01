# src/imgofup/userstudy/param_search.py
from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Targets (fraction of features affected) per intensity
# ============================================================

DEFAULT_SELECT_REMOVAL_TARGET: Dict[str, float] = {"low": 0.30, "medium": 0.50, "high": 0.70}
DEFAULT_AGG_MERGE_TARGET: Dict[str, float] = {"low": 0.30, "medium": 0.50, "high": 0.70}
DEFAULT_SIMPLIFY_CHANGE_TARGET: Dict[str, float] = {"low": 0.30, "medium": 0.50, "high": 0.70}
DEFAULT_DISPLACE_CHANGE_TARGET: Dict[str, float] = {"low": 0.30, "medium": 0.50, "high": 0.70}


# ============================================================
# Helpers
# ============================================================

def _ensure_nonempty(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(gdf, "crs", None))
    return gdf


def _nn_distances(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Nearest-neighbor distances between polygon centroids.
    Returns empty array if not enough features.
    """
    gdf = _ensure_nonempty(gdf)
    if len(gdf) < 2:
        return np.array([], dtype=float)

    cent = np.c_[gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
    nbrs = NearestNeighbors(n_neighbors=2).fit(cent)
    d = nbrs.kneighbors(cent, return_distance=True)[0][:, 1]
    return np.asarray(d, dtype=float)


def _grid(lo: float, hi: float, n_steps: int) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if n_steps <= 1:
        return np.array([lo], dtype=float)
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, int(n_steps), dtype=float)


def _search(values: np.ndarray, target: float, achieved_fn) -> Tuple[float, float]:
    """
    1D search: choose value minimizing |achieved - target|.
    Returns (best_value, best_err).
    """
    best_v = float(values[0])
    best_err = float("inf")

    for v in values:
        a = float(achieved_fn(float(v)))
        err = abs(a - float(target))
        if err < best_err:
            best_err = err
            best_v = float(v)

    return best_v, best_err


def _estimate_merge_fraction(gdf_in: gpd.GeoDataFrame, gdf_out: gpd.GeoDataFrame) -> float:
    """
    Approx fraction merged based on count reduction:
      frac = 1 - n_out / n_in
    """
    n_in = int(len(gdf_in))
    if n_in <= 0:
        return 0.0
    n_out = max(1, int(len(gdf_out)))
    frac = 1.0 - (n_out / n_in)
    return float(np.clip(frac, 0.0, 1.0))


def _estimate_simplify_change_fraction(
    gdf_in: gpd.GeoDataFrame,
    gdf_out: gpd.GeoDataFrame,
) -> float:
    """
    Your original notebook used “noticeable change” heuristics.
    Here we use a simple proxy: how much the feature count changed.
    This is stable and fast, and works well for intensity calibration.
    """
    n_in = int(len(gdf_in))
    if n_in <= 0:
        return 0.0
    n_out = int(len(gdf_out))
    return float(np.clip(abs(n_out - n_in) / n_in, 0.0, 1.0))


def estimate_displace_change_fraction(
    gdf_in: gpd.GeoDataFrame,
    gdf_out: gpd.GeoDataFrame,
    *,
    move_tol: float,
) -> float:
    """
    Fraction of geometries whose centroid moved more than move_tol (meters).
    Assumes indices are preserved when possible.
    """
    n_in = int(len(gdf_in))
    if n_in == 0:
        return 0.0

    c0 = gdf_in.geometry.centroid
    c1 = gdf_out.geometry.centroid

    moved = 0

    for idx, p0 in c0.items():
        p1 = c1.get(idx)
        if p1 is None:
            moved += 1
        elif p0.distance(p1) > float(move_tol):
            moved += 1

    return float(np.clip(moved / n_in, 0.0, 1.0))



# ============================================================
# Public API (names expected by sample_generation.py)
# ============================================================

def choose_select_param_for_tile(
    g_tile: gpd.GeoDataFrame,
    intensity: str,
    *,
    removal_target: Optional[Dict[str, float]] = None,
) -> float:
    """
    Select operator parameter = area threshold.
    Chooses threshold so ≈ fraction of smallest polygons removed.
    """
    g_tile = _ensure_nonempty(g_tile)
    if g_tile.empty or len(g_tile) <= 1:
        return 0.0

    targets = removal_target or DEFAULT_SELECT_REMOVAL_TARGET
    key = str(intensity).strip().lower()
    if key not in targets:
        raise ValueError(f"Unknown intensity '{intensity}'. Expected one of {sorted(targets.keys())}")

    frac = float(np.clip(targets[key], 0.0, 1.0))
    areas = np.asarray(g_tile.geometry.area.values, dtype=float)
    areas = areas[np.isfinite(areas)]
    if areas.size == 0:
        return 0.0

    return float(np.quantile(np.sort(areas), frac))


def choose_aggregate_param_for_tile(
    g_tile: gpd.GeoDataFrame,
    intensity: str,
    *,
    op_fn,
    n_steps: int = 8,
    merge_target: Optional[Dict[str, float]] = None,
) -> Tuple[float, gpd.GeoDataFrame]:
    """
    Aggregate operator parameter search.

    op_fn signature must be: op_fn(gdf, dist) -> gdf_out
      e.g. aggregate_buildings(gdf, dist=...)

    Returns:
      (best_dist, out_gdf_for_best_dist)
    """
    g_tile = _ensure_nonempty(g_tile)
    if g_tile.empty or len(g_tile) <= 1:
        return 0.0, g_tile.copy()

    targets = merge_target or DEFAULT_AGG_MERGE_TARGET
    key = str(intensity).strip().lower()
    if key not in targets:
        raise ValueError(f"Unknown intensity '{intensity}'. Expected one of {sorted(targets.keys())}")
    target = float(targets[key])

    dists = _nn_distances(g_tile)
    if dists.size == 0:
        # fall back to a conservative range
        lo, hi = 1.0, 6.0
    else:
        d25, d75 = np.quantile(dists, [0.25, 0.75])
        base = max(0.2, float(d25))
        lo = 0.0
        hi = max(float(d75) * 2.0, base * 2.5)

    candidates = _grid(lo, hi, n_steps)

    cache: Dict[float, gpd.GeoDataFrame] = {}

    def achieved_fn(dist: float) -> float:
        out = op_fn(g_tile, dist)
        out = _ensure_nonempty(out)
        cache[dist] = out
        return _estimate_merge_fraction(g_tile, out)

    best_dist, _ = _search(candidates, target, achieved_fn)
    best_out = cache.get(best_dist)
    if best_out is None:
        best_out = op_fn(g_tile, best_dist)

    return float(best_dist), best_out


def choose_simplify_param_for_tile(
    g_tile: gpd.GeoDataFrame,
    intensity: str,
    *,
    op_fn,
    n_steps: int = 6,
    change_target: Optional[Dict[str, float]] = None,
) -> Tuple[float, gpd.GeoDataFrame]:
    """
    Simplify operator parameter search.

    op_fn signature: op_fn(gdf, eps) -> gdf_out
      e.g. simplify_buildings(gdf, eps=...)

    Returns:
      (best_eps, out_gdf_for_best_eps)
    """
    g_tile = _ensure_nonempty(g_tile)
    if g_tile.empty:
        return 0.0, g_tile.copy()

    targets = change_target or DEFAULT_SIMPLIFY_CHANGE_TARGET
    key = str(intensity).strip().lower()
    if key not in targets:
        raise ValueError(f"Unknown intensity '{intensity}'. Expected one of {sorted(targets.keys())}")
    target = float(targets[key])

    areas = np.asarray(g_tile.geometry.area.values, dtype=float)
    areas = areas[np.isfinite(areas)]
    if areas.size == 0:
        med_side = 5.0
    else:
        med_side = float(np.median(np.sqrt(np.maximum(areas, 1e-6))))

    if key == "low":
        lo, hi = 0.1 * med_side, 0.3 * med_side
    elif key == "medium":
        lo, hi = 0.3 * med_side, 0.7 * med_side
    else:  # high
        lo, hi = 0.7 * med_side, 1.5 * med_side

    candidates = _grid(lo, hi, n_steps)

    cache: Dict[float, gpd.GeoDataFrame] = {}

    def achieved_fn(eps: float) -> float:
        out = op_fn(g_tile, eps)
        out = _ensure_nonempty(out)
        cache[eps] = out
        return _estimate_simplify_change_fraction(g_tile, out)

    best_eps, _ = _search(candidates, target, achieved_fn)
    best_out = cache.get(best_eps)
    if best_out is None:
        best_out = op_fn(g_tile, best_eps)

    return float(best_eps), best_out


def choose_displace_param_for_tile(
    g_tile: gpd.GeoDataFrame,
    intensity: str,
    *,
    op_fn,
    n_steps: int = 6,
    move_tol: float = 0.3,
    change_target: Optional[Dict[str, float]] = None,
) -> Tuple[float, gpd.GeoDataFrame]:
    """
    Displace operator parameter search.

    op_fn signature: op_fn(gdf, clearance) -> gdf_out
      e.g. displace_buildings(gdf, clearance=...)

    Returns:
      (best_clearance, out_gdf_for_best_clearance)
    """
    g_tile = _ensure_nonempty(g_tile)
    if g_tile.empty:
        return 0.0, g_tile.copy()

    targets = change_target or DEFAULT_DISPLACE_CHANGE_TARGET
    key = str(intensity).strip().lower()
    if key not in targets:
        raise ValueError(f"Unknown intensity '{intensity}'. Expected one of {sorted(targets.keys())}")
    target = float(targets[key])

    dists = _nn_distances(g_tile)
    if dists.size == 0:
        base, max_clear = 0.8, 3.0
    else:
        d25, d75 = np.quantile(dists, [0.25, 0.75])
        base = max(0.5, float(d25) * 0.5)
        max_clear = max(float(d75) * 1.2, base * 2.0)

    if key == "low":
        lo, hi = base * 0.5, base * 1.0
    elif key == "medium":
        lo, hi = base * 1.0, base * 1.8
    else:  # high
        lo, hi = base * 1.8, max_clear

    candidates = _grid(lo, hi, n_steps)

    cache: Dict[float, gpd.GeoDataFrame] = {}

    def achieved_fn(clearance: float) -> float:
        out = op_fn(g_tile, clearance)
        out = _ensure_nonempty(out)
        cache[clearance] = out
        return estimate_displace_change_fraction(g_tile, out, move_tol=float(move_tol))

    best_clear, _ = _search(candidates, target, achieved_fn)
    best_out = cache.get(best_clear)
    if best_out is None:
        best_out = op_fn(g_tile, best_clear)

    return float(best_clear), best_out
