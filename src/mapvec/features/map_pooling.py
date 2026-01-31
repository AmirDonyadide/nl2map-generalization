# src/mapvec/features/map_pooling.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    # pooling defaults
    MAP_POOL_EXCLUDE_COLS_DEFAULT,
    MAP_POOL_STATS_DEFAULT,
    MAP_POOL_QUANTILES_DEFAULT,
    MAP_POOL_ADD_GLOBALS_DEFAULT,
    # global-feature input columns
    POLY_CENTROID_X_COL,
    POLY_CENTROID_Y_COL,
    POLY_AREA_COL,
    # quantile settings
    MAP_POOL_QUANTILE_METHOD,
    # global-feature output names
    MAP_POOL_GLOBAL_FEATURE_NAMES,
    # clamps / numeric policy
    UNIT_INTERVAL_MIN,
    UNIT_INTERVAL_MAX,
)


def pool_map_embedding(
    df_polys: pd.DataFrame,
    exclude=MAP_POOL_EXCLUDE_COLS_DEFAULT,                 # columns not to pool
    stats=MAP_POOL_STATS_DEFAULT,
    quantiles=MAP_POOL_QUANTILES_DEFAULT,                 # robust stats
    add_globals=MAP_POOL_ADD_GLOBALS_DEFAULT,             # extra global scalars
    max_polygons: float | int | None = None,              # dataset-wide max polygon count (for normalization)
):
    """
    Turn a per-polygon feature table into one fixed-length map embedding.

    What changes when you change defaults (in constants.py)
    ------------------------------------------------------
    - exclude: which columns are ignored when pooling
    - stats: which nan-safe aggregations are included (mean/std/min/max)
    - quantiles: which quantiles are included as robust statistics
    - add_globals: whether extra global scalars are appended (count/spread/coverage)
    - MAP_POOL_GLOBAL_FEATURE_NAMES: names of the appended global scalars
    """
    # --- guards & numeric-only view ---
    if df_polys is None or len(df_polys) == 0:
        return np.zeros(0, dtype=np.float32), []

    # Work only with numeric columns (common when a GeoDataFrame was the source)
    num_df = df_polys.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return np.zeros(0, dtype=np.float32), []

    # sanitize: inf → NaN
    num_df = num_df.replace([np.inf, -np.inf], np.nan)

    # determine feature columns to pool
    exclude_set = set(exclude or ())
    feat_cols = [c for c in num_df.columns if c not in exclude_set]
    if not feat_cols:
        return np.zeros(0, dtype=np.float32), []

    F = num_df[feat_cols].to_numpy(dtype=float)  # (N, d)

    names: list[str] = []
    parts: list[np.ndarray] = []

    # --- classic stats (nan-safe) ---
    if "mean" in stats:
        parts.append(np.nanmean(F, axis=0))
        names += [f"{c}__mean" for c in feat_cols]
    if "std" in stats:
        # population std (ddof=0) to match feature computation
        parts.append(np.nanstd(F, axis=0))
        names += [f"{c}__std" for c in feat_cols]
    if "min" in stats:
        parts.append(np.nanmin(F, axis=0))
        names += [f"{c}__min" for c in feat_cols]
    if "max" in stats:
        parts.append(np.nanmax(F, axis=0))
        names += [f"{c}__max" for c in feat_cols]

    # --- quantiles (nan-safe) with NumPy version fallback ---
    if quantiles and F.size:
        q = np.array(list(quantiles), dtype=float)
        try:
            Q = np.nanquantile(F, q=q, axis=0, method=MAP_POOL_QUANTILE_METHOD)
        except TypeError:
            # older numpy uses "interpolation" kwarg
            Q = np.nanquantile(F, q=q, axis=0, interpolation=MAP_POOL_QUANTILE_METHOD)  # type: ignore

        for qq, row in zip(quantiles, Q):
            parts.append(row)
            names += [f"{c}__q{int(round(float(qq) * 100))}" for c in feat_cols]

    # --- optional global scalars ---
    if add_globals:
        N = float(len(df_polys))

        # normalized polygon count (roughly in [0,1] if max_polygons is dataset-wide max)
        denom = float(max_polygons) if (max_polygons is not None and float(max_polygons) > 0) else 1.0
        N_norm = N / denom

        # centroid-based spread in normalized space (0..1)
        if {POLY_CENTROID_X_COL, POLY_CENTROID_Y_COL}.issubset(df_polys.columns):
            cx = pd.to_numeric(df_polys[POLY_CENTROID_X_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
            cy = pd.to_numeric(df_polys[POLY_CENTROID_Y_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)

            if cx.notna().any() and cy.notna().any():
                minx = float(np.nanmin(cx)); maxx = float(np.nanmax(cx))
                miny = float(np.nanmin(cy)); maxy = float(np.nanmax(cy))
                spread_w = max(maxx - minx, 0.0)   # fraction in [0,1]
                spread_h = max(maxy - miny, 0.0)   # fraction in [0,1]
            else:
                spread_w = spread_h = 0.0
        else:
            spread_w = spread_h = 0.0

        # coverage ratio (Σ normalized polygon areas / map area)
        area_ser = df_polys.get(POLY_AREA_COL)
        if area_ser is None:
            area_ser = pd.Series(dtype=float)

        area_norm: pd.Series = pd.to_numeric(area_ser, errors="coerce")
        coverage_ratio = float(np.nansum(area_norm))

        # clamp to [0,1] to keep this feature bounded
        coverage_ratio = max(UNIT_INTERVAL_MIN, min(UNIT_INTERVAL_MAX, coverage_ratio))

        parts.append(np.array([N_norm, spread_w, spread_h, coverage_ratio], dtype=float))
        names += list(MAP_POOL_GLOBAL_FEATURE_NAMES)

    # --- finalize ---
    if parts:
        vec = np.concatenate(parts, axis=0)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        vec = np.zeros(0, dtype=float)

    return vec.astype(np.float32), names
