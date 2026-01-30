#src/mapvec/features/map_pooling.py
import numpy as np
import pandas as pd

def pool_map_embedding(
    df_polys: pd.DataFrame,
    exclude=("id",),                 # columns not to pool
    stats=("mean","std","min","max"),
    quantiles=(0.25, 0.50, 0.75),    # add robust stats
    add_globals=True,                # extra global scalars
    max_polygons: float | int | None = None,  # <-- NEW: dataset-wide max polygon count
):
    """
    Turn a per-polygon feature table into one fixed-length map embedding.

    Notes
    -----
    - With the new polygon_features, centroid_x/centroid_y are already
      bbox-normalized to [0,1] within each map. The global 'spread' values
      below reflect the fraction of the map's extent actually used by the
      polygons, not absolute sizes.
    """
    # --- guards & numeric-only view ---
    if df_polys is None or len(df_polys) == 0:
        return np.zeros(0, dtype=np.float32), []

    # work only with numeric columns (common when a GeoDataFrame was the source)
    num_df = df_polys.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return np.zeros(0, dtype=np.float32), []

    # sanitize: inf → NaN → fill-safe ops via np.nan*
    num_df = num_df.replace([np.inf, -np.inf], np.nan)

    # determine feature columns to pool
    exclude_set = set(exclude or ())
    feat_cols = [c for c in num_df.columns if c not in exclude_set]
    if not feat_cols:
        return np.zeros(0, dtype=np.float32), []

    F = num_df[feat_cols].to_numpy(dtype=float)  # (N, d)

    names, parts = [], []

    # --- classic stats (nan-safe) ---
    if "mean" in stats:
        parts.append(np.nanmean(F, axis=0))
        names += [f"{c}__mean" for c in feat_cols]
    if "std" in stats:
        parts.append(np.nanstd(F, axis=0))  
        # population std (ddof=0) to match feature computation
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
            Q = np.nanquantile(F, q=q, axis=0, method="linear")
        except TypeError:
            Q = np.nanquantile(F, q=q, axis=0, interpolation="linear") #type: ignore
        for qq, row in zip(quantiles, Q):
            parts.append(row)
            names += [f"{c}__q{int(round(qq*100))}" for c in feat_cols]

    # --- optional global scalars ---
    if add_globals:
        N = float(len(df_polys))
        # NEW: normalized polygon count in [0,1]
        denom = float(max_polygons) if (max_polygons is not None and max_polygons > 0) else 1.0
        N_norm = N / denom
        
        # centroid-based spread in normalized space (0..1)
        if {"centroid_x", "centroid_y"}.issubset(df_polys.columns):
            cx = pd.to_numeric(df_polys["centroid_x"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            cy = pd.to_numeric(df_polys["centroid_y"], errors="coerce").replace([np.inf, -np.inf], np.nan)

            if cx.notna().any() and cy.notna().any():
                minx = float(np.nanmin(cx)); maxx = float(np.nanmax(cx))
                miny = float(np.nanmin(cy)); maxy = float(np.nanmax(cy))
                spread_w = max(maxx - minx, 0.0)          # fraction in [0,1]
                spread_h = max(maxy - miny, 0.0)          # fraction in [0,1]
            else:
                spread_w = spread_h = 0.0
        else:
            spread_w = spread_h = 0.0
        
        # --- coverage ratio (Σ normalized polygon areas / map area) ---
        area_ser = df_polys.get("area")
        if area_ser is None:
            area_ser = pd.Series(dtype=float)

        # explicitly tell type checker it's a Series[float]
        area_norm: pd.Series = pd.to_numeric(area_ser, errors="coerce")
        coverage_ratio = float(np.nansum(area_norm))
        coverage_ratio = max(0.0, min(1.0, coverage_ratio))  # clamp to [0,1]


        parts.append(np.array([N_norm, spread_w, spread_h, coverage_ratio], dtype=float))
        names += ["poly_count", "poly_spread_w", "poly_spread_h", "coverage_ratio"]


    # --- finalize ---
    if parts:
        vec = np.concatenate(parts, axis=0)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        vec = np.zeros(0, dtype=float)

    return vec.astype(np.float32), names