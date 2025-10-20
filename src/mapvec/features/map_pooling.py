import numpy as np
import pandas as pd

def pool_map_embedding(
    df_polys: pd.DataFrame,
    exclude=("id",),                 # columns not to pool
    stats=("mean","std","min","max"),
    quantiles=(0.25, 0.50, 0.75),    # add robust stats
    add_globals=True                 # extra global scalars
):
    """
    Turn a per-polygon feature table into one fixed-length map embedding.

    Returns
    -------
    vec : np.ndarray  shape (D,)
    names : list[str] feature names aligned with vec
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
        # population std to mirror your original ddof=0
        parts.append(np.nanstd(F, axis=0))
        names += [f"{c}__std" for c in feat_cols]
    if "min" in stats:
        parts.append(np.nanmin(F, axis=0))
        names += [f"{c}__min" for c in feat_cols]
    if "max" in stats:
        parts.append(np.nanmax(F, axis=0))
        names += [f"{c}__max" for c in feat_cols]

    # --- quantiles (nan-safe) ---
    if quantiles and F.size:
        q = np.array(list(quantiles), dtype=float)
        Q = np.nanquantile(F, q=q, axis=0, method="linear")  # shape: (len(q), d)
        for qq, row in zip(quantiles, Q):
            parts.append(row)
            names += [f"{c}__q{int(round(qq*100))}" for c in feat_cols]

    # --- optional global scalars ---
    if add_globals:
        N = float(len(df_polys))
        # Only compute bbox if centroid columns are present; otherwise fallback to zeros
        if {"centroid_x", "centroid_y"}.issubset(df_polys.columns):
            cx = pd.to_numeric(df_polys["centroid_x"], errors="coerce")
            cy = pd.to_numeric(df_polys["centroid_y"], errors="coerce")
            cx = cx.replace([np.inf, -np.inf], np.nan)
            cy = cy.replace([np.inf, -np.inf], np.nan)

            minx = float(np.nanmin(cx)) if cx.notna().any() else 0.0
            miny = float(np.nanmin(cy)) if cy.notna().any() else 0.0
            maxx = float(np.nanmax(cx)) if cx.notna().any() else 0.0
            maxy = float(np.nanmax(cy)) if cy.notna().any() else 0.0

            bbox_w = max(maxx - minx, 1e-12)
            bbox_h = max(maxy - miny, 1e-12)
            bbox_aspect = bbox_w / bbox_h
        else:
            bbox_w = 0.0
            bbox_h = 0.0
            bbox_aspect = 0.0

        parts.append(np.array([N, bbox_w, bbox_h, bbox_aspect], dtype=float))
        names += ["poly_count", "map_bbox_w", "map_bbox_h", "map_bbox_aspect"]

    # --- finalize ---
    if parts:
        vec = np.concatenate(parts, axis=0)
        # replace remaining NaNs (if any all-NaN columns were present)
        if np.isnan(vec).any():
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        vec = np.zeros(0, dtype=float)

    return vec.astype(np.float32), names