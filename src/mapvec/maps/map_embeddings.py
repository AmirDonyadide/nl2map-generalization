# src/mapvec/maps/map_embeddings.py
# Compute fixed-length map vectors from GeoJSONs and save artifacts.
# Drop-in compatible with the original script (same CLI, defaults, outputs).

from __future__ import annotations

import sys, json, argparse, logging, warnings
from pathlib import Path
from typing import Iterator, Tuple, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon

from src.mapvec.features.polygon_features import embed_polygons_handcrafted
from src.mapvec.features.map_pooling import pool_map_embedding

from src.constants import (
    # repo/data roots
    PROJECT_ROOT_MARKER_LEVELS_UP,
    DEFAULT_DATA_DIRNAME,
    # mapvec default paths
    MAP_EMBED_ROOT_DEFAULT,
    MAP_EMBED_PATTERN_DEFAULT,
    MAP_EMBED_OUTDIR_DEFAULT,
    MAP_EMBED_VERBOSE_DEFAULT,
    MAP_EMBED_SAVE_CSV_DEFAULT,
    # normalization defaults
    POLY_NORM_MODE_DEFAULT,
    POLY_NORM_FIXED_WH_DEFAULT,
    MAP_EMBED_PROJECT_IF_GEOGRAPHIC,
    MAP_EMBED_GEOGRAPHIC_CRS_EPSG,
    MAP_EMBED_METRIC_CRS_EPSG,
    # numeric eps + filtering
    EPS_POSITIVE,
    MAP_POLY_AREA_EPS,
    # warning filters
    MAP_EMBED_WARNINGS_TO_IGNORE,
    # schema / columns / filenames
    MAPS_ID_COL,
    MAP_GEOJSON_COL,
    MAP_N_POLYGONS_COL,
    MAP_EMBEDDINGS_NPZ_NAME,
    MAPS_PARQUET_NAME,
    MAPS_CSV_NAME,
    MAPS_META_JSON_NAME,
    MAPS_FEATURE_NAMES_JSON_NAME,
    # extent columns
    EXTENT_MINX_COL,
    EXTENT_MINY_COL,
    EXTENT_MAXX_COL,
    EXTENT_MAXY_COL,
    EXTENT_WIDTH_COL,
    EXTENT_HEIGHT_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    # pooling defaults (already moved in previous steps)
    MAP_POOL_EXCLUDE_COLS_DEFAULT,
    MAP_POOL_STATS_DEFAULT,
    MAP_POOL_QUANTILES_DEFAULT,
    MAP_POOL_ADD_GLOBALS_DEFAULT,
)

# Apply warning filters (keeps logs readable on messy geometries)
for msg in MAP_EMBED_WARNINGS_TO_IGNORE:
    warnings.filterwarnings("ignore", message=msg)

# ----------------------- paths -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[int(PROJECT_ROOT_MARKER_LEVELS_UP)]
DATA_DIR = (PROJECT_ROOT / DEFAULT_DATA_DIRNAME).resolve()


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    cand = (Path.cwd() / p).resolve()
    return cand if cand.exists() or cand.parent.exists() else (DATA_DIR / p).resolve()


def setup_logging(verbosity: int = MAP_EMBED_VERBOSE_DEFAULT) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.debug("PROJECT_ROOT=%s", PROJECT_ROOT)
    logging.debug("DATA_DIR=%s", DATA_DIR)


# ----------------------- core helpers -----------------------
def find_geojsons(root: Path, pattern: str) -> Iterator[Tuple[str, Path]]:
    """
    Yield (map_id, path) for every subfolder in `root` that has a file matching `pattern`.
    Example: data/samples/pairs/0001/0001_input.geojson
    """
    if not root.exists():
        return
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        hits = list(sub.glob(pattern))
        if not hits:
            logging.debug("No match in %s for pattern %s", sub, pattern)
            continue
        yield sub.name, hits[0]


def _read_geo(gj_path: Path) -> gpd.GeoDataFrame:
    """Read a vector file robustly; filter empties/nulls."""
    gdf = gpd.read_file(gj_path)

    # If CRS missing, assume WGS84 (typical for GeoJSON)
    if gdf.crs is None:
        gdf = gdf.set_crs(MAP_EMBED_GEOGRAPHIC_CRS_EPSG)

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty.")
    if "geometry" not in gdf.columns:
        raise ValueError("No 'geometry' column found.")

    gdf = gdf[gdf.geometry.notnull()].copy()
    if "is_empty" in dir(gdf.geometry):  # geopandas>=0.10
        gdf = gdf[~gdf.geometry.is_empty].copy()
    else:
        gdf = gdf[[not geom.is_empty for geom in gdf.geometry]].copy()

    if gdf.empty:
        raise ValueError("All geometries were empty/invalid.")
    return gdf


def _maybe_project_to_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project geographic CRS to metric CRS for distance/area-consistent measures."""
    if not MAP_EMBED_PROJECT_IF_GEOGRAPHIC:
        return gdf
    try:
        if gdf.crs and getattr(gdf.crs, "is_geographic", False):
            return gdf.to_crs(MAP_EMBED_METRIC_CRS_EPSG)
    except Exception:
        pass
    return gdf


def compute_extent_refs(gj_path: Path) -> Dict[str, float]:
    """
    Compute per-map extent metrics from total_bounds.

    If CRS is geographic and MAP_EMBED_PROJECT_IF_GEOGRAPHIC is True, project to meters first.
    """
    gdf = _maybe_project_to_meters(_read_geo(gj_path))
    minx, miny, maxx, maxy = map(float, gdf.total_bounds)

    w = maxx - minx
    h = maxy - miny

    if not np.isfinite(w) or w < 0:
        w = float("nan")
    if not np.isfinite(h) or h < 0:
        h = float("nan")

    diag = float(np.sqrt(w * w + h * h)) if np.isfinite(w) and np.isfinite(h) else float("nan")
    area = float(w * h) if np.isfinite(w) and np.isfinite(h) else float("nan")

    return {
        EXTENT_MINX_COL: float(minx),
        EXTENT_MINY_COL: float(miny),
        EXTENT_MAXX_COL: float(maxx),
        EXTENT_MAXY_COL: float(maxy),
        EXTENT_WIDTH_COL: float(w),
        EXTENT_HEIGHT_COL: float(h),
        EXTENT_DIAG_COL: float(diag),
        EXTENT_AREA_COL: float(area),
    }


def _iter_polygons(geom):
    """Yield Polygon parts from any geometry (Polygon, MultiPolygon, GeometryCollection)."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype == "MultiPolygon":
        for p in geom.geoms:
            if not p.is_empty:
                yield p
    elif gtype == "GeometryCollection":
        for g in geom.geoms:
            yield from _iter_polygons(g)
    else:
        return


def _fix_polygon(poly: Polygon) -> Polygon | None:
    """Try to repair/clean a polygon; return None if still bad."""
    try:
        p = poly.buffer(0)
        if not p.is_valid or p.is_empty:
            return None
        if isinstance(p, MultiPolygon):
            parts = [q for q in p.geoms if q.is_valid and not q.is_empty]
            if not parts:
                return None
            p = max(parts, key=lambda q: q.area)
        return p if isinstance(p, Polygon) else None
    except Exception:
        return None


def _flatten_and_clean_to_polygons(gdf: gpd.GeoDataFrame, *, area_eps: float = MAP_POLY_AREA_EPS) -> List[Polygon]:
    """Return a list of valid Polygons suitable for feature extraction."""
    polys: List[Polygon] = []
    for geom in gdf.geometry:
        for p in _iter_polygons(geom):
            p2 = _fix_polygon(p)
            if p2 is None:
                continue
            if not np.isfinite(p2.area) or p2.area <= float(area_eps):
                continue
            polys.append(p2)
    return polys


def _count_valid_polygons(gj_path: Path) -> int:
    """Read, clean, and count valid polygon parts in a single map file."""
    gdf = _maybe_project_to_meters(_read_geo(gj_path))
    geoms = _flatten_and_clean_to_polygons(gdf)
    return len(geoms)


def embed_one_map(
    gj_path: Path,
    *,
    max_polygons: int | float | None = None,
    norm: str = POLY_NORM_MODE_DEFAULT,
    norm_wh: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    gdf = _maybe_project_to_meters(_read_geo(gj_path))
    geoms = _flatten_and_clean_to_polygons(gdf)

    if len(geoms) == 0:
        raise ValueError("No valid polygon parts after flatten/clean.")

    # Parse norm_wh (e.g., "400x400") only if norm == "fixed"
    fixed = None
    if norm == "fixed":
        if not norm_wh or "x" not in norm_wh.lower():
            raise ValueError("Use --norm-wh like '400x400' with --norm=fixed")
        w_str, h_str = norm_wh.lower().split("x", 1)
        fixed = (float(w_str), float(h_str))
    else:
        fixed = None

    df_polys = embed_polygons_handcrafted(geoms, norm_mode=norm, fixed_wh=fixed)

    vec, names = pool_map_embedding(
        df_polys,
        exclude=MAP_POOL_EXCLUDE_COLS_DEFAULT,
        stats=MAP_POOL_STATS_DEFAULT,
        quantiles=MAP_POOL_QUANTILES_DEFAULT,
        add_globals=MAP_POOL_ADD_GLOBALS_DEFAULT,
        max_polygons=max_polygons,
    )

    if vec is None or vec.size == 0:
        raise ValueError("Pooled vector is empty.")
    if np.any(~np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return vec, list(names or [])


def _safe_parquet_write(df: pd.DataFrame, out_path: Path) -> None:
    """Write parquet; if not available, fall back to CSV next to it."""
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        logging.warning("Parquet write failed (%s). Writing CSV fallback.", e)
        df.to_csv(out_path.with_suffix(".csv"), index=False)


def save_outputs(
    out_dir: Path,
    rows: List[Dict],
    E: np.ndarray,
    ids: List[str],
    feat_names: List[str],
    save_csv: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) embeddings (npz)
    np.savez_compressed(
        out_dir / MAP_EMBEDDINGS_NPZ_NAME,
        E=E.astype(np.float32, copy=False),
        ids=np.array(ids, dtype=object),
    )

    # 2) maps table
    df = pd.DataFrame(rows)
    _safe_parquet_write(df, out_dir / MAPS_PARQUET_NAME)

    if save_csv:
        df.to_csv(out_dir / MAPS_CSV_NAME, index=False)

    # 3) meta + feature names
    meta = {
        "dim": int(E.shape[1]),
        "count": int(E.shape[0]),
        "files": {
            "maps_embeddings_npz": MAP_EMBEDDINGS_NPZ_NAME,
            "maps_parquet": MAPS_PARQUET_NAME,
        },
        "vector_schema": "pooled stats over per-polygon features",
    }
    (out_dir / MAPS_META_JSON_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / MAPS_FEATURE_NAMES_JSON_NAME).write_text(json.dumps(feat_names, indent=2), encoding="utf-8")


# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute fixed-D map embeddings from GeoJSON tiles.")
    ap.add_argument("--norm", choices=["extent", "fixed"], default=POLY_NORM_MODE_DEFAULT)
    ap.add_argument("--norm-wh", type=str, default=None, help="WidthxHeight when --norm=fixed, e.g. '400x400'.")

    ap.add_argument("--root", type=str, default=str(DATA_DIR / MAP_EMBED_ROOT_DEFAULT))
    ap.add_argument("--pattern", type=str, default=str(MAP_EMBED_PATTERN_DEFAULT))
    ap.add_argument("--out_dir", type=str, default=str(DATA_DIR / MAP_EMBED_OUTDIR_DEFAULT))
    ap.add_argument("--save_csv", action="store_true", default=MAP_EMBED_SAVE_CSV_DEFAULT)
    ap.add_argument("-v", "--verbose", action="count", default=MAP_EMBED_VERBOSE_DEFAULT)

    args = ap.parse_args()
    setup_logging(int(args.verbose))

    root = _resolve(args.root)
    outdir = _resolve(args.out_dir)

    if not root.exists():
        logging.error("Root folder not found: %s", root)
        sys.exit(1)

    logging.info("Scanning %s (pattern=%s)…", root, args.pattern)

    pairs = list(find_geojsons(root, args.pattern))
    if not pairs:
        logging.error("No GeoJSONs embedded. Check --root and --pattern.")
        sys.exit(2)

    # First pass: dataset-wide max_polygons (for normalized poly_count)
    logging.info("First pass: counting polygons to normalize poly_count…")
    counts: Dict[str, int] = {}
    for map_id, path in pairs:
        try:
            counts[str(map_id)] = _count_valid_polygons(path)
        except Exception as e:
            logging.warning("Count failed for %s: %s", map_id, e)
            counts[str(map_id)] = 0

    max_polygons = max(max(counts.values()), 1)
    logging.info("Max polygons across dataset: %d", max_polygons)

    # Second pass: embed
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    rows: List[Dict] = []
    feat_names: List[str] | None = None
    first_dim: int | None = None
    failed = 0

    for map_id, path in pairs:
        try:
            vec, names = embed_one_map(path, max_polygons=max_polygons, norm=args.norm, norm_wh=args.norm_wh)

            if first_dim is None:
                first_dim = int(vec.shape[0])
                feat_names = list(names)
                if not feat_names or len(feat_names) != first_dim:
                    feat_names = [f"f{i:03d}" for i in range(first_dim)]
            elif vec.shape[0] != first_dim:
                failed += 1
                logging.error("Skipping %s: vector dim %d != expected %d", map_id, vec.shape[0], first_dim)
                continue

            ids.append(str(map_id))
            vecs.append(vec)

            extent = compute_extent_refs(path)

            rows.append(
                {
                    MAPS_ID_COL: str(map_id),
                    MAP_GEOJSON_COL: str(path),
                    MAP_N_POLYGONS_COL: int(counts.get(str(map_id), 0)),
                    **extent,
                }
            )

            logging.info("OK  map_id=%s  -> vector[%d]", map_id, vec.shape[0])
        except Exception as e:
            failed += 1
            logging.error("FAIL map_id=%s: %s", map_id, e)

    if not ids:
        logging.error("No GeoJSONs embedded. failed=%d", failed)
        sys.exit(2)

    E = np.vstack(vecs).astype(np.float32, copy=False)
    save_outputs(outdir, rows, E, ids, feat_names or [], bool(args.save_csv))
    logging.info("Saved %d vectors (failed=%d) to %s", len(ids), failed, outdir)


if __name__ == "__main__":
    main()
