# src/mapvec/maps/map_embeddings.py 
# Compute fixed-length map vectors from GeoJSONs and save artifacts.
# Drop-in compatible with the original script (same CLI, defaults, outputs).

from __future__ import annotations

import sys, json, argparse, logging
from pathlib import Path
from typing import Iterator, Tuple, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon

# ↳ local modules (unchanged import paths)
from src.mapvec.features.polygon_features import embed_polygons_handcrafted
from src.mapvec.features.map_pooling import pool_map_embedding

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in within")
warnings.filterwarnings("ignore", message="invalid value encountered in contains")
warnings.filterwarnings("ignore", message="invalid value encountered in buffer")

# ----------------------- paths & logging -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/CODES
DATA_DIR     = (PROJECT_ROOT / "data").resolve()

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    # prefer CWD; if missing, try under data/
    cand = (Path.cwd() / p).resolve()
    return cand if cand.exists() or cand.parent.exists() else (DATA_DIR / p).resolve()

def setup_logging(verbosity: int = 1):
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
    Example subfolder: data/samples/pairs/0001/0001_input.geojson
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
    # gpd.read_file handles many formats (GeoJSON, GPKG, Shapefile, etc.)
    gdf = gpd.read_file(gj_path) 
    # If CRS missing, assume WGS84 (typical for GeoJSON) so we can safely project later
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty.")
    if "geometry" not in gdf.columns:
        raise ValueError("No 'geometry' column found.")
    # filter invalid/empty/null geometry rows
    gdf = gdf[gdf.geometry.notnull()].copy()
    if "is_empty" in dir(gdf.geometry):  # geopandas>=0.10
        gdf = gdf[~gdf.geometry.is_empty].copy()
    else:
        gdf = gdf[[not geom.is_empty for geom in gdf.geometry]].copy()
    if gdf.empty:
        raise ValueError("All geometries were empty/invalid.")
    return gdf

def compute_extent_refs(gj_path: Path) -> Dict[str, float]:
    """
    Compute per-map extent metrics.
    Returns width/height/diag/area computed from GeoDataFrame total_bounds.

    IMPORTANT:
    - If CRS is geographic (lat/lon), we project to EPSG:3857 (meters).
    - If CRS is missing or not geographic, we do NOT force a CRS; we just compute bounds.
      (In your dataset, you said values are in meters already → this is fine.)
    """
    gdf = _read_geo(gj_path)

    # If CRS is geographic, project to meters
    try:
        if gdf.crs and getattr(gdf.crs, "is_geographic", False):
            gdf = gdf.to_crs(3857)
    except Exception:
        pass

    minx, miny, maxx, maxy = map(float, gdf.total_bounds)

    w = maxx - minx
    h = maxy - miny

    # guard against degenerate
    if not np.isfinite(w) or w < 0:
        w = float("nan")
    if not np.isfinite(h) or h < 0:
        h = float("nan")

    diag = float(np.sqrt(w * w + h * h)) if np.isfinite(w) and np.isfinite(h) else float("nan")
    area = float(w * h) if np.isfinite(w) and np.isfinite(h) else float("nan")

    return {
        "extent_minx": float(minx),
        "extent_miny": float(miny),
        "extent_maxx": float(maxx),
        "extent_maxy": float(maxy),
        "extent_width_m": float(w),
        "extent_height_m": float(h),
        "extent_diag_m": float(diag),
        "extent_area_m2": float(area),
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
        # ignore Points/Lines, etc.
        return


def _fix_polygon(poly: Polygon) -> Polygon | None:
    """Try to repair/clean a polygon; return None if still bad."""
    try:
        # buffer(0) repairs many self-intersections
        p = poly.buffer(0)
        if not p.is_valid or p.is_empty:
            return None
        # Sometimes buffer(0) returns MultiPolygon; keep largest area part
        if isinstance(p, MultiPolygon):
            parts = [q for q in p.geoms if q.is_valid and not q.is_empty]
            if not parts:
                return None
            p = max(parts, key=lambda q: q.area)
        return p if isinstance(p, Polygon) else None
    except Exception:
        return None

def _count_valid_polygons(gj_path: Path) -> int:
    """Read, clean, and count valid polygon parts in a single map file."""
    gdf = _read_geo(gj_path)
    try:
        if gdf.crs and getattr(gdf.crs, "is_geographic", False):
            gdf = gdf.to_crs(3857)
    except Exception:
        pass
    geoms = _flatten_and_clean_to_polygons(gdf)
    return len(geoms)

def _flatten_and_clean_to_polygons(gdf, area_eps=1e-12):
    """Return a list of valid Polygons suitable for feature extraction."""
    polys = []
    for geom in gdf.geometry:
        for p in _iter_polygons(geom):
            p2 = _fix_polygon(p)
            if p2 is None:
                continue
            # drop tiny slivers / NaN-ish geometries
            if not np.isfinite(p2.area) or p2.area <= area_eps:
                continue
            polys.append(p2)
    return polys

def embed_one_map(gj_path: Path, max_polygons: int | float | None = None,norm: str = "extent", norm_wh: str | None = None
                  ) -> Tuple[np.ndarray, List[str]]:
    gdf = _read_geo(gj_path)

    try:
        if gdf.crs and getattr(gdf.crs, "is_geographic", False):
            gdf = gdf.to_crs(3857)  # Web Mercator: meters
    except Exception:
        pass

    geoms = _flatten_and_clean_to_polygons(gdf)
    if len(geoms) == 0:
        raise ValueError("No valid polygon parts after flatten/clean (all invalid/empty?).")

    # Parse norm_wh when provided (e.g., "400x400")
    _fixed = None
    if norm == "fixed":
        if not norm_wh or "x" not in norm_wh.lower():
            raise ValueError("Use --norm-wh like '400x400' with --norm=fixed")
        w_str, h_str = norm_wh.lower().split("x", 1)
        _fixed = (float(w_str), float(h_str))

    df_polys = embed_polygons_handcrafted(geoms, norm_mode=norm, fixed_wh=_fixed)    

    vec, names = pool_map_embedding(
        df_polys,
        exclude=("id",),
        stats=("mean", "std", "min", "max"),
        quantiles=(0.25, 0.50, 0.75),
        add_globals=True,
        max_polygons=max_polygons,   # <-- NEW
    )

    if vec is None or vec.size == 0:
        raise ValueError("Pooled vector is empty.")
    if np.any(~np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return vec, list(names or [])

def maybe_read_tile_meta(sample_dir: Path) -> Dict:
    """
    Optional: collect metadata from data/samples/metadata/meta.csv using the numeric sample_id (=folder name).
    Returns {} if not found or on failure.
    """
    try:
        meta_csv = DATA_DIR / "samples" / "metadata" / "meta.csv"
        if meta_csv.exists():
            dfm = pd.read_csv(meta_csv)
            sid = int(sample_dir.name)
            row = dfm.loc[dfm["sample_id"] == sid]
            if not row.empty:
                return row.iloc[0].to_dict()
    except Exception as e:
        logging.debug("Meta read failed for %s: %s", sample_dir, e)
    return {}

def _safe_parquet_write(df: pd.DataFrame, out_path: Path) -> None:
    """
    Try writing parquet; if pyarrow/fastparquet is missing, fall back to CSV next to it.
    This preserves usability without adding hard deps.
    """
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
    try:
        np.savez_compressed(out_dir / "maps_embeddings.npz", E=E.astype(np.float32, copy=False), ids=np.array(ids, dtype=object))
    except Exception as e:
        logging.error("Failed to save maps_embeddings.npz: %s", e)
        raise

    # 2) table of maps (light metadata)
    df = pd.DataFrame(rows)
    _safe_parquet_write(df, out_dir / "maps.parquet")
    if save_csv:
        try:
            df.to_csv(out_dir / "maps.csv", index=False)
        except Exception as e:
            logging.warning("Failed to save maps.csv: %s", e)

    # 3) meta + feature names
    meta = {
        "dim": int(E.shape[1]),
        "count": int(E.shape[0]),
        "files": {
            "maps_embeddings_npz": "maps_embeddings.npz",
            "maps_parquet": "maps.parquet",  # may have CSV fallback
        },
        "vector_schema": "pooled stats over per-polygon features",
    }
    try:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "feature_names.json").write_text(json.dumps(feat_names, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning("Failed to save JSON sidecars: %s", e)

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute fixed-D map embeddings from GeoJSON tiles.")
    ap.add_argument("--norm", choices=["extent","fixed"], default="extent",
                    help="Normalization scale: per-tile extent (default) or fixed width/height.")
    ap.add_argument("--norm-wh", type=str, default=None,
                    help="WidthxHeight in meters when --norm=fixed, e.g. '400x400'.")

    ap.add_argument("--root", type=str, default=str(DATA_DIR / "samples" / "pairs"),
                    help="Root folder with <map_id>/ subfolders (default: data/samples/pairs).")
    ap.add_argument("--pattern", type=str, default="*_input.geojson",
                    help="Glob pattern per <map_id> dir (e.g. '*_generalized.geojson').")
    ap.add_argument("--out_dir", type=str, default=str(DATA_DIR / "map_out"),
                    help="Output directory (default: data/map_out).")
    ap.add_argument("--save_csv", action="store_true", help="Also save maps.csv.")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    args = ap.parse_args()

    setup_logging(args.verbose)

    root   = _resolve(args.root)
    outdir = _resolve(args.out_dir)

    if not root.exists():
        logging.error("Root folder not found: %s", root)
        sys.exit(1)

    logging.info("Scanning %s (pattern=%s)…", root, args.pattern)

    # Collect all candidates once so we can do two passes
    pairs = list(find_geojsons(root, args.pattern))
    if not pairs:
        logging.error("No GeoJSONs embedded. Check --root and --pattern.")
        sys.exit(2)
    # --- Optional: filter map_ids using UserStudy.xlsx ---
    try:
        from src.config import PATHS  # uses your config.py
        dfu = pd.read_excel(PATHS.USER_STUDY_XLSX, sheet_name=PATHS.RESPONSES_SHEET)

        # Keep only complete=True and remove=False
        dfu[PATHS.COMPLETE_COL] = dfu[PATHS.COMPLETE_COL].astype(bool)
        dfu[PATHS.REMOVE_COL]   = dfu[PATHS.REMOVE_COL].astype(bool)

        mask = pd.Series(True, index=dfu.index)
        if PATHS.ONLY_COMPLETE:
            mask &= (dfu[PATHS.COMPLETE_COL] == True)
        if PATHS.EXCLUDE_REMOVED:
            mask &= (dfu[PATHS.REMOVE_COL] == False)
        dfu = dfu[mask].copy()

        # Allowed tile_ids (map_ids)
        tile_raw = dfu[PATHS.TILE_ID_COL]
        tile_num = pd.to_numeric(tile_raw, errors="coerce")

        if tile_num.notna().all():
            # ZERO-PAD to match folder names like 0001
            allowed_tile_ids = set(tile_num.astype(int).astype(str).str.zfill(4))
        else:
            allowed_tile_ids = set(
                tile_raw.astype(str).str.strip().str.zfill(4)
            )

        before = len(pairs)
        pairs = [(map_id, path) for (map_id, path) in pairs if str(map_id).strip() in allowed_tile_ids]
        after = len(pairs)

        logging.info("Filtered maps by UserStudy.xlsx: %d -> %d", before, after)

        if after == 0:
            logging.error("After Excel filtering, there are no maps left to embed.")
            sys.exit(2)

    except Exception as e:
        logging.warning("Excel-based filtering skipped (reason: %s). Embedding all maps in root.", e)


    # ---------- First pass: compute dataset-wide max_polygons ----------
    logging.info("First pass: counting polygons to normalize poly_count…")
    counts: Dict[str, int] = {}
    for map_id, path in pairs:
        try:
            counts[map_id] = _count_valid_polygons(path)
        except Exception as e:
            logging.warning("Count failed for %s: %s", map_id, e)
            counts[map_id] = 0
    max_polygons = max(max(counts.values()), 1)
    logging.info("Max polygons across dataset: %d", max_polygons)

    # ---------- Second pass: embed with normalized poly_count ----------
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    rows: List[Dict] = []
    feat_names: List[str] | None = None

    total = 0
    failed = 0
    first_dim: int | None = None

    for map_id, path in pairs:
        total += 1
        try:
            vec, names = embed_one_map(path, max_polygons=max_polygons,norm=args.norm, norm_wh=args.norm_wh)
            # ensure consistent dimensionality across tiles
            if first_dim is None:
                first_dim = int(vec.shape[0])
                feat_names = list(names)
                if not feat_names or len(feat_names) != first_dim:
                    logging.debug("Feature names length mismatch; using index-derived names.")
                    feat_names = [f"f{i:03d}" for i in range(first_dim)]
            elif vec.shape[0] != first_dim:
                failed += 1
                logging.error("Skipping %s: vector dim %d != expected %d", map_id, vec.shape[0], first_dim)
                continue

            ids.append(map_id)
            vecs.append(vec)

            meta = maybe_read_tile_meta(path.parent)

            extent = compute_extent_refs(path)

            rows.append({
                "map_id": map_id,
                "geojson": str(path),
                "n_polygons": int(counts.get(map_id, 0)),

                # NEW: save extent metrics into maps.parquet
                **extent,

                **{k: meta.get(k) for k in (
                    "operator","intensity","param_value","param_unit",
                    "input_png","target_png","input_geojson","target_geojson",
                    "n_input_polys","n_target_polys","is_target_empty"
                ) if k in meta}
            })

            logging.info("OK  map_id=%s  -> vector[%d]", map_id, vec.shape[0])
        except Exception as e:
            failed += 1
            logging.error("FAIL map_id=%s: %s", map_id, e)

    if not ids:
        logging.error("No GeoJSONs embedded. (processed=%d, failed=%d)", total, failed)
        sys.exit(2)

    # Stack to (M, D) and persist artifacts
    try:
        E = np.vstack(vecs).astype(np.float32, copy=False)
    except Exception as e:
        logging.error("Failed to stack embeddings: %s", e)
        sys.exit(3)

    save_outputs(outdir, rows, E, ids, feat_names or [], args.save_csv)
    logging.info("Saved %d vectors (failed=%d) to %s", len(ids), failed, outdir)

if __name__ == "__main__":
    main()