# src/mapvec/maps/pair_map_embeddings.py
# Build pair-wise map embeddings: concat[input, generalized, delta, log-ratio].
# Mirrors the latest single-map script's robustness & outputs.

from __future__ import annotations
import sys, json, argparse, logging
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Iterable, Union, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from src.mapvec.features.polygon_features import embed_polygons_handcrafted
from src.mapvec.features.map_pooling import pool_map_embedding

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in within")
warnings.filterwarnings("ignore", message="invalid value encountered in contains")
warnings.filterwarnings("ignore", message="invalid value encountered in buffer")

# ----------------------- paths & logging -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]    # …/CODES
DATA_DIR     = (PROJECT_ROOT / "data").resolve()

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    cand = (Path.cwd() / p).resolve()
    return cand if cand.exists() or cand.parent.exists() else (DATA_DIR / p).resolve()

def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    logging.debug("PROJECT_ROOT=%s", PROJECT_ROOT)
    logging.debug("DATA_DIR=%s", DATA_DIR)

# ----------------------- file discovery -----------------------
def find_pairs(root: Path, in_pat: str, gen_pat: str) -> Iterator[Tuple[str, Path, Path]]:
    """
    Yield (map_id, input_path, generalized_path) for every subfolder in `root`
    that contains BOTH patterns.
    """
    if not root.exists():
        return
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        ins  = list(sub.glob(in_pat))
        gens = list(sub.glob(gen_pat))
        if not ins or not gens:
            logging.debug("Missing input/gen in %s (in=%d, gen=%d)", sub, len(ins), len(gens))
            continue
        yield sub.name, ins[0], gens[0]

# ----------------------- Geo helpers (copied from your latest script) -------
def _read_geo(gj_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gj_path)
    if gdf.empty:                 raise ValueError("GeoDataFrame is empty.")
    if "geometry" not in gdf:     raise ValueError("No 'geometry' column found.")
    gdf = gdf[gdf.geometry.notnull()].copy()
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        gdf = gdf[[not geom.is_empty for geom in gdf.geometry]].copy()
    if gdf.empty:                 raise ValueError("All geometries were empty/invalid.")
    return gdf

def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    t = geom.geom_type
    if t == "Polygon":
        yield geom
    elif t == "MultiPolygon":
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    elif t == "GeometryCollection":
        for g in geom.geoms:
            yield from _iter_polygons(g)
    else:
        return

def _fix_polygon(poly: Polygon) -> Optional[Polygon]:
    try:
        p = poly.buffer(0)
        if p.is_empty or not p.is_valid:
            return None
        if isinstance(p, MultiPolygon):
            parts = [q for q in p.geoms if q.is_valid and not q.is_empty]
            if not parts:
                return None
            p = max(parts, key=lambda q: q.area)
        return p if isinstance(p, Polygon) else None
    except Exception:
        return None

def _flatten_and_clean_to_polygons(gdf, area_eps=1e-12) -> List[Polygon]:
    polys: List[Polygon] = []
    for geom in gdf.geometry:
        for p in _iter_polygons(geom):
            p2 = _fix_polygon(p)
            if p2 is None:                         continue
            if not np.isfinite(p2.area):           continue
            if p2.area <= area_eps:                continue
            polys.append(p2)
    return polys

def _embed_one(gj_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Single-map embedding using your robust pipeline."""
    gdf = _read_geo(gj_path)
    try:
        if gdf.crs and gdf.crs.is_geographic:
            gdf = gdf.to_crs(3857)  # metric
    except Exception:
        pass
    geoms = _flatten_and_clean_to_polygons(gdf)
    if len(geoms) == 0:
        raise ValueError("No valid polygon parts after clean/flatten.")

    df_polys = embed_polygons_handcrafted(geoms, normalize=False)
    if isinstance(df_polys, tuple):
        df_polys = df_polys[0]
    if df_polys is None or df_polys.empty:
        raise ValueError("Feature table empty after extraction.")

    vec, names = pool_map_embedding(
        df_polys,
        exclude=("id",),
        stats=("mean", "std", "min", "max"),
        quantiles=(0.25, 0.50, 0.75),
        add_globals=True,
    )
    if vec is None or vec.size == 0:
        raise ValueError("Pooled vector is empty.")
    if np.any(~np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vec.astype(np.float32, copy=False), list(names or [])

# ----------------------- save helpers -----------------------
def _safe_parquet_write(df: pd.DataFrame, out_path: Path) -> None:
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        logging.warning("Parquet write failed (%s). Writing CSV fallback.", e)
        df.to_csv(out_path.with_suffix(".csv"), index=False)

def _build_feature_names(base: List[str]) -> List[str]:
    # concat[input, generalized, delta, log-ratio]
    suff = ["__in", "__gen", "__delta", "__logratio"]
    out: List[str] = []
    for s in suff:
        out.extend([f"{n}{s}" for n in base])
    return out

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Pair-wise map embeddings: concat[input, generalized, delta, log-ratio].")
    ap.add_argument("--root", type=str, default=str(DATA_DIR / "samples" / "pairs"),
                    help="Root folder with <map_id>/ subfolders (default: data/samples/pairs).")
    ap.add_argument("--input_pattern", type=str, default="*_input.geojson",
                    help="Glob for input map inside each <map_id> dir.")
    ap.add_argument("--gen_pattern", type=str, default="*_generalized.geojson",
                    help="Glob for generalized/target map inside each <map_id> dir.")
    ap.add_argument("--out_dir", type=str, default=str(DATA_DIR / "pair_map_out"),
                    help="Output directory (default: data/pair_map_out).")
    ap.add_argument("--save_csv", action="store_true", help="Also save pairs.csv.")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    args = ap.parse_args()

    setup_logging(args.verbose)

    root   = _resolve(args.root)
    outdir = _resolve(args.out_dir)
    if not root.exists():
        logging.error("Root folder not found: %s", root)
        sys.exit(1)

    ids: List[str] = []
    rows: List[Dict] = []
    vecs: List[np.ndarray] = []
    feature_names_full: Optional[List[str]] = None
    per_map_dim: Optional[int] = None

    eps = 1e-9

    logging.info("Scanning %s (in=%s, gen=%s)…", root, args.input_pattern, args.gen_pattern)
    total = 0
    failed = 0

    for map_id, in_path, gen_path in find_pairs(root, args.input_pattern, args.gen_pattern):
        total += 1
        try:
            x, names = _embed_one(in_path)       # input
            g, names2 = _embed_one(gen_path)     # generalized

            if x.shape[0] != g.shape[0]:
                failed += 1
                logging.error("Dim mismatch for %s: input %d vs gen %d", map_id, x.shape[0], g.shape[0])
                continue

            if per_map_dim is None:
                per_map_dim = int(x.shape[0])
                # feature names only once
                base_names = names if names and len(names) == per_map_dim else [f"f{i:03d}" for i in range(per_map_dim)]
                feature_names_full = _build_feature_names(base_names)

            # delta & log-ratio
            delta = (g - x).astype(np.float32, copy=False)
            logratio = np.log((g + eps) / (x + eps)).astype(np.float32, copy=False)

            vec = np.concatenate([x, g, delta, logratio], dtype=np.float32)
            vecs.append(vec)
            ids.append(map_id)

            rows.append({
                "map_id": map_id,
                "input_geojson": str(in_path),
                "generalized_geojson": str(gen_path),
            })
            logging.info("OK  map_id=%s  -> pair_vec[%d] (per_map_dim=%d)", map_id, vec.shape[0], per_map_dim)
        except Exception as e:
            failed += 1
            logging.error("FAIL %s: %s", map_id, e)

    if not ids:
        logging.error("No pairs embedded. (processed=%d, failed=%d)", total, failed)
        sys.exit(2)

    # stack & save
    E = np.vstack(vecs).astype(np.float32, copy=False)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) embeddings
    np.savez_compressed(outdir / "embeddings.npz", E=E, ids=np.array(ids, dtype=object))

    # 2) table
    df = pd.DataFrame(rows)
    _safe_parquet_write(df, outdir / "pairs.parquet")
    if args.save_csv:
        try:
            df.to_csv(outdir / "pairs.csv", index=False)
        except Exception as e:
            logging.warning("Failed to save pairs.csv: %s", e)

    # 3) names + meta
    (outdir / "feature_names.json").write_text(
        json.dumps(feature_names_full or [], indent=2), encoding="utf-8"
    )

    meta = {
        "dim": int(E.shape[1]),
        "count": int(E.shape[0]),
        "per_map_dim": int(per_map_dim or 0),
        "schema": "concat[input, generalized, delta, log-ratio]",
        "sections": ["in", "gen", "delta", "logratio"],
        "files": {
            "embeddings_npz": "embeddings.npz",
            "pairs_parquet": "pairs.parquet"
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logging.info("Saved %d pair vectors (failed=%d) to %s", len(ids), failed, outdir)

if __name__ == "__main__":
    main()