# src/mapvec/concat/concat_embeddings.py
# Join pair-map + prompt embeddings using prompts.parquet and export:
#  - X_concat.npy  (row-wise [map_vec | prompt_vec])
#  - train_pairs.parquet (joined rows with original pair metadata)
#  - meta.json (shapes, sources)

from __future__ import annotations
from pathlib import Path
import argparse, sys, json, logging, time
from typing import Tuple, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

def load_npz(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as z:
        if "E" not in z or "ids" not in z:
            raise ValueError(f"{npz_path} must contain arrays 'E' and 'ids'")
        E = z["E"]
        ids = [str(x) for x in z["ids"].tolist()]
    if E.shape[0] != len(ids):
        raise ValueError(f"{npz_path}: rows {E.shape[0]} != ids {len(ids)}")
    return E, ids

def _normalize_prompt_id_series(s: pd.Series, *, pad_numeric: bool, width: int) -> pd.Series:
    """
    Keep prompt_id as string and optionally zero-pad purely numeric IDs.
    This prevents mismatches between Excel -> prompts.parquet -> embeddings NPZ.
    """
    out = s.astype(str).str.strip()
    out = out.mask(out.isin(["", "nan", "None"]), pd.NA)

    if pad_numeric:
        m = out.notna() & out.str.fullmatch(r"\d+")
        out.loc[m] = out.loc[m].str.zfill(int(width))

    return out

def main():
    ap = argparse.ArgumentParser(description="Concatenate map & prompt embeddings via prompts.parquet.")
    ap.add_argument("--map_npz",     type=str, default=str(DATA_DIR / "output" / "map_out" / "maps_embeddings.npz"))
    ap.add_argument("--prompt_npz",  type=str, default=str(DATA_DIR / "output" / "prompt_out" / "prompts_embeddings.npz"))
    ap.add_argument("--out_dir",     type=str, default=str(DATA_DIR / "output" / "train_out"))
    ap.add_argument("--fail_on_missing", action="store_true")
    ap.add_argument("--drop_dupes",      action="store_true")
    ap.add_argument("-v", "--verbose",   action="count", default=1)
    ap.add_argument("--l2-prompt", action="store_true",
                    help="L2-normalize prompt embeddings row-wise before concatenation.")
    ap.add_argument("--save-blocks", action="store_true",
                    help="Also save X_map.npy, X_prompt.npy, map_ids.npy, prompt_ids.npy.")
    ap.add_argument("--maps_parquet", type=str,
                    default=str(DATA_DIR / "output" / "map_out" / "maps.parquet"),
                    help="maps.parquet containing extent_* columns (from map_embeddings.py).")

    # ✅ NEW: prompt_id normalization controls
    ap.add_argument("--pad_numeric_prompt_ids", action="store_true",
                    help="If prompt_id looks numeric, zero-pad it to --prompt_id_width.")
    ap.add_argument("--prompt_id_width", type=int, default=8,
                    help="Width for zero-padding numeric prompt_id values (default 8).")

    args = ap.parse_args()

    setup_logging(args.verbose)
    t0 = time.time()

    map_npz_path = _resolve(args.map_npz)
    prm_npz_path = _resolve(args.prompt_npz)
    out_dir      = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build pairs from prompts.parquet (authoritative) ---
    prompts_pq = _resolve(Path(args.prompt_npz).parent / "prompts.parquet")
    if not prompts_pq.exists():
        logging.error("prompts.parquet not found next to prompt_npz: %s", prompts_pq)
        sys.exit(1)

    pairs = pd.read_parquet(prompts_pq)

    if "prompt_id" not in pairs.columns:
        logging.error("prompts.parquet must contain 'prompt_id'")
        sys.exit(1)
    if "tile_id" not in pairs.columns:
        logging.error("prompts.parquet must contain 'tile_id' (ensure prompt_embeddings writes tile_id)")
        sys.exit(1)

    keep_cols = ["tile_id", "prompt_id", "text"]
    for c in ["operator", "intensity", "param_value"]:
        if c in pairs.columns:
            keep_cols.append(c)

    pairs = pairs[keep_cols].rename(columns={"tile_id": "map_id"}).copy()

    pairs["map_id"] = pairs["map_id"].astype(str).str.strip().str.zfill(4)
    pairs["prompt_id"] = _normalize_prompt_id_series(
        pairs["prompt_id"],
        pad_numeric=bool(args.pad_numeric_prompt_ids),
        width=int(args.prompt_id_width),
    )

    pairs = pairs.dropna(subset=["map_id", "prompt_id"])
    pairs = pairs[(pairs["map_id"] != "") & (pairs["prompt_id"] != "")]

    if args.drop_dupes:
        pairs = pairs.drop_duplicates(subset=["map_id", "prompt_id"])

    logging.info("Built %d pairs from prompts.parquet", len(pairs))

    # -------------------------------
    # Merge per-map extent refs (ONCE)
    # -------------------------------
    maps_pq = _resolve(args.maps_parquet)
    if not maps_pq.exists():
        logging.error("maps.parquet not found: %s (run map_embeddings.py first)", maps_pq)
        sys.exit(1)

    maps_df = pd.read_parquet(maps_pq)

    needed = ["map_id", "extent_diag_m", "extent_area_m2"]
    missing_cols = [c for c in needed if c not in maps_df.columns]
    if missing_cols:
        logging.error("maps.parquet is missing columns %s. Re-run map_embeddings.py with extent saving.", missing_cols)
        sys.exit(1)

    extent_keep = [
        "map_id",
        "extent_diag_m",
        "extent_area_m2",
        "extent_width_m",
        "extent_height_m",
        "extent_minx",
        "extent_miny",
        "extent_maxx",
        "extent_maxy",
    ]
    extent_keep = [c for c in extent_keep if c in maps_df.columns]

    maps_df = maps_df[extent_keep].copy()
    maps_df["map_id"] = maps_df["map_id"].astype(str).str.strip().str.zfill(4)

    before = len(pairs)
    pairs = pairs.merge(maps_df, on="map_id", how="left")

    miss_extent = int(pairs["extent_diag_m"].isna().sum())
    if miss_extent:
        logging.warning("⚠️ %d rows missing extent_diag_m after merge (map_id not found in maps.parquet).", miss_extent)
        if args.fail_on_missing:
            logging.error("Failing because --fail_on_missing is set.")
            sys.exit(2)

    logging.info("Merged extent refs from %s into pairs (%d -> %d rows).", maps_pq, before, len(pairs))

    # --- load embeddings
    E_map, map_ids = load_npz(map_npz_path)
    E_prm, prm_ids = load_npz(prm_npz_path)
    logging.info("Map  embeddings: %s from %s", E_map.shape, map_npz_path)
    logging.info("Prompt embeddings: %s from %s", E_prm.shape, prm_npz_path)

    idx_map: dict[str, int] = {k: i for i, k in enumerate(map_ids)}
    idx_prm: dict[str, int] = {k: i for i, k in enumerate(prm_ids)}

    # --- match & build X
    chosen_rows: List[int] = []
    im_list: List[int] = []
    ip_list: List[int] = []
    missing = 0

    pairs = pairs.reset_index(drop=True)

    for i, row in enumerate(pairs.itertuples(index=False), start=0):
        mid = str(row.map_id).strip().zfill(4)
        pid = str(row.prompt_id).strip().zfill(4)
        im_opt = idx_map.get(mid)
        ip_opt = idx_prm.get(pid)
        if im_opt is None or ip_opt is None:
            missing += 1
            if args.fail_on_missing:
                logging.error("Missing ID (map_id=%s, prompt_id=%s)", mid, pid)
                sys.exit(2)
            continue
        chosen_rows.append(i)
        im_list.append(im_opt)
        ip_list.append(ip_opt)

    if not im_list:
        logging.error("No valid pairs after ID matching.")
        sys.exit(2)
    if missing:
        logging.warning("Skipped %d pairs with missing IDs (use --fail_on_missing to stop).", missing)

    sel_map_idx = np.asarray(im_list, dtype=int)
    sel_prm_idx = np.asarray(ip_list, dtype=int)
    X_map = E_map[sel_map_idx].astype(np.float32, copy=False)
    X_prm = E_prm[sel_prm_idx].astype(np.float32, copy=False)

    if X_map.shape[1] == 0 or X_prm.shape[1] == 0:
        logging.error("Zero-dimension map or prompt block. map_dim=%d, prompt_dim=%d",
                      X_map.shape[1], X_prm.shape[1])
        sys.exit(3)

    def _l2_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.sqrt((A * A).sum(axis=1, keepdims=True))
        return A / np.maximum(n, eps)

    if getattr(args, "l2_prompt", False):
        X_prm = _l2_rows(X_prm)

    X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)
    np.save(out_dir / "X_concat.npy", X)

    if args.save_blocks:
        np.save(out_dir / "X_map.npy",    X_map)
        np.save(out_dir / "X_prompt.npy", X_prm)
        np.save(out_dir / "map_ids.npy",  np.asarray([map_ids[i] for i in sel_map_idx], dtype=object))
        np.save(out_dir / "prompt_ids.npy", np.asarray([prm_ids[i] for i in sel_prm_idx], dtype=object))

    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    cols = [c for c in join_df.columns if c not in ("map_id", "prompt_id")]
    join_df = join_df[["map_id", "prompt_id", *cols]]

    assert X.shape[0] == len(join_df), "Row count mismatch between X and join_df."
    join_df.to_parquet(out_dir / "train_pairs.parquet", index=False)

    outputs = {
        "X_concat_npy": "X_concat.npy",
        "train_pairs_parquet": "train_pairs.parquet",
    }
    if args.save_blocks:
        outputs.update({
            "X_map_npy": "X_map.npy",
            "X_prompt_npy": "X_prompt.npy",
            "map_ids_npy": "map_ids.npy",
            "prompt_ids_npy": "prompt_ids.npy",
        })

    meta = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "map_dim": int(E_map.shape[1]),
        "prompt_dim": int(E_prm.shape[1]),
        "rows": int(X.shape[0]),
        "skipped_pairs": int(missing),
        "sources": {
            "prompts_parquet": str(prompts_pq),
            "map_npz": str(map_npz_path),
            "prompt_npz": str(prm_npz_path),
        },
        "outputs": outputs,
        "options": {
            "l2_prompt": bool(args.l2_prompt),
            "drop_dupes": bool(args.drop_dupes),
            "fail_on_missing": bool(args.fail_on_missing),
            "save_blocks": bool(args.save_blocks),
            "pad_numeric_prompt_ids": bool(args.pad_numeric_prompt_ids),
            "prompt_id_width": int(args.prompt_id_width),
        },
        "preview_ids": {
            "map_ids":    [map_ids[i] for i in sel_map_idx[:10]],
            "prompt_ids": [prm_ids[i] for i in sel_prm_idx[:10]],
        },
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("X shape = %s  (map_dim=%d, prompt_dim=%d)", X.shape, E_map.shape[1], E_prm.shape[1])
    logging.info("Saved to %s in %.2fs", out_dir, time.time() - t0)

if __name__ == "__main__":
    main()
