# src/mapvec/concat/concat_embeddings.py
# Join pair-map + prompt embeddings using pairs.csv and export:
#  - X_concat.npy  (row-wise [map_vec | prompt_vec])
#  - train_pairs.parquet (joined rows with original pair metadata)
#  - meta.json (shapes, sources)

from __future__ import annotations
from pathlib import Path
import argparse, sys, json, logging, time
from typing import Tuple, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # …/CODES
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

def main():
    ap = argparse.ArgumentParser(description="Concatenate pair-map & prompt embeddings via pairs.csv.")
    ap.add_argument("--pairs",       type=str, default=str(DATA_DIR / "input"  / "pairs.csv"))
    ap.add_argument("--map_npz",     type=str, default=str(DATA_DIR / "output" / "map_out" / "maps_embeddings.npz"))
    ap.add_argument("--prompt_npz",  type=str, default=str(DATA_DIR / "output" / "prompt_out"   / "prompts_embeddings.npz"))
    ap.add_argument("--out_dir",     type=str, default=str(DATA_DIR / "output" / "train_out"))
    ap.add_argument("--fail_on_missing", action="store_true")
    ap.add_argument("--drop_dupes",      action="store_true")
    ap.add_argument("-v", "--verbose",   action="count", default=1)
    args = ap.parse_args()

    setup_logging(args.verbose)
    t0 = time.time()

    pairs_path   = _resolve(args.pairs)
    map_npz_path = _resolve(args.map_npz)
    prm_npz_path = _resolve(args.prompt_npz)
    out_dir      = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load pairs
    if not pairs_path.exists():
        logging.error("pairs.csv not found: %s", pairs_path)
        sys.exit(1)
    pairs = pd.read_csv(pairs_path, dtype=str)

    if not {"map_id", "prompt_id"}.issubset(pairs.columns):
        logging.error("pairs.csv must have columns: map_id,prompt_id")
        sys.exit(1)

    pairs["map_id"] = pairs["map_id"].astype(str).str.strip()
    pairs["prompt_id"] = pairs["prompt_id"].astype(str).str.strip()
    before = len(pairs)
    pairs = pairs.dropna(subset=["map_id", "prompt_id"])
    pairs = pairs[(pairs["map_id"] != "") & (pairs["prompt_id"] != "")]
    if args.drop_dupes:
        pairs = pairs.drop_duplicates(subset=["map_id", "prompt_id"])
    after = len(pairs)
    if after == 0:
        logging.error("pairs.csv has no valid rows after cleaning.")
        sys.exit(1)
    if after < before:
        logging.warning("Dropped %d rows (empty/duplicates).", before - after)

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

    pairs = pairs.reset_index(drop=True)  # ensures loop index i is int-like

    for i, row in enumerate(pairs.itertuples(index=False), start=0):
        mid = row.map_id
        pid = row.prompt_id
        im_opt = idx_map.get(mid) # type: ignore
        ip_opt = idx_prm.get(pid) # type: ignore
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

    # row-wise concat (dimension-agnostic)
    X_map = E_map[np.asarray(im_list, dtype=int)]
    X_prm = E_prm[np.asarray(ip_list, dtype=int)]
    X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)

    # --- save artifacts
    np.save(out_dir / "X_concat.npy", X)

    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    cols = [c for c in join_df.columns if c not in ("map_id", "prompt_id")]
    join_df = join_df[["map_id", "prompt_id", *cols]]
    join_df.to_parquet(out_dir / "train_pairs.parquet", index=False)

    meta = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "map_dim": int(E_map.shape[1]),
        "prompt_dim": int(E_prm.shape[1]),
        "rows": int(X.shape[0]),
        "skipped_pairs": int(missing),
        "sources": {
            "pairs_csv": str(pairs_path),
            "map_npz": str(map_npz_path),
            "prompt_npz": str(prm_npz_path),
        },
        "outputs": {
            "X_concat_npy": "X_concat.npy",
            "train_pairs_parquet": "train_pairs.parquet",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("X shape = %s  (map_dim=%d, prompt_dim=%d)", X.shape, E_map.shape[1], E_prm.shape[1])
    logging.info("Saved to %s in %.2fs", out_dir, time.time() - t0)

if __name__ == "__main__":
    main()