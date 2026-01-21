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
    ap = argparse.ArgumentParser(description="Concatenate map & prompt embeddings via UserStudy.xlsx.")
    ap.add_argument("--user_study", type=str, default=str(DATA_DIR / "input" / "UserStudy.xlsx"))
    ap.add_argument("--sheet", type=str, default="Responses")
    ap.add_argument("--tile_id_col", type=str, default="tile_id")
    ap.add_argument("--complete_col", type=str, default="complete")
    ap.add_argument("--remove_col", type=str, default="remove")
    ap.add_argument("--text_col", type=str, default="cleaned_text")  # not used here but kept for clarity
    ap.add_argument("--map_npz",     type=str, default=str(DATA_DIR / "output" / "map_out" / "maps_embeddings.npz"))
    ap.add_argument("--prompt_npz",  type=str, default=str(DATA_DIR / "output" / "prompt_out"   / "prompts_embeddings.npz"))
    ap.add_argument("--out_dir",     type=str, default=str(DATA_DIR / "output" / "train_out"))
    ap.add_argument("--fail_on_missing", action="store_true")
    ap.add_argument("--drop_dupes",      action="store_true")
    ap.add_argument("-v", "--verbose",   action="count", default=1)
    ap.add_argument("--l2-prompt", action="store_true",help="L2-normalize prompt embeddings row-wise before concatenation.")
    ap.add_argument("--save-blocks", action="store_true",help="Also save X_map.npy, X_prompt.npy, map_ids.npy, prompt_ids.npy.")

    args = ap.parse_args()

    setup_logging(args.verbose)
    t0 = time.time()
    
    map_npz_path = _resolve(args.map_npz)
    prm_npz_path = _resolve(args.prompt_npz)
    out_dir      = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load pairs
    # --- build pairs from prompts.parquet (authoritative) ---
    prompts_pq = _resolve(Path(args.prompt_npz).parent / "prompts.parquet")
    if not prompts_pq.exists():
        logging.error("prompts.parquet not found next to prompt_npz: %s", prompts_pq)
        sys.exit(1)

    pairs = pd.read_parquet(prompts_pq)

    # Expect columns: prompt_id, text, tile_id
    if "prompt_id" not in pairs.columns:
        logging.error("prompts.parquet must contain 'prompt_id'")
        sys.exit(1)
    if "tile_id" not in pairs.columns:
        logging.error(
            "prompts.parquet must contain 'tile_id' "
            "(ensure prompt_embeddings.py writes tile_id)"
        )
        sys.exit(1)

    # keep label/meta columns if they exist in prompts.parquet
    keep_cols = ["tile_id", "prompt_id"]
    for c in ["operator", "intensity", "param_value"]:
        if c in pairs.columns:
            keep_cols.append(c)

    pairs = pairs[keep_cols].rename(columns={"tile_id": "map_id"}).copy()

    pairs["map_id"] = pairs["map_id"].astype(str).str.strip().str.zfill(4)
    pairs["prompt_id"] = pairs["prompt_id"].astype(str).str.strip()

    pairs = pairs.dropna(subset=["map_id", "prompt_id"])
    pairs = pairs[(pairs["map_id"] != "") & (pairs["prompt_id"] != "")]

    if args.drop_dupes:
        pairs = pairs.drop_duplicates(subset=["map_id", "prompt_id"])

    logging.info("Built %d pairs from prompts.parquet", len(pairs))

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
        im_opt = idx_map.get(mid)  # type: ignore
        ip_opt = idx_prm.get(pid)  # type: ignore
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


    # --- OPTIONAL: prompt L2 normalize (safety if upstream --l2 was not used)
    def _l2_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.sqrt((A * A).sum(axis=1, keepdims=True))
        return A / np.maximum(n, eps)

    # add CLI flag
    # ap.add_argument("--l2-prompt", action="store_true")
    if getattr(args, "l2_prompt", False):
        X_prm = _l2_rows(X_prm)

    # --- save concatenated and (optionally) separate blocks
    X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)
    np.save(out_dir / "X_concat.npy", X)

    if args.save_blocks:  # <-- wrap with the flag
        np.save(out_dir / "X_map.npy",    X_map)
        np.save(out_dir / "X_prompt.npy", X_prm)
        np.save(out_dir / "map_ids.npy",  np.asarray([map_ids[i] for i in sel_map_idx], dtype=object))
        np.save(out_dir / "prompt_ids.npy", np.asarray([prm_ids[i] for i in sel_prm_idx], dtype=object))


    # --- joined pairs parquet (unchanged)
    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    cols = [c for c in join_df.columns if c not in ("map_id", "prompt_id")]
    join_df = join_df[["map_id", "prompt_id", *cols]]
    # after join_df is built, before saving files:
    assert X.shape[0] == len(join_df), "Row count mismatch between X and join_df."

    join_df.to_parquet(out_dir / "train_pairs.parquet", index=False)

    # --- simple sanity checks
    if not np.isfinite(X).all():
        logging.warning("X contains non-finite values (NaN/Inf). Downstream imputer should handle this.")

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
        },
        # small preview helps debugging without dumping everything
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