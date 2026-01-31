# src/mapvec/concat/concat_embeddings.py
"""
Join map + prompt embeddings using prompts.parquet and export:
  - X_concat.npy  (row-wise [map_vec | prompt_vec])
  - train_pairs.parquet (joined rows with original pair metadata)
  - meta.json (shapes, sources)
"""

from __future__ import annotations

from pathlib import Path
import argparse, sys, json, logging, time
from typing import Tuple, List

import numpy as np
import pandas as pd

from src.constants import (
    # repo/data roots
    PROJECT_ROOT_MARKER_LEVELS_UP,
    DEFAULT_DATA_DIRNAME,
    DEFAULT_OUTPUT_DIRNAME,
    # schemas / ids
    NA_TOKENS,
    MAP_ID_WIDTH,
    PROMPT_ID_WIDTH_DEFAULT,
    PROMPTS_TILE_ID_COL,
    PROMPTS_MAP_ID_COL,
    PROMPTS_PROMPT_ID_COL,
    PROMPTS_TEXT_COL,
    OPERATOR_COL,
    INTENSITY_COL,
    PARAM_VALUE_COL,
    MAPS_ID_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    # filenames
    PROMPT_EMBEDDINGS_NPZ_NAME,
    PROMPTS_PARQUET_NAME,
    MAP_EMBEDDINGS_NPZ_NAME,
    MAPS_PARQUET_NAME,
    # concat output names
    CONCAT_X_CONCAT_NAME,
    CONCAT_TRAIN_PAIRS_NAME,
    CONCAT_META_JSON_NAME,
    CONCAT_SAVE_BLOCKS_NAMES,
    # behavior defaults
    CONCAT_VERBOSE_DEFAULT,
    CONCAT_FAIL_ON_MISSING_DEFAULT,
    CONCAT_DROP_DUPES_DEFAULT,
    CONCAT_L2_PROMPT_DEFAULT,
    CONCAT_SAVE_BLOCKS_DEFAULT,
    CONCAT_L2_EPS,
    CONCAT_PAD_NUMERIC_PROMPT_IDS_DEFAULT,
)


# Resolve project root relative to this file (stable in repo)
PROJECT_ROOT = Path(__file__).resolve().parents[int(PROJECT_ROOT_MARKER_LEVELS_UP)]
DATA_DIR = (PROJECT_ROOT / DEFAULT_DATA_DIRNAME).resolve()


def _resolve(p: str | Path) -> Path:
    """
    Resolve a path relative to:
      1) absolute paths as-is
      2) current working directory
      3) DATA_DIR fallback
    """
    p = Path(p)
    if p.is_absolute():
        return p
    cand = (Path.cwd() / p).resolve()
    return cand if cand.exists() or cand.parent.exists() else (DATA_DIR / p).resolve()


def setup_logging(verbosity: int = CONCAT_VERBOSE_DEFAULT) -> None:
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


def _normalize_id_series(s: pd.Series, *, width: int, pad_numeric: bool) -> pd.Series:
    """
    Keep IDs as string and optionally zero-pad purely numeric IDs.
    Prevents mismatches between Excel/parquet and embedding NPZ ids.
    """
    out = s.astype(str).str.strip()
    out = out.mask(out.str.lower().isin(NA_TOKENS | {"none"}), pd.NA)

    if pad_numeric:
        m = out.notna() & out.str.fullmatch(r"\d+")
        out.loc[m] = out.loc[m].str.zfill(int(width))

    return out.astype("string")


def _l2_rows(A: np.ndarray, eps: float = CONCAT_L2_EPS) -> np.ndarray:
    n = np.sqrt((A * A).sum(axis=1, keepdims=True))
    return A / np.maximum(n, float(eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Concatenate map & prompt embeddings via prompts.parquet.")

    # Default I/O locations (still overridable from CLI)
    ap.add_argument(
        "--map_npz",
        type=str,
        default=str(DATA_DIR / DEFAULT_OUTPUT_DIRNAME / "map_out" / MAP_EMBEDDINGS_NPZ_NAME),
    )
    ap.add_argument(
        "--prompt_npz",
        type=str,
        default=str(DATA_DIR / DEFAULT_OUTPUT_DIRNAME / "prompt_out" / PROMPT_EMBEDDINGS_NPZ_NAME),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DATA_DIR / DEFAULT_OUTPUT_DIRNAME / "train_out"),
    )
    ap.add_argument(
        "--maps_parquet",
        type=str,
        default=str(DATA_DIR / DEFAULT_OUTPUT_DIRNAME / "map_out" / MAPS_PARQUET_NAME),
        help="maps.parquet containing extent_* columns (from map embeddings).",
    )

    # Behavior toggles
    ap.add_argument("--fail_on_missing", action="store_true", default=CONCAT_FAIL_ON_MISSING_DEFAULT)
    ap.add_argument("--drop_dupes", action="store_true", default=CONCAT_DROP_DUPES_DEFAULT)
    ap.add_argument("-v", "--verbose", action="count", default=CONCAT_VERBOSE_DEFAULT)

    ap.add_argument(
        "--l2-prompt",
        action="store_true",
        default=CONCAT_L2_PROMPT_DEFAULT,
        help="L2-normalize prompt embeddings row-wise before concatenation.",
    )
    ap.add_argument(
        "--save-blocks",
        action="store_true",
        default=CONCAT_SAVE_BLOCKS_DEFAULT,
        help="Also save X_map.npy, X_prompt.npy, map_ids.npy, prompt_ids.npy.",
    )

    # Prompt-id normalization controls
    ap.add_argument(
        "--pad_numeric_prompt_ids",
        action="store_true",
        default=CONCAT_PAD_NUMERIC_PROMPT_IDS_DEFAULT,
        help="If prompt_id looks numeric, zero-pad it to --prompt_id_width.",
    )
    ap.add_argument(
        "--prompt_id_width",
        type=int,
        default=int(PROMPT_ID_WIDTH_DEFAULT),
        help="Width for zero-padding numeric prompt_id values.",
    )

    args = ap.parse_args()

    setup_logging(int(args.verbose))
    t0 = time.time()

    map_npz_path = _resolve(args.map_npz)
    prm_npz_path = _resolve(args.prompt_npz)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- prompts.parquet (authoritative for pairs)
    prompts_pq = _resolve(Path(args.prompt_npz).parent / PROMPTS_PARQUET_NAME)
    if not prompts_pq.exists():
        logging.error("%s not found next to prompt_npz: %s", PROMPTS_PARQUET_NAME, prompts_pq)
        sys.exit(1)

    pairs = pd.read_parquet(prompts_pq)

    # Required columns
    for col in (PROMPTS_PROMPT_ID_COL, PROMPTS_TILE_ID_COL):
        if col not in pairs.columns:
            logging.error("%s must contain '%s'", PROMPTS_PARQUET_NAME, col)
            sys.exit(1)

    # Keep canonical cols + optional columns if present
    keep_cols = [PROMPTS_TILE_ID_COL, PROMPTS_PROMPT_ID_COL, PROMPTS_TEXT_COL]
    for c in (OPERATOR_COL, INTENSITY_COL, PARAM_VALUE_COL):
        if c in pairs.columns:
            keep_cols.append(c)

    pairs = pairs[keep_cols].rename(columns={PROMPTS_TILE_ID_COL: PROMPTS_MAP_ID_COL}).copy()

    # Normalize IDs
    pairs[PROMPTS_MAP_ID_COL] = _normalize_id_series(
        pairs[PROMPTS_MAP_ID_COL],
        width=int(MAP_ID_WIDTH),
        pad_numeric=True,
    )
    pairs[PROMPTS_PROMPT_ID_COL] = _normalize_id_series(
        pairs[PROMPTS_PROMPT_ID_COL],
        width=int(args.prompt_id_width),
        pad_numeric=bool(args.pad_numeric_prompt_ids),
    )

    pairs = pairs.dropna(subset=[PROMPTS_MAP_ID_COL, PROMPTS_PROMPT_ID_COL])
    pairs = pairs[(pairs[PROMPTS_MAP_ID_COL] != "") & (pairs[PROMPTS_PROMPT_ID_COL] != "")]

    if args.drop_dupes:
        pairs = pairs.drop_duplicates(subset=[PROMPTS_MAP_ID_COL, PROMPTS_PROMPT_ID_COL])

    logging.info("Built %d pairs from %s", len(pairs), PROMPTS_PARQUET_NAME)

    # -------------------------------
    # Merge per-map extent refs (ONCE)
    # -------------------------------
    maps_pq = _resolve(args.maps_parquet)
    if not maps_pq.exists():
        logging.error("%s not found: %s (run map embeddings first)", MAPS_PARQUET_NAME, maps_pq)
        sys.exit(1)

    maps_df = pd.read_parquet(maps_pq)

    needed = [MAPS_ID_COL, EXTENT_DIAG_COL, EXTENT_AREA_COL]
    missing_cols = [c for c in needed if c not in maps_df.columns]
    if missing_cols:
        logging.error("%s missing columns %s. Re-run map embeddings with extent saving.", MAPS_PARQUET_NAME, missing_cols)
        sys.exit(1)

    # Keep extent columns (only those that exist)
    extent_keep = [
        MAPS_ID_COL,
        EXTENT_DIAG_COL,
        EXTENT_AREA_COL,
        "extent_width_m",
        "extent_height_m",
        "extent_minx",
        "extent_miny",
        "extent_maxx",
        "extent_maxy",
    ]
    extent_keep = [c for c in extent_keep if c in maps_df.columns]

    maps_df = maps_df[extent_keep].copy()
    maps_df[MAPS_ID_COL] = _normalize_id_series(maps_df[MAPS_ID_COL], width=int(MAP_ID_WIDTH), pad_numeric=True)

    before = len(pairs)
    pairs = pairs.merge(maps_df, left_on=PROMPTS_MAP_ID_COL, right_on=MAPS_ID_COL, how="left")

    miss_extent = int(pairs[EXTENT_DIAG_COL].isna().sum())
    if miss_extent:
        logging.warning("⚠️ %d rows missing %s after merge (map_id not found in %s).", miss_extent, EXTENT_DIAG_COL, MAPS_PARQUET_NAME)
        if args.fail_on_missing:
            logging.error("Failing because --fail_on_missing is set.")
            sys.exit(2)

    logging.info("Merged extent refs from %s into pairs (%d -> %d rows).", maps_pq, before, len(pairs))

    # --- load embeddings
    E_map, map_ids = load_npz(map_npz_path)
    E_prm, prm_ids = load_npz(prm_npz_path)
    logging.info("Map   embeddings: %s from %s", E_map.shape, map_npz_path)
    logging.info("Prompt embeddings: %s from %s", E_prm.shape, prm_npz_path)

    idx_map: dict[str, int] = {str(k).strip().zfill(int(MAP_ID_WIDTH)): i for i, k in enumerate(map_ids)}
    idx_prm: dict[str, int] = {str(k).strip(): i for i, k in enumerate(prm_ids)}

    # --- match & build X
    chosen_rows: List[int] = []
    im_list: List[int] = []
    ip_list: List[int] = []
    missing = 0

    pairs = pairs.reset_index(drop=True)

    for i, row in enumerate(pairs.itertuples(index=False), start=0):
        mid = str(getattr(row, PROMPTS_MAP_ID_COL)).strip().zfill(int(MAP_ID_WIDTH))
        pid_raw = str(getattr(row, PROMPTS_PROMPT_ID_COL)).strip()
        pid = pid_raw.zfill(int(args.prompt_id_width)) if (bool(args.pad_numeric_prompt_ids) and pid_raw.isdigit()) else pid_raw

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
        logging.error("Zero-dimension map or prompt block. map_dim=%d, prompt_dim=%d", X_map.shape[1], X_prm.shape[1])
        sys.exit(3)

    if bool(args.l2_prompt):
        X_prm = _l2_rows(X_prm)

    X = np.hstack([X_map, X_prm]).astype(np.float32, copy=False)
    np.save(out_dir / CONCAT_X_CONCAT_NAME, X)

    # Optional block saves (debug/analysis)
    if bool(args.save_blocks):
        np.save(out_dir / CONCAT_SAVE_BLOCKS_NAMES["X_map"], X_map)
        np.save(out_dir / CONCAT_SAVE_BLOCKS_NAMES["X_prompt"], X_prm)
        np.save(out_dir / CONCAT_SAVE_BLOCKS_NAMES["map_ids"], np.asarray([map_ids[i] for i in sel_map_idx], dtype=object))
        np.save(out_dir / CONCAT_SAVE_BLOCKS_NAMES["prompt_ids"], np.asarray([prm_ids[i] for i in sel_prm_idx], dtype=object))

    join_df = pairs.iloc[chosen_rows].reset_index(drop=True)
    cols = [c for c in join_df.columns if c not in (PROMPTS_MAP_ID_COL, PROMPTS_PROMPT_ID_COL)]
    join_df = join_df[[PROMPTS_MAP_ID_COL, PROMPTS_PROMPT_ID_COL, *cols]]

    assert X.shape[0] == len(join_df), "Row count mismatch between X and join_df."
    join_df.to_parquet(out_dir / CONCAT_TRAIN_PAIRS_NAME, index=False)

    outputs = {
        "X_concat_npy": CONCAT_X_CONCAT_NAME,
        "train_pairs_parquet": CONCAT_TRAIN_PAIRS_NAME,
    }
    if bool(args.save_blocks):
        outputs.update({
            "X_map_npy": CONCAT_SAVE_BLOCKS_NAMES["X_map"],
            "X_prompt_npy": CONCAT_SAVE_BLOCKS_NAMES["X_prompt"],
            "map_ids_npy": CONCAT_SAVE_BLOCKS_NAMES["map_ids"],
            "prompt_ids_npy": CONCAT_SAVE_BLOCKS_NAMES["prompt_ids"],
        })

    meta = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "map_dim": int(E_map.shape[1]),
        "prompt_dim": int(E_prm.shape[1]),
        "rows": int(X.shape[0]),
        "skipped_pairs": int(missing),
        "sources": {
            "prompts_parquet": str(prompts_pq),
            "maps_parquet": str(maps_pq),
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
            "map_ids": [map_ids[i] for i in sel_map_idx[:10]],
            "prompt_ids": [prm_ids[i] for i in sel_prm_idx[:10]],
        },
    }

    (out_dir / CONCAT_META_JSON_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("X shape = %s  (map_dim=%d, prompt_dim=%d)", X.shape, E_map.shape[1], E_prm.shape[1])
    logging.info("Saved to %s in %.2fs", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
