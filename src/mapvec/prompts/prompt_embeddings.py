# src/mapvec/prompts/prompt_embeddings.py
# Embed prompts (Excel) with USE or OpenAI embeddings and save artifacts.

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import logging
from typing import Tuple, List, Optional, Callable, Sequence
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.constants import (
    # repo/data discovery
    DEFAULT_DATA_DIRNAME,
    PROJECT_ROOT_MARKER_LEVELS_UP,
    MAPVEC_DATA_DIR_ENVVAR,
    DOTENV_FILENAME,
    # logging
    PROMPT_EMBED_LOG_DATEFMT,
    PROMPT_EMBED_VERBOSITY_DEFAULT,
    # schema / column names
    PROMPTS_TILE_ID_COL,
    PROMPTS_TEXT_COL,
    PROMPTS_PROMPT_ID_COL,
    PROMPTS_EXCEL_SHEET_DEFAULT,
    EXCEL_COMPLETE_COL_DEFAULT,
    EXCEL_REMOVE_COL_DEFAULT,
    EXCEL_TEXT_COL_DEFAULT,
    # filtering policy defaults
    EXCEL_ONLY_COMPLETE_DEFAULT,
    EXCEL_EXCLUDE_REMOVED_DEFAULT,
    # ID policy
    MAP_ID_WIDTH,
    PROMPT_ID_WIDTH_DEFAULT,
    NA_TOKENS,
    # output filenames
    PROMPT_EMBEDDINGS_NPZ_NAME,
    PROMPTS_PARQUET_NAME,
    PROMPTS_META_JSON_NAME,
    PROMPTS_EMBEDDINGS_CSV_NAME,
    # models / backends
    PROMPT_ENCODER_CHOICES,
    USE_KAGGLE_MODEL_IDS,
    USE_MODEL_DIR_DAN,
    USE_MODEL_DIR_TRANSFORMER,
    USE_BATCH_SIZE_DEFAULT,
    OPENAI_BATCH_SIZE_DEFAULT,
    OPENAI_MODEL_NAME_SMALL,
    OPENAI_MODEL_NAME_LARGE,
    OPENAI_PROMPT_PREFIX,
    # numeric stability
    L2_NORM_EPS,
)

# --- TF Hub is optional (only needed for USE)
try:
    import tensorflow_hub as hub  # type: ignore
except Exception:  # pragma: no cover
    hub = None


# ----------------------- project/data discovery -----------------------
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[int(PROJECT_ROOT_MARKER_LEVELS_UP)]
load_dotenv(dotenv_path=PROJECT_ROOT / DOTENV_FILENAME)


def get_default_data_dir() -> Path:
    env = os.environ.get(MAPVEC_DATA_DIR_ENVVAR)
    if env and env.strip():
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / DEFAULT_DATA_DIRNAME).resolve()


DEFAULT_DATA_DIR = get_default_data_dir()


# ----------------------- logging -----------------------
def setup_logging(verbosity: int = PROMPT_EMBED_VERBOSITY_DEFAULT) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt=PROMPT_EMBED_LOG_DATEFMT,
    )
    logging.debug("FILE_DIR=%s", FILE_DIR)
    logging.debug("PROJECT_ROOT=%s", PROJECT_ROOT)
    logging.debug("DEFAULT_DATA_DIR=%s", DEFAULT_DATA_DIR)


# ----------------------- Model discovery & loader (USE) -----------------------
def _default_model_dir(data_dir: Path, which: str) -> Path:
    which = which.lower()
    if which == "dan":
        return (data_dir / USE_MODEL_DIR_DAN).resolve()
    return (data_dir / USE_MODEL_DIR_TRANSFORMER).resolve()


def _has_saved_model(folder: Path) -> bool:
    return (folder / "saved_model.pb").exists() and (folder / "variables").exists()


def _copy_into(dst: Path, src: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _download_with_kagglehub(which: str) -> Optional[Path]:
    try:
        import kagglehub  # type: ignore
    except Exception:
        logging.error("kagglehub is not installed. Install it with: pip install kagglehub")
        return None

    model_id = USE_KAGGLE_MODEL_IDS[which]
    logging.info("Downloading USE-%s via kagglehub (%s)…", which, model_id)

    try:
        local_path = Path(kagglehub.model_download(model_id)).resolve()
        if _has_saved_model(local_path):
            return local_path
        for sub in local_path.iterdir():
            if sub.is_dir() and _has_saved_model(sub):
                return sub.resolve()
        logging.error("Downloaded content does not look like a TF SavedModel at %s", local_path)
        return None
    except Exception as e:
        logging.exception("Download failed: %s", e)
        return None


def ensure_local_use_model(which: str, dest_dir: Path) -> Path:
    which = which.lower()
    if which not in ("dan", "transformer"):
        raise ValueError("Unknown USE model. Choose 'dan' or 'transformer'.")

    if hub is None:
        raise RuntimeError(
            "tensorflow_hub is not available but a USE model was requested. "
            "Install tensorflow and tensorflow_hub, or choose an OpenAI model."
        )

    if _has_saved_model(dest_dir):
        logging.info("Using local USE-%s at %s", which, dest_dir)
        return dest_dir

    logging.info("Local USE model not found at %s. Attempting download…", dest_dir)
    dl_path = _download_with_kagglehub(which)
    if dl_path is None:
        raise RuntimeError(
            "Could not download the USE model automatically.\n"
            "Options:\n"
            "  A) Install kagglehub and try again:  pip install kagglehub\n"
            f"  B) Manually place the unpacked SavedModel under: {dest_dir}\n"
            "     The folder must contain: saved_model.pb, variables/, assets/ (assets optional)."
        )

    _copy_into(dest_dir, dl_path)
    if not _has_saved_model(dest_dir):
        raise RuntimeError(f"Model copy failed; SavedModel not found under {dest_dir}")

    logging.info("USE model ready at %s", dest_dir)
    return dest_dir


def load_use_local_or_download(which: str, data_dir: Path):
    dest_dir = _default_model_dir(data_dir, which)
    ready_dir = ensure_local_use_model(which, dest_dir)
    t0 = time.time()
    logging.info("Loading USE-%s from local path: %s …", which, ready_dir)
    model = hub.load(str(ready_dir))  # type: ignore[arg-type]
    logging.info("USE model loaded in %.2fs", time.time() - t0)
    return model


# ----------------------- Embedding helpers -----------------------
def _sanitize_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for i, t in enumerate(texts):
        if t is None:
            raise ValueError(f"Row {i}: text is None")
        s = str(t).strip()
        if not s:
            raise ValueError(f"Row {i}: text is empty after stripping")
        cleaned.append(s)
    return cleaned


def embed_texts_use(model, texts: List[str], *, l2_normalize: bool, batch_size: int) -> np.ndarray:
    texts = _sanitize_texts(texts)
    n = len(texts)
    if n == 0:
        raise ValueError("No prompts to embed (after cleaning).")

    logging.info("Embedding %d prompts with USE (batch_size=%d, l2=%s)…", n, batch_size, l2_normalize)
    t0 = time.time()

    probe = model([texts[0]]).numpy().astype(np.float32)
    dim = probe.shape[1]
    E = np.empty((n, dim), dtype=np.float32)
    E[0] = probe[0]

    start = 1
    while start < n:
        end = min(start + batch_size, n)
        batch = texts[start:end]
        E[start:end] = model(batch).numpy().astype(np.float32)
        logging.debug("  USE embedded rows [%d:%d)", start, end)
        start = end

    if l2_normalize:
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + float(L2_NORM_EPS))

    logging.info("Done USE embedding in %.2fs (dim=%d).", time.time() - t0, dim)
    return E


def embed_texts_openai(model_name: str, texts: List[str], *, l2_normalize: bool, batch_size: int) -> np.ndarray:
    from openai import OpenAI  # lazy import

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Create a .env file with your key.\n"
            "Example:\nOPENAI_API_KEY=sk-xxxx"
        )

    client = OpenAI(api_key=api_key)

    texts = _sanitize_texts(texts)
    n = len(texts)
    if n == 0:
        raise ValueError("No prompts to embed (after cleaning).")

    logging.info("Embedding %d prompts with OpenAI model=%s (batch_size=%d, l2=%s)…", n, model_name, batch_size, l2_normalize)
    t0 = time.time()

    all_vecs: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        # Optional domain context (controlled by constants)
        batch_ctx = [f"{OPENAI_PROMPT_PREFIX}{t}" for t in batch]

        resp = client.embeddings.create(model=model_name, input=batch_ctx)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
        logging.debug("  OpenAI embedded rows [%d:%d)", start, end)

    E = np.vstack(all_vecs).astype(np.float32)
    if l2_normalize:
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + float(L2_NORM_EPS))

    logging.info("Done OpenAI embedding in %.2fs (dim=%d).", time.time() - t0, E.shape[1])
    return E


def get_embedder(kind: str, *, data_dir: Path, l2_normalize: bool, batch_size: int) -> Tuple[Callable[[List[str]], np.ndarray], str]:
    kind = kind.lower()

    if kind in ("dan", "transformer"):
        use_model = load_use_local_or_download(kind, data_dir)

        def embed_fn(texts: List[str]) -> np.ndarray:
            return embed_texts_use(use_model, texts, l2_normalize=l2_normalize, batch_size=batch_size)

        return embed_fn, f"USE-{kind}"

    if kind == "openai-small":
        model_name = OPENAI_MODEL_NAME_SMALL
    elif kind == "openai-large":
        model_name = OPENAI_MODEL_NAME_LARGE
    else:
        raise ValueError(f"Unknown model kind '{kind}'. Use one of: {list(PROMPT_ENCODER_CHOICES)}")

    def embed_fn(texts: List[str]) -> np.ndarray:
        return embed_texts_openai(model_name, texts, l2_normalize=l2_normalize, batch_size=batch_size)

    return embed_fn, f"OpenAI-{model_name}"


# ----------------------- Excel loader -----------------------
def load_prompts_from_source(
    *,
    input_path: Path,
    sheet_name: str,
    tile_id_col: str,
    complete_col: str,
    remove_col: str,
    text_col: str,
    prompt_id_col: str = PROMPTS_PROMPT_ID_COL,
    only_complete: bool = EXCEL_ONLY_COMPLETE_DEFAULT,
    exclude_removed: bool = EXCEL_EXCLUDE_REMOVED_DEFAULT,
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Returns (ids, texts, tile_ids, id_colname) from an Excel user-study file.

    Filtering:
      - if only_complete: keep rows where complete_col is truthy
      - if exclude_removed: drop rows where remove_col is truthy

    IDs:
      - prompt_id_col is REQUIRED (this pipeline expects stable prompt IDs from Excel).
      - numeric-like IDs are zero-padded to PROMPT_ID_WIDTH_DEFAULT by default.
      - tile ids are zero-padded to MAP_ID_WIDTH.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    logging.info("Reading Excel: %s (sheet=%s)", input_path, sheet_name)
    df = pd.read_excel(input_path, sheet_name=sheet_name)

    required = [tile_id_col, complete_col, remove_col, text_col, prompt_id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Excel sheet '{sheet_name}' is missing required columns: {missing}")

    # booleans
    df[complete_col] = df[complete_col].astype(bool)
    df[remove_col] = df[remove_col].astype(bool)

    mask = pd.Series(True, index=df.index)
    if only_complete:
        mask &= (df[complete_col] == True)
    if exclude_removed:
        mask &= (df[remove_col] == False)

    df = df.loc[mask].copy()

    # text
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().ne("")].copy()
    if len(df) == 0:
        raise ValueError(f"No non-empty values found in '{text_col}' after filtering.")

    # prompt IDs (preserve leading zeros)
    raw_pid = df[prompt_id_col]
    pid_num = pd.to_numeric(raw_pid, errors="coerce")
    if pid_num.notna().all():
        df[prompt_id_col] = pid_num.astype(int).astype(str).str.zfill(int(PROMPT_ID_WIDTH_DEFAULT))
    else:
        pid_s = raw_pid.astype(str).str.strip()
        pid_s = pid_s.mask(pid_s.str.lower().isin(NA_TOKENS | {"none"}), pd.NA)
        df = df.dropna(subset=[prompt_id_col]).copy()
        pid_s = df[prompt_id_col].astype(str).str.strip()
        pid_num2 = pd.to_numeric(pid_s, errors="coerce")
        if pid_num2.notna().all():
            df[prompt_id_col] = pid_num2.astype(int).astype(str).str.zfill(int(PROMPT_ID_WIDTH_DEFAULT))
        else:
            df[prompt_id_col] = pid_s.str.zfill(int(PROMPT_ID_WIDTH_DEFAULT))

    if df[prompt_id_col].duplicated().any():
        dupes = df.loc[df[prompt_id_col].duplicated(), prompt_id_col].astype(str).tolist()
        raise ValueError(f"Duplicate {prompt_id_col} values found after filtering. Examples: {dupes[:10]}")

    # tile ids
    tile_raw = df[tile_id_col]
    tile_num = pd.to_numeric(tile_raw, errors="coerce")
    if tile_num.notna().all():
        tile_ids = tile_num.astype(int).astype(str).str.zfill(int(MAP_ID_WIDTH)).tolist()
    else:
        tile_ids = tile_raw.astype(str).str.strip().str.zfill(int(MAP_ID_WIDTH)).tolist()

    ids = df[prompt_id_col].astype(str).tolist()
    texts = df[text_col].tolist()
    return ids, texts, tile_ids, str(prompt_id_col)


# ----------------------- Save outputs -----------------------
def save_outputs(
    *,
    out_dir: Path,
    ids: List[str],
    texts: List[str],
    tile_ids: Optional[Sequence[Optional[str]]],
    E: np.ndarray,
    model_name: str,
    l2_normalized: bool,
    id_colname: str = PROMPTS_PROMPT_ID_COL,
    also_save_embeddings_csv: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to %s", out_dir)

    npz_path = out_dir / PROMPT_EMBEDDINGS_NPZ_NAME
    np.savez_compressed(npz_path, E=E, ids=np.array(ids, dtype=object))
    logging.info("  saved %s (shape=%s)", npz_path.name, tuple(E.shape))

    df_out = pd.DataFrame({id_colname: ids, PROMPTS_TEXT_COL: texts})

    if tile_ids is not None:
        if len(tile_ids) != len(ids):
            raise ValueError(f"tile_ids length {len(tile_ids)} != ids length {len(ids)}")
        df_out[PROMPTS_TILE_ID_COL] = [None if v is None else str(v).zfill(int(MAP_ID_WIDTH)) for v in tile_ids]

    pq_path = out_dir / PROMPTS_PARQUET_NAME
    df_out.to_parquet(pq_path, index=False)
    logging.info("  saved %s (rows=%d)", pq_path.name, len(df_out))

    csv_name = None
    if also_save_embeddings_csv:
        D = E.shape[1]
        cols = [f"e{i:04d}" for i in range(D)]
        dfw = pd.DataFrame(E, columns=cols)
        dfw.insert(0, id_colname, ids)
        csv_path = out_dir / PROMPTS_EMBEDDINGS_CSV_NAME
        dfw.to_csv(csv_path, index=False)
        logging.info("  saved %s", csv_path.name)
        csv_name = csv_path.name

    meta = {
        "model": model_name,
        "dim": int(E.shape[1]),
        "count": int(E.shape[0]),
        "l2_normalized": bool(l2_normalized),
        "id_colname": id_colname,
        "files": {
            "embeddings_npz": npz_path.name,
            "prompts_parquet": pq_path.name,
            "embeddings_csv": csv_name,
        },
    }
    meta_path = out_dir / PROMPTS_META_JSON_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("  saved %s", meta_path.name)


# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="Embed prompts with USE or OpenAI embeddings and save artifacts.")

    parser.add_argument("--input", type=str, default=None, help="Path to Excel user-study file (.xlsx/.xls).")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help=f"Data directory (env: {MAPVEC_DATA_DIR_ENVVAR}).")

    parser.add_argument("--model", type=str, default="dan", choices=list(PROMPT_ENCODER_CHOICES))
    parser.add_argument("--l2", action="store_true", default=True, help="L2-normalize embeddings (recommended).")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <data_dir>/output/prompt_out).")
    parser.add_argument("--embeddings_csv", action="store_true", default=False, help="Also save embeddings CSV.")
    parser.add_argument("--batch_size", type=int, default=int(USE_BATCH_SIZE_DEFAULT))
    parser.add_argument("--openai_batch_size", type=int, default=int(OPENAI_BATCH_SIZE_DEFAULT))
    parser.add_argument("--sheet", type=str, default=PROMPTS_EXCEL_SHEET_DEFAULT)

    parser.add_argument("--tile_id_col", type=str, default=PROMPTS_TILE_ID_COL)
    parser.add_argument("--prompt_id_col", type=str, default=PROMPTS_PROMPT_ID_COL)
    parser.add_argument("--complete_col", type=str, default=EXCEL_COMPLETE_COL_DEFAULT)
    parser.add_argument("--remove_col", type=str, default=EXCEL_REMOVE_COL_DEFAULT)
    parser.add_argument("--text_col", type=str, default=EXCEL_TEXT_COL_DEFAULT)

    parser.add_argument("--only_complete", action="store_true", default=EXCEL_ONLY_COMPLETE_DEFAULT)
    parser.add_argument("--exclude_removed", action="store_true", default=EXCEL_EXCLUDE_REMOVED_DEFAULT)

    parser.add_argument("-v", "--verbose", action="count", default=PROMPT_EMBED_VERBOSITY_DEFAULT)
    args = parser.parse_args()

    setup_logging(int(args.verbose))

    data_dir = Path(args.data_dir).resolve()
    in_path = Path(args.input).expanduser().resolve() if args.input else (data_dir / "input" / "UserStudy.xlsx")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (data_dir / "output" / "prompt_out")

    if not in_path.exists():
        logging.error("Input not found: %s", in_path)
        sys.exit(2)

    try:
        ids, texts, tile_ids, id_colname = load_prompts_from_source(
            input_path=in_path,
            sheet_name=args.sheet,
            tile_id_col=args.tile_id_col,
            complete_col=args.complete_col,
            remove_col=args.remove_col,
            text_col=args.text_col,
            prompt_id_col=args.prompt_id_col,
            only_complete=bool(args.only_complete),
            exclude_removed=bool(args.exclude_removed),
        )

        # Choose batch size depending on backend
        batch_size = int(args.openai_batch_size) if str(args.model).startswith("openai") else int(args.batch_size)

        embed_fn, model_label = get_embedder(
            str(args.model),
            data_dir=data_dir,
            l2_normalize=bool(args.l2),
            batch_size=batch_size,
        )

        E = embed_fn(texts)

        save_outputs(
            out_dir=out_dir,
            ids=ids,
            texts=texts,
            tile_ids=tile_ids,
            E=E,
            model_name=model_label,
            l2_normalized=bool(args.l2),
            id_colname=id_colname,
            also_save_embeddings_csv=bool(args.embeddings_csv),
        )

        logging.info("All done ✅")
    except Exception as e:
        logging.exception("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
