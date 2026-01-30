# prompt_embeddings.py
# Embed prompts (CSV/TXT) with USE-DAN/Transformer or OpenAI LLM embeddings and save artifacts.
# Default I/O lives under the project ./data folder (at repo root, not inside src/).

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

# --- TF Hub is optional now (only needed for USE)  # === NEW
try:
    import tensorflow_hub as hub  # type: ignore
except Exception:  # pragma: no cover
    hub = None

# ----------------------- project/data discovery -----------------------
def find_project_root(start: Path) -> Path:
    """
    Walk up from `start` to find a directory that looks like the project root:
    prefers a directory that contains 'data' or '.git', or the parent of 'src'.
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "data").is_dir() or (p / ".git").is_dir():
            return p
        if p.name == "src":
            return p.parent
    # fallback to top-level parent
    return cur.parents[-1] if len(cur.parents) else cur

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(FILE_DIR)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def get_default_data_dir() -> Path:
    env = os.environ.get("MAPVEC_DATA_DIR")
    if env and env.strip():
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "data").resolve()

DEFAULT_DATA_DIR = get_default_data_dir()

# ----------------------- logging -----------------------
def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.debug("FILE_DIR=%s", FILE_DIR)
    logging.debug("PROJECT_ROOT=%s", PROJECT_ROOT)
    logging.debug("DEFAULT_DATA_DIR=%s", DEFAULT_DATA_DIR)

# ----------------------- Model discovery & loader (USE) -----------------------
# Kaggle Models IDs for TF2 variants (public, same models TF-Hub points to)
_KAGGLE_MODEL_IDS = {
    "dan": "google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2",
    "transformer": "google/universal-sentence-encoder-large/tensorFlow2/universal-sentence-encoder-large/2",
}

def _default_model_dir(data_dir: Path, which: str) -> Path:
    # Stable local folders; also check legacy 'model' folder as a fallback
    if which == "dan":
        return (data_dir / "input" / "model_dan").resolve()
    else:
        return (data_dir / "input" / "model_transformer").resolve()

def _has_saved_model(folder: Path) -> bool:
    return (folder / "saved_model.pb").exists() and (folder / "variables").exists()

def _copy_into(dst: Path, src: Path):
    dst.mkdir(parents=True, exist_ok=True)
    # If the Kaggle download gives a folder that already contains saved_model.pb,
    # just copy its content into dst.
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _download_with_kagglehub(which: str) -> Optional[Path]:
    """
    Downloads the USE model via kagglehub and returns the local path to the SavedModel.
    Returns None on failure.
    """
    try:
        import kagglehub  # pip install kagglehub
    except Exception:
        logging.error("kagglehub is not installed. Install it with: pip install kagglehub")
        return None

    model_id = _KAGGLE_MODEL_IDS[which]
    logging.info("Downloading USE-%s via kagglehub (%s)…", which, model_id)
    try:
        local_path = Path(kagglehub.model_download(model_id)).resolve()
        # The returned path is a folder that should contain saved_model.pb
        if _has_saved_model(local_path):
            logging.info("Downloaded to %s", local_path)
            return local_path
        # Some packages place the SavedModel one level deeper; scan one level
        for sub in local_path.iterdir():
            if sub.is_dir() and _has_saved_model(sub):
                logging.info("Found SavedModel inside %s", sub)
                return sub.resolve()
        logging.error("Downloaded content does not look like a TF SavedModel at %s", local_path)
        return None
    except Exception as e:
        logging.exception("Download failed: %s", e)
        return None

def ensure_local_use_model(which: str, dest_dir: Path) -> Path:
    """
    Ensure the USE model is present locally under dest_dir.
    - If present, return dest_dir.
    - If missing, try to download via kagglehub and copy there.
    """
    which = which.lower()
    if which not in ("dan", "transformer"):
        raise ValueError("Unknown USE model. Choose 'dan' or 'transformer'.")

    if hub is None:  # === NEW
        raise RuntimeError(
            "tensorflow_hub is not available but a USE model was requested. "
            "Install tensorflow and tensorflow_hub, or choose an OpenAI model."
        )

    # 1) Already present?
    if _has_saved_model(dest_dir):
        logging.info("Using local USE-%s at %s", which, dest_dir)
        return dest_dir

    # 2) Attempt to download
    logging.info("Local USE model not found at %s. Attempting download…", dest_dir)
    dl_path = _download_with_kagglehub(which)
    if dl_path is None:
        # Give a clear, actionable message
        raise RuntimeError(
            "Could not download the USE model automatically.\n"
            "Options:\n"
            f"  A) Install kagglehub and try again:  pip install kagglehub\n"
            f"  B) Manually place the unpacked SavedModel under: {dest_dir}\n"
            "     The folder must contain: saved_model.pb, variables/, assets/ (assets optional)."
        )

    # 3) Copy downloaded SavedModel into our canonical dest_dir
    _copy_into(dest_dir, dl_path)
    if not _has_saved_model(dest_dir):
        raise RuntimeError(f"Model copy failed; SavedModel not found under {dest_dir}")

    logging.info("USE model ready at %s", dest_dir)
    return dest_dir

def load_use_local_or_download(which: str, data_dir: Path):
    """
    Resolves a local path for the USE model (data/input/model_*) and loads it with hub.load.
    If missing, downloads via kagglehub, installs it into that path, and loads.
    """
    dest_dir = _default_model_dir(data_dir, which)
    ready_dir = ensure_local_use_model(which, dest_dir)
    t0 = time.time()
    logging.info("Loading USE-%s from local path: %s …", which, ready_dir)
    model = hub.load(str(ready_dir))  # type: ignore[arg-type]
    logging.info("USE model loaded in %.2fs", time.time() - t0)
    return model

# ----------------------- Embedding helpers (common) -----------------------
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

def embed_texts_use(model, texts: List[str], l2_normalize: bool = True,
                    batch_size: int = 512) -> np.ndarray:
    """Embeds a list of strings with USE. Returns (N, D) float32."""
    texts = _sanitize_texts(texts)
    n = len(texts)
    if n == 0:
        raise ValueError("No prompts to embed (after cleaning).")

    logging.info("Embedding %d prompts with USE (batch_size=%d, l2=%s)…",
                 n, batch_size, l2_normalize)
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
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    logging.info("Done USE embedding in %.2fs (dim=%d).", time.time() - t0, dim)
    return E

# ----------------------- OpenAI LLM embedding helpers  # === NEW -----------------------
def embed_texts_openai(model_name: str,
                       texts: List[str],
                       l2_normalize: bool = True,
                       batch_size: int = 256) -> np.ndarray:
    """
    Embed texts with an OpenAI embedding model (e.g. text-embedding-3-small).
    Returns (N, D) float32.
    """
    from openai import OpenAI  # imported lazily so it's optional
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. "
            "Create a .env file with your key. Example:\n"
            "OPENAI_API_KEY=sk-xxxx"
        )
        
    client = OpenAI(api_key=api_key)  # uses OPENAI_API_KEY from env

    texts = _sanitize_texts(texts)
    n = len(texts)
    if n == 0:
        raise ValueError("No prompts to embed (after cleaning).")

    logging.info("Embedding %d prompts with OpenAI model=%s (batch_size=%d, l2=%s)…",
                 n, model_name, batch_size, l2_normalize)
    t0 = time.time()

    all_vecs: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        # Optional domain context – you can adjust or remove this line:
        batch_ctx = [f"Cartographic map generalization instruction: {t}" for t in batch]

        resp = client.embeddings.create(
            model=model_name,
            input=batch_ctx,
        )

        # resp.data is a list of objects with .embedding
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
        logging.debug("  OpenAI embedded rows [%d:%d)", start, end)

    E = np.vstack(all_vecs).astype(np.float32)

    if l2_normalize:
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    logging.info("Done OpenAI embedding in %.2fs (dim=%d).",
                 time.time() - t0, E.shape[1])
    return E

# ----------------------- Backend selector  # === NEW -----------------------
def get_embedder(
    kind: str,
    data_dir: Path,
    l2_normalize: bool,
    batch_size: int,
) -> Tuple[Callable[[List[str]], np.ndarray], str]:
    """
    Returns (embed_fn, model_label) for the requested backend.
    - embed_fn(texts) -> np.ndarray[N, D]
    - model_label: human-readable name stored in meta.json
    """
    kind = kind.lower()

    if kind in ("dan", "transformer"):
        # USE backend
        use_model = load_use_local_or_download(kind, data_dir)

        def embed_fn(texts: List[str]) -> np.ndarray:
            return embed_texts_use(
                use_model,
                texts,
                l2_normalize=l2_normalize,
                batch_size=batch_size,
            )

        model_label = f"USE-{kind}"
        return embed_fn, model_label

    # OpenAI backends
    if kind == "openai-small":
        model_name = "text-embedding-3-small"
    elif kind == "openai-large":
        model_name = "text-embedding-3-large"
    else:
        raise ValueError(
            f"Unknown model kind '{kind}'. "
            "Use one of: dan, transformer, openai-small, openai-large."
        )

    def embed_fn(texts: List[str]) -> np.ndarray:
        return embed_texts_openai(
            model_name=model_name,
            texts=texts,
            l2_normalize=l2_normalize,
            batch_size=batch_size,
        )

    model_label = f"OpenAI-{model_name}"
    return embed_fn, model_label

# ----------------------- I/O helpers -----------------------
def load_prompts_from_source(
    input_path: Path,
    sheet_name: str,
    tile_id_col: str,
    complete_col: str,
    remove_col: str,
    text_col: str,
    prompt_id_col: str = "prompt_id",   # ✅ NEW
):

    """
    Returns (ids, texts, tile_ids, id_colname)

    Supported inputs:
      - .txt  → ids are p000, p001, ...
      - .csv  → columns (prompt_id,text) or (id,text)
      - .xlsx → user study Excel:
                keeps rows where complete==True AND remove==False,
                embeds text_col, generates prompt_id.
    """
    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"Missing or invalid --input path: {input_path}")

    ext = input_path.suffix.lower()

    # ===================== EXCEL =====================
    if ext in (".xlsx", ".xls"):
        logging.info("Reading Excel: %s (sheet=%s)", input_path, sheet_name)
        df = pd.read_excel(input_path, sheet_name=sheet_name)

        # ---- Validate schema ----
        required = [tile_id_col, complete_col, remove_col, text_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Excel sheet '{sheet_name}' is missing required columns: {missing}"
            )

        # ---- Normalize booleans (Excel-safe) ----
        df[complete_col] = df[complete_col].astype(bool)
        df[remove_col]   = df[remove_col].astype(bool)

        # ---- Filter rows ----
        before = len(df)
        # Optional: read defaults from src.config if available
        only_complete = True
        exclude_removed = True
        try:
            from src.config import PATHS
            only_complete = getattr(PATHS, "ONLY_COMPLETE", True)
            exclude_removed = getattr(PATHS, "EXCLUDE_REMOVED", True)
        except Exception:
            pass

        mask = pd.Series(True, index=df.index)
        if only_complete:
            mask &= (df[complete_col] == True)
        if exclude_removed:
            mask &= (df[remove_col] == False)

        df = df[mask].copy()

        after = len(df)

        if after == 0:
            raise ValueError(
                f"No rows left after filtering (only_complete={only_complete}, exclude_removed={exclude_removed})."
            )

        logging.info(
            "Filtered Excel rows: %d → %d (only_complete=%s, exclude_removed=%s)",
            before, after, only_complete, exclude_removed
        )

        # ---- Clean text ----
        df[text_col] = df[text_col].astype(str)
        df = df[df[text_col].str.strip().ne("")].copy()

        if len(df) == 0:
            raise ValueError(
                f"No non-empty values found in text column '{text_col}' after filtering."
            )

        # ---- Generate prompt IDs ----
        # ---- Use prompt_id from Excel (do NOT generate IDs) ----
        # ---- Use prompt_id from Excel (do NOT generate IDs) ----
        if prompt_id_col not in df.columns:
            raise ValueError(
                f"Excel sheet '{sheet_name}' is missing required column '{prompt_id_col}'. "
                "You want to use Excel IDs (0001, 0002, ...), so this column must exist."
            )

        # Preserve leading zeros robustly:
        # - If Excel stored as numeric (e.g., 1.0), convert safely to int then zfill.
        # - Otherwise treat as string and zfill.
        raw_pid = df[prompt_id_col]

        pid_num = pd.to_numeric(raw_pid, errors="coerce")
        if pid_num.notna().all():
            # Excel numeric column: 1, 2, 3 ... -> "0001", "0002", ...
            df[prompt_id_col] = pid_num.astype(int).astype(str).str.zfill(4)
        else:
            # Mixed / string column: keep as string, strip, normalize
            df[prompt_id_col] = raw_pid.astype(str).str.strip()

            # Treat typical "empty" tokens as missing
            df.loc[df[prompt_id_col].str.lower().isin(["", "nan", "none"]), prompt_id_col] = pd.NA
            df = df.dropna(subset=[prompt_id_col]).copy()

            # If IDs look numeric but as strings ("1", "2"), also normalize
            pid_num2 = pd.to_numeric(df[prompt_id_col], errors="coerce")
            if pid_num2.notna().all():
                df[prompt_id_col] = pid_num2.astype(int).astype(str).str.zfill(4)
            else:
                # Otherwise just pad to 4 (safe even if already "0001")
                df[prompt_id_col] = df[prompt_id_col].astype(str).str.zfill(4)

        # Final sanity checks
        if df[prompt_id_col].duplicated().any():
            dupes = df.loc[df[prompt_id_col].duplicated(), prompt_id_col].astype(str).tolist()
            raise ValueError(
                "Duplicate prompt_id values found in Excel after filtering. "
                f"Examples: {dupes[:10]}"
            )

        # OPTIONAL strict format check (uncomment if you want exactly 4 digits)
        # bad = df.loc[~df[prompt_id_col].str.fullmatch(r"\d{4}"), prompt_id_col].tolist()
        # if bad:
        #     raise ValueError(f"prompt_id must be exactly 4 digits. Bad examples: {bad[:10]}")
        # ---- tile_ids ----
        tile_raw = df[tile_id_col]
        tile_num = pd.to_numeric(tile_raw, errors="coerce")
        if tile_num.notna().all():
            tile_ids = tile_num.astype(int).astype(str).str.zfill(4).tolist()
        else:
            tile_ids = tile_raw.astype(str).str.strip().str.zfill(4).tolist()

        ids = df[prompt_id_col].tolist()
        texts = df[text_col].tolist()
        return ids, texts, tile_ids, "prompt_id"


    # ===================== UNSUPPORTED =====================
    raise ValueError(
        "Unsupported input type. Use .xlsx/.xls."
    )


def save_outputs(
    out_dir: Path,
    ids: List[str],
    texts: List[str],
    E: np.ndarray,
    model_name: str,
    l2_normalized: bool,
    id_colname: str = "prompt_id",
    tile_ids: Optional[Sequence[Optional[str]]] = None,  # ✅ NEW
    also_save_embeddings_csv: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to %s", out_dir)

    npz_path = out_dir / "prompts_embeddings.npz"
    np.savez_compressed(npz_path, E=E, ids=np.array(ids, dtype=object))
    logging.info("  saved %s (shape=%s)", npz_path.name, tuple(E.shape))

    # --- build prompts table ---
    df_out = pd.DataFrame({
        id_colname: ids,
        "text": texts,
    })

    # ✅ Add tile_id if provided
    if tile_ids is not None:
        if len(tile_ids) != len(ids):
            raise ValueError(f"tile_ids length {len(tile_ids)} != ids length {len(ids)}")
        df_out["tile_id"] = [None if v is None else str(v).zfill(4) for v in tile_ids]

    pq_path = out_dir / "prompts.parquet"
    df_out.to_parquet(pq_path, index=False)
    logging.info("  saved %s (rows=%d)", pq_path.name, len(df_out))

    csv_name = None
    if also_save_embeddings_csv:
        D = E.shape[1]
        cols = [f"e{i:04d}" for i in range(D)]
        dfw = pd.DataFrame(E, columns=cols)
        dfw.insert(0, id_colname, ids)
        csv_path = out_dir / "embeddings.csv"
        dfw.to_csv(csv_path, index=False)
        logging.info("  saved %s", csv_path.name)
        csv_name = csv_path.name

    meta = {
        "model": model_name,  # no longer hard-coded "USE-" prefix  # === NEW
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
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("  saved %s", meta_path.name)

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Embed prompts with USE or OpenAI embeddings and save artifacts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to .txt (one per line) or .csv with columns prompt_id,text (or id,text). "
            "Default: <data_dir>/prompts.csv"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=(
            "Directory that holds inputs/outputs (default: auto-detected project data dir). "
            "Env override: MAPVEC_DATA_DIR"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dan",
        choices=["dan", "transformer", "openai-small", "openai-large"],  # === NEW
        help=(
            "Prompt encoder backend: "
            "dan/transformer = USE; openai-small/openai-large = OpenAI embedding models."
        ),
    )
    parser.add_argument(
        "--l2",
        action="store_true",
        help="L2-normalize embeddings (recommended).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <data_dir>/output/prompt_out).",
    )
    parser.add_argument(
        "--embeddings_csv",
        action="store_true",
        help="Also save a wide embeddings.csv with columns prompt_id,e0000..eXXXX.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for embedding calls.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v, -vv).",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_dir = Path(args.data_dir).resolve()
    in_path = Path(args.input).expanduser().resolve() if args.input else (data_dir / "input" / "prompts.csv")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (data_dir / "output" / "prompt_out")

    logging.info("DATA_DIR=%s", data_dir)
    logging.info("INPUT=%s", in_path)
    logging.info("OUT_DIR=%s", out_dir)

    if not in_path.exists():
        logging.error("Input not found: %s", in_path)
        logging.error("Tips: run with --data_dir <your-repo>/data  or set MAPVEC_DATA_DIR")
        sys.exit(2)

    try:
        ids, texts, tile_ids, id_colname = load_prompts_from_source(
            input_path=in_path,
            sheet_name="Responses",
            tile_id_col="tile_id",
            complete_col="complete",
            remove_col="remove",
            text_col="cleaned_text",
            prompt_id_col="prompt_id",  # ✅ NEW
        )

        logging.info(
            "Loaded %d prompts (id_col=%s). Sample IDs: %s",
            len(ids),
            id_colname,
            ", ".join(ids[:3]) + ("…" if len(ids) > 3 else ""),
        )

        # Backend selection (USE or OpenAI)  # === NEW
        embed_fn, model_label = get_embedder(
            kind=args.model,
            data_dir=data_dir,
            l2_normalize=args.l2,
            batch_size=args.batch_size,
        )

        E = embed_fn(texts)

        save_outputs(
            out_dir=out_dir,
            ids=ids,
            texts=texts,
            E=E,
            model_name=model_label,
            l2_normalized=args.l2,
            id_colname=id_colname if id_colname in ("prompt_id", "id") else "prompt_id",
            tile_ids=tile_ids,     
            also_save_embeddings_csv=args.embeddings_csv,
        )

        logging.info("All done ✅")
    except Exception as e:
        logging.exception("Failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()