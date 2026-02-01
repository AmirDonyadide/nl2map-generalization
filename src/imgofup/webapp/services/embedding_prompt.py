# src/imgofup/webapp/services/embedding_prompt.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np

# (this is the file you pasted: src/mapvec/prompts/prompt_embeddings.py)
from imgofup.embeddings.prompts import (
    DEFAULT_DATA_DIR,
    OPENAI_BATCH_SIZE_DEFAULT,
    USE_BATCH_SIZE_DEFAULT,
    get_embedder,
)


def _default_batch_size(kind: str) -> int:
    k = (kind or "").strip().lower()
    # OpenAI embeddings are much slower -> use their batch size default
    if k.startswith("openai"):
        return int(OPENAI_BATCH_SIZE_DEFAULT)
    return int(USE_BATCH_SIZE_DEFAULT)


@lru_cache(maxsize=16)
def _get_embedder_cached(
    kind: str,
    data_dir_str: str,
    l2_normalize: bool,
    batch_size: int,
) -> Tuple[Callable[[List[str]], np.ndarray], str]:
    """
    Cached embedder factory for the webapp.

    Returns:
      (embed_fn, model_label)

    embed_fn: callable that accepts List[str] and returns np.ndarray (n, D)
    """
    kind_norm = (kind or "").strip().lower()
    if kind_norm == "use":
        kind_norm = "dan"  # your experiments use "dan" for USE

    data_dir = Path(data_dir_str).expanduser().resolve()

    embed_fn, label = get_embedder(
        kind_norm,
        data_dir=data_dir,
        l2_normalize=bool(l2_normalize),
        batch_size=int(batch_size),
    )

    if not callable(embed_fn):
        raise TypeError(f"get_embedder({kind_norm!r}) did not return a callable embed_fn.")

    return embed_fn, str(label)


def embed_prompt(
    prompt: str,
    *,
    encoder_kind: str,
    data_dir: Optional[Path] = None,
    l2_normalize: bool = True,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Embed a single prompt string into a 1D float32 vector.

    Uses your existing pipeline:
      - USE ('dan' / 'transformer') via TF Hub
      - OpenAI ('openai-small' / 'openai-large')

    Returns:
      vec: np.ndarray shape (D,) float32
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Prompt is empty.")

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if batch_size is None:
        batch_size = _default_batch_size(encoder_kind)

    embed_fn, _label = _get_embedder_cached(
        encoder_kind,
        str(data_dir),
        bool(l2_normalize),
        int(batch_size),
    )

    E = np.asarray(embed_fn([text]))

    if E.ndim != 2 or E.shape[0] != 1:
        raise ValueError(f"Prompt embedding has unexpected shape {E.shape}, expected (1, D).")

    vec = E[0].astype(np.float32, copy=False)

    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    return vec
