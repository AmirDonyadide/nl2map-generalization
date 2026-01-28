# src/utils/infer_dims.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np


def infer_dims_from_embeddings(
    *,
    prompt_out: Path,
    map_out: Path,
    prompt_key: str = "E",
    map_key: str = "E",
) -> Tuple[int, int, int]:
    """
    Infer MAP_DIM, PROMPT_DIM, and FUSED_DIM directly from saved embeddings.

    Expects:
      - prompt_out / "prompts_embeddings.npz"
      - map_out    / "maps_embeddings.npz"

    Returns
    -------
    MAP_DIM : int
    PROMPT_DIM : int
    FUSED_DIM : int
    """
    prm_npz = prompt_out / "prompts_embeddings.npz"
    map_npz = map_out / "maps_embeddings.npz"

    if not prm_npz.exists():
        raise FileNotFoundError(
            f"Missing {prm_npz}. Run prompt embedding step first."
        )

    if not map_npz.exists():
        raise FileNotFoundError(
            f"Missing {map_npz}. Run map embedding step first."
        )

    zp = np.load(prm_npz, allow_pickle=True)
    zm = np.load(map_npz, allow_pickle=True)

    if prompt_key not in zp or map_key not in zm:
        raise KeyError("Embedding NPZ must contain key 'E'.")

    PROMPT_DIM = int(zp[prompt_key].shape[1])
    MAP_DIM = int(zm[map_key].shape[1])
    FUSED_DIM = MAP_DIM + PROMPT_DIM

    return MAP_DIM, PROMPT_DIM, FUSED_DIM
