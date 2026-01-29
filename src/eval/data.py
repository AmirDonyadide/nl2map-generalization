# src/eval/data.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd


FeatureMode = Literal["prompt_only", "map_only", "prompt_plus_map"]


def load_pairs_and_features(
    *,
    pairs_path: Path,
    X_path: Path,
    ensure_alignment: bool = True,
    dtype=np.float64,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the evaluation pairs table and the feature matrix.
    """
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs parquet not found: {pairs_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {X_path}")

    df = pd.read_parquet(pairs_path).reset_index(drop=True)
    X = np.load(X_path).astype(dtype, copy=False)

    if ensure_alignment and len(df) != X.shape[0]:
        raise ValueError(
            f"Row mismatch: df has {len(df)} rows but X has {X.shape[0]} rows."
        )

    return df, X


def select_features(
    X: np.ndarray,
    *,
    feature_mode: FeatureMode,
    map_dim: int,
    prompt_dim: int,
) -> np.ndarray:
    """
    Select features for evaluation from a (possibly fused) feature matrix.

    Assumes fused layout when applicable:
        [ map_embedding | prompt_embedding ]

    feature_mode:
      - "prompt_only": return prompt block only
      - "map_only": return map block only
      - "prompt_plus_map": return [map | prompt] blocks
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D). Got shape {X.shape}")

    map_dim = int(map_dim)
    prompt_dim = int(prompt_dim)

    if map_dim < 0 or prompt_dim < 0:
        raise ValueError("map_dim and prompt_dim must be non-negative.")

    required = map_dim + prompt_dim
    if X.shape[1] < required:
        raise ValueError(
            f"X has {X.shape[1]} columns but expected at least {required} "
            f"(map_dim={map_dim}, prompt_dim={prompt_dim})."
        )

    if feature_mode == "prompt_only":
        if prompt_dim <= 0:
            raise ValueError("feature_mode='prompt_only' requires prompt_dim > 0.")
        return X[:, -prompt_dim:]

    if feature_mode == "map_only":
        if map_dim <= 0:
            raise ValueError("feature_mode='map_only' requires map_dim > 0.")
        return X[:, :map_dim]

    if feature_mode == "prompt_plus_map":
        return X[:, :required]

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def infer_dims_from_X(
    X: np.ndarray,
    *,
    prompt_dim: int,
) -> int:
    """
    Infer map_dim from a fused (or prompt-only) feature matrix given prompt_dim.

    If prompt_dim == D, this is interpreted as a prompt-only matrix -> map_dim = 0.
    """
    D = int(X.shape[1])
    prompt_dim = int(prompt_dim)

    if prompt_dim < 0:
        raise ValueError("prompt_dim must be non-negative.")

    if prompt_dim > D:
        raise ValueError(f"prompt_dim={prompt_dim} must be <= total feature dim D={D}.")

    return D - prompt_dim
