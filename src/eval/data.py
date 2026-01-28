# src/eval/data.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd


FeatureMode = Literal["prompt_only", "prompt_plus_map"]


def load_pairs_and_features(
    *,
    pairs_path: Path,
    X_path: Path,
    ensure_alignment: bool = True,
    dtype=np.float64,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the evaluation pairs table and the feature matrix.

    Parameters
    ----------
    pairs_path : Path
        Path to train_pairs.parquet (or equivalent).
    X_path : Path
        Path to feature matrix (.npy). Must align row-wise with pairs_path.
    ensure_alignment : bool
        If True, assert df rows == X rows.
    dtype : numpy dtype
        Cast feature matrix to this dtype (default float64).

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe with reset index.
    X : np.ndarray
        Feature matrix (N, D).
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
    Select features for evaluation from a fused feature matrix.

    Assumes fused layout:
        [ map_embedding | prompt_embedding ]

    Parameters
    ----------
    X : np.ndarray
        Fused feature matrix of shape (N, map_dim + prompt_dim + ...).
    feature_mode : {"prompt_only", "prompt_plus_map"}
        - "prompt_only": return only the prompt embedding block
        - "prompt_plus_map": return map + prompt blocks
    map_dim : int
        Dimensionality of map embedding block.
    prompt_dim : int
        Dimensionality of prompt embedding block.

    Returns
    -------
    np.ndarray
        Feature matrix sliced according to feature_mode.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D). Got shape {X.shape}")

    required = map_dim + prompt_dim
    if X.shape[1] < required:
        raise ValueError(
            f"X has {X.shape[1]} columns but expected at least {required} "
            f"(map_dim={map_dim}, prompt_dim={prompt_dim})."
        )

    if feature_mode == "prompt_only":
        return X[:, -prompt_dim:]

    if feature_mode == "prompt_plus_map":
        return X[:, : required]

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def infer_dims_from_X(
    X: np.ndarray,
    *,
    prompt_dim: int,
) -> int:
    """
    Infer map_dim from a fused feature matrix given prompt_dim.

    This is a convenience helper when map_dim is not explicitly stored.

    Parameters
    ----------
    X : np.ndarray
        Fused feature matrix (N, D).
    prompt_dim : int
        Known prompt embedding dimension.

    Returns
    -------
    int
        Inferred map_dim.

    Raises
    ------
    ValueError
        If prompt_dim >= D.
    """
    D = int(X.shape[1])
    if prompt_dim >= D:
        raise ValueError(
            f"prompt_dim={prompt_dim} must be < total feature dim D={D}."
        )
    return D - prompt_dim
