# src/inference/predict.py
"""
Unified inference entrypoint.

Goal
----
Given:
  - a map (GeoJSON path) OR precomputed map embedding
  - a prompt text
Return:
  - operator prediction
  - param_norm prediction
  - param_value prediction (meters for distance ops, m² for select)
  - debug info

Supports strategies:
  - "mlp": use trained classifier + per-operator regressors (MLPRegressor)
  - "hybrid": rule-based parsing + map refs (+ optional fallback to mlp)

This file intentionally depends on:
  - src/param_estimation/mlp_param.py
  - src/param_estimation/hybrid_param.py

You can plug in your existing embedding/preprocessing pipeline in the TODO sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence, cast


import numpy as np
import joblib

import math

def is_finite_number(x: Optional[float]) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def to_float_or_nan(x: Optional[float]) -> float:
    if x is None:
        return float("nan")
    # now x is float (not Optional) for type-checker
    return float(x) if math.isfinite(float(x)) else float("nan")


# --- local modules ---
from src.param_estimation.mlp_param import estimate_param_mlp_from_bundle
from src.param_estimation.hybrid_param import (
    estimate_param_hybrid,
    build_map_refs_from_geojson,
)


@dataclass
class InferenceArtifacts:
    bundle: Dict[str, Any]
    preproc: Optional[Dict[str, Any]] = None
    # Optional: store embedding functions, tokenizers, etc.
    # prompt_embedder: Optional[Any] = None
    # map_embedder: Optional[Any] = None


# -----------------------
# Loading
# -----------------------
def load_artifacts(
    bundle_path: Union[str, Path],
    preproc_path: Optional[Union[str, Path]] = None,
) -> InferenceArtifacts:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle: {bundle_path}")
    bundle = joblib.load(bundle_path)

    preproc = None
    if preproc_path is not None:
        preproc_path = Path(preproc_path)
        if not preproc_path.exists():
            raise FileNotFoundError(f"Missing preproc: {preproc_path}")
        preproc = joblib.load(preproc_path)

    return InferenceArtifacts(bundle=bundle, preproc=preproc)


# -----------------------
# Feature building
# -----------------------
def _apply_preproc(X_fused: np.ndarray, preproc: Dict[str, Any]) -> np.ndarray:
    """
    Apply the same preprocessing you used in training.
    Your notebook preproc is custom (split blocks, l2 prompt, impute/clip/scale map block).
    You saved pieces in preproc.joblib: imp, q_lo, q_hi, keep_mask, scaler, map_dim, prompt_dim.

    This function assumes X_fused is already [X_map | X_prompt] in the same raw space.
    """
    if preproc is None:
        return X_fused

    imp = preproc["imp"]
    q_lo = np.asarray(preproc["q_lo"])
    q_hi = np.asarray(preproc["q_hi"])
    keep_mask = np.asarray(preproc["keep_mask"]).astype(bool)
    scaler = preproc["scaler"]
    map_dim = int(preproc["map_dim"])
    prompt_dim = int(preproc["prompt_dim"])

    X = np.asarray(X_fused, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X_fused must be 2D")

    if X.shape[1] != map_dim + prompt_dim:
        raise ValueError(
            f"Dim mismatch: got {X.shape[1]}, expected {map_dim + prompt_dim} "
            f"(map_dim={map_dim}, prompt_dim={prompt_dim})"
        )

    Xm = X[:, :map_dim].copy()
    Xp = X[:, map_dim:map_dim + prompt_dim].copy()

    # L2 normalize prompt rows
    nrm = np.sqrt((Xp * Xp).sum(axis=1, keepdims=True))
    Xp = Xp / np.maximum(nrm, 1e-12)

    # maps: inf->nan
    Xm[~np.isfinite(Xm)] = np.nan

    # impute
    Xm_imp = imp.transform(Xm)

    # clip using train thresholds
    Xm_imp = np.clip(Xm_imp, q_lo, q_hi)

    # robust scale kept cols; dropped cols remain 0
    Xm_kept = scaler.transform(Xm_imp[:, keep_mask])

    Xm_out = np.zeros_like(Xm_imp, dtype=np.float64)
    Xm_out[:, keep_mask] = Xm_kept

    return np.concatenate([Xm_out, Xp], axis=1).astype(np.float64, copy=False)


def build_X_fused_from_embeddings(
    map_emb: np.ndarray,
    prompt_emb: np.ndarray,
) -> np.ndarray:
    """
    Simple fuse: [map_vec | prompt_vec].
    Inputs must be 1D vectors; returns shape (1, D).
    """
    m = np.asarray(map_emb).reshape(1, -1)
    p = np.asarray(prompt_emb).reshape(1, -1)
    return np.concatenate([m, p], axis=1)


# -----------------------
# Operator prediction
# -----------------------
def predict_operator(
    bundle: Dict[str, Any],
    X_fused_preproc: np.ndarray,
) -> Tuple[str, int, Dict[str, Any]]:
    clf = bundle.get("classifier")
    class_names = bundle.get("class_names")

    if clf is None or class_names is None:
        raise ValueError("Bundle must contain 'classifier' and 'class_names'")

    pred_idx = int(np.asarray(clf.predict(X_fused_preproc)).reshape(-1)[0])
    op = str(class_names[pred_idx]).strip().lower()

    debug = {"operator_idx": pred_idx, "operator": op}
    return op, pred_idx, debug


# -----------------------
# Unified prediction API
# -----------------------
def predict_operator_and_param(
    *,
    map_geojson: Union[str, Path],
    prompt_text: str,
    strategy: str = "mlp",  # "mlp" | "hybrid"
    artifacts: Optional[InferenceArtifacts] = None,
    bundle_path: Optional[Union[str, Path]] = None,
    preproc_path: Optional[Union[str, Path]] = None,
    # optional: precomputed embeddings (to avoid re-embedding)
    map_emb: Optional[np.ndarray] = None,
    prompt_emb: Optional[np.ndarray] = None,
    # optional: embedding callables if you want this function to compute embeddings
    map_embed_fn: Optional[Any] = None,
    prompt_embed_fn: Optional[Any] = None,
    # hybrid knobs
    hybrid_fallback_to_mlp: bool = True,
) -> Dict[str, Any]:
    """
    Main inference function.

    You can call it in 2 ways:
    A) Provide embeddings directly: map_emb + prompt_emb
    B) Provide embed functions: map_embed_fn(map_geojson)->vec, prompt_embed_fn(text)->vec

    Returns dict:
      {
        "operator": str,
        "operator_idx": int,
        "param_norm": float,
        "param_value": float,
        "strategy": "mlp"|"hybrid",
        "debug": {...}
      }
    """
    if artifacts is None:
        if bundle_path is None:
            raise ValueError("Provide artifacts or bundle_path.")
        artifacts = load_artifacts(bundle_path=bundle_path, preproc_path=preproc_path)

    bundle = artifacts.bundle
    preproc = artifacts.preproc

    gj_path = Path(map_geojson)

    # --- Build per-map refs (diag/area) from geojson (needed for un-normalization and hybrid logic)
    # This does NOT require extent CRS column; it computes bounds in the data CRS or projected if possible.
    map_refs_obj = build_map_refs_from_geojson(gj_path)
    map_refs: Dict[str, Any] = {
        "extent_diag_m": float(map_refs_obj.extent_diag_m),
        "extent_area_m2": float(map_refs_obj.extent_area_m2),
    }

    # --- Compute embeddings if not provided
    if map_emb is None:
        if map_embed_fn is None:
            raise ValueError("Provide map_emb or map_embed_fn.")
        map_emb = map_embed_fn(gj_path)

    if prompt_emb is None:
        if prompt_embed_fn is None:
            raise ValueError("Provide prompt_emb or prompt_embed_fn.")
        prompt_emb = prompt_embed_fn(prompt_text)
    
    assert map_emb is not None
    assert prompt_emb is not None

    # --- Fuse + preprocess like training
    X_fused = build_X_fused_from_embeddings(map_emb, prompt_emb)
    X_pre = _apply_preproc(X_fused, preproc) if preproc is not None else X_fused.astype(np.float64)

    # --- Predict operator
    op, op_idx, op_debug = predict_operator(bundle, X_pre)

    # --- Predict param according to strategy
    strategy = str(strategy).strip().lower()
    debug: Dict[str, Any] = {"operator": op_debug, "map_refs": map_refs}

    if strategy == "mlp":
        # bundle regressors need per-row refs -> pass as list of dicts length N (N=1)
        param_norm, param_value, mlp_dbg = estimate_param_mlp_from_bundle(
            bundle=bundle,
            X_fused=X_pre,
            operator_pred=[op],
            map_refs_batch=[map_refs],
            allow_nan=True,
        )
        debug["mlp"] = mlp_dbg
        out = {
            "operator": op,
            "operator_idx": op_idx,
            "param_norm": float(param_norm[0]) if is_finite_number(param_norm[0]) else float("nan"),
            "param_value": float(param_value[0]) if is_finite_number(param_value[0]) else float("nan"),
            "strategy": "mlp",
            "debug": debug,
        }
        return out

    if strategy == "hybrid":
        # Hybrid may still call MLP as fallback
        fallback_fn = None
        if hybrid_fallback_to_mlp:
            def _fallback_model(
                prompt_text_: str,
                operator_: str,
                map_refs_: Dict[str, Any],
                map_stats_: Dict[str, Any],
            ):
                pn, pv, mlp_dbg = estimate_param_mlp_from_bundle(
                    bundle=bundle,
                    X_fused=X_pre,
                    operator_pred=[operator_],
                    map_refs_batch=[map_refs_],
                    allow_nan=True,
                )
                # pn/pv are arrays length 1
                return float(pn[0]), float(pv[0]), {"source": "mlp_fallback", **(mlp_dbg or {})}

            fallback_fn = _fallback_model

        pn, pv, hy_dbg = estimate_param_hybrid(
            prompt_text=prompt_text,
            operator=op,
            map_refs=map_refs,
            map_stats={},
            fallback_model=fallback_fn,
        )

        debug["hybrid"] = hy_dbg

        return {
            "operator": op,
            "operator_idx": op_idx,
            "param_norm": to_float_or_nan(pn),
            "param_value": to_float_or_nan(pv),
            "strategy": "hybrid",
            "debug": debug,
        }

    raise ValueError("strategy must be 'mlp' or 'hybrid'")


# -----------------------
# Minimal CLI (optional)
# -----------------------
if __name__ == "__main__":
    import argparse, json as _json

    ap = argparse.ArgumentParser("Unified inference: operator + param.")
    ap.add_argument("--bundle", required=True, help="Path to cls_plus_regressors.joblib")
    ap.add_argument("--preproc", default=None, help="Path to preproc.joblib (optional)")
    ap.add_argument("--geojson", required=True, help="Path to input geojson")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--strategy", default="mlp", choices=["mlp", "hybrid"])
    args = ap.parse_args()

    # NOTE: For CLI you still must provide embedding functions OR modify this CLI
    # to load your embedding pipelines. We keep it simple and error out.
    raise SystemExit(
        "CLI stub: please call predict_operator_and_param() from your app/notebook "
        "with map_embed_fn and prompt_embed_fn (or precomputed embeddings)."
    )
