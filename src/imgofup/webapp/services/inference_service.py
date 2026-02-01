# src/imgofup/webapp/services/inference_service.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from imgofup.config.constants import EXTENT_AREA_COL, EXTENT_DIAG_COL
from imgofup.webapp.schemas import Prediction
from imgofup.webapp.services.embedding_map import embed_geojson_map
from imgofup.webapp.services.embedding_prompt import embed_prompt
from imgofup.webapp.services.model_registry import ModelHandle


# -----------------------------------------------------------------------------
# Small, robust loaders (cached)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _load_joblib(p: str) -> Any:
    return joblib.load(p)


@lru_cache(maxsize=128)
def _load_json(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _bundle_get(bundle: Any, key: str, default: Any = None) -> Any:
    """Support bundle as dict-like or object-like."""
    if isinstance(bundle, dict):
        return bundle.get(key, default)
    return getattr(bundle, key, default)


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _resolve_artifact(model_dir: Path, rel: str) -> Path:
    p = (model_dir / rel).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {p}")
    return p


def _infer_repo_root_from_model_dir(model_dir: Path) -> Path:
    """
    model_dir is expected to be: <repo>/models/<model_id>
    so repo root is model_dir.parents[1].
    """
    return model_dir.resolve().parents[1]


# -----------------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------------
def predict_operator_and_param(
    model: ModelHandle,
    prompt: str,
    geojson: Dict[str, Any],
) -> Prediction:
    """
    Real inference for your saved sklearn joblib artifacts.

    Expected model folder structure (per /models/<model_id>/config.json):
      - preproc.joblib
      - classifier_meta.json
      - cls_plus_regressors.joblib   (bundle)
      - classifier.joblib            (optional; bundle also contains classifier)

    Pipeline:
      1) Embed geojson -> map_vec + extent refs
      2) Embed prompt -> prompt_vec (USE/OpenAI) depending on config
      3) X = [map_vec | prompt_vec]  (matches training concat)
      4) preproc.transform(X)
      5) classifier predicts operator (+ confidence)
      6) bundle chooses regressors_by_class[operator] -> (dist_reg, area_reg)
      7) predict normalized parameter, inverse log1p if needed
      8) denormalize using extent_diag or extent_area
    """
    model_dir = Path(model.model_dir)
    cfg = model.config or {}

    # ---- read artifact names from config (fallbacks to your standard filenames) ----
    artifacts = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts", {}), dict) else {}
    preproc_name = artifacts.get("preprocessor", "preproc.joblib")
    meta_name = artifacts.get("meta", "classifier_meta.json")
    bundle_name = artifacts.get("regressors", "cls_plus_regressors.joblib")
    classifier_name = artifacts.get("classifier", "classifier.joblib")  # optional

    preproc_path = _resolve_artifact(model_dir, preproc_name)
    meta_path = _resolve_artifact(model_dir, meta_name)
    bundle_path = _resolve_artifact(model_dir, bundle_name)

    preproc = _load_joblib(str(preproc_path))
    meta = _load_json(str(meta_path))
    bundle = _load_joblib(str(bundle_path))

    # classifier: prefer bundle.classifier; fallback to classifier.joblib if present
    classifier = _bundle_get(bundle, "classifier", None)
    if classifier is None and (model_dir / classifier_name).exists():
        classifier = _load_joblib(str(_resolve_artifact(model_dir, classifier_name)))
    if classifier is None:
        raise RuntimeError("No classifier found (neither in bundle nor as classifier.joblib).")

    # ---- determine feature mode & prompt encoder ----
    feature_mode = str(cfg.get("feature_mode", meta.get("feature_mode", "")) or "").strip().lower()
    prompt_encoder = str(cfg.get("prompt_encoder", cfg.get("prompt_encoder_kind", "")) or "").strip().lower()

    # Normalize common labels used in your experiments
    if prompt_encoder in {"use"}:
        prompt_encoder = "dan"  # your USE model is "dan"

    # ---- build embeddings ----
    map_vec = np.array([], dtype=np.float32)
    extent_refs: Dict[str, float] = {}

    if feature_mode in {"map_only", "prompt_plus_map"}:
        map_vec, extent_refs = embed_geojson_map(geojson)

    prompt_vec = np.array([], dtype=np.float32)
    if feature_mode in {"prompt_only", "prompt_plus_map"}:
        if not prompt_encoder:
            prompt_encoder = "dan"

        # ✅ FIX: derive data_dir robustly (no REPO_ROOT global)
        repo_root = _infer_repo_root_from_model_dir(model_dir)
        data_dir = repo_root / "data"

        prompt_vec = embed_prompt(
            prompt,
            encoder_kind=prompt_encoder,
            data_dir=data_dir,
        )

    # concat order must match training: X = [map | prompt]
    X_raw = np.hstack([map_vec, prompt_vec]).astype(np.float32, copy=False).reshape(1, -1)

    # ---- preprocess ----
    try:
        X_s = preproc.transform(X_raw)
    except Exception as e:
        raise RuntimeError(f"Preprocessor transform failed. X_raw shape={X_raw.shape}. Error: {e}") from e

    # ---- predict operator + confidence ----
    try:
        op_pred = classifier.predict(X_s)
        op = str(op_pred[0])
    except Exception as e:
        raise RuntimeError(f"Classifier.predict failed: {e}") from e

    confidence: Optional[float] = None
    try:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(X_s)
            confidence = float(np.max(proba))
    except Exception:
        confidence = None

    # ---- parameter regression via bundle ----
    regressors_by_class = _bundle_get(bundle, "regressors_by_class", None)
    if regressors_by_class is None:
        raise RuntimeError("Bundle missing 'regressors_by_class'.")

    if op not in regressors_by_class:
        # class label mismatch is unexpected; keep error explicit
        raise RuntimeError(f"No regressors found for predicted operator '{op}' in regressors_by_class.")

    dist_reg, area_reg = regressors_by_class[op]

    distance_ops = set(map(str, _as_list(_bundle_get(bundle, "distance_ops", []))))
    area_ops = set(map(str, _as_list(_bundle_get(bundle, "area_ops", []))))
    use_log1p = bool(_bundle_get(bundle, "use_log1p", False))

    # choose which regressor/denorm to use
    if op in area_ops:
        chosen = "area"
        reg = area_reg
        scale = float(extent_refs.get(EXTENT_AREA_COL, float("nan")))
    else:
        chosen = "distance"
        reg = dist_reg
        scale = float(extent_refs.get(EXTENT_DIAG_COL, float("nan")))

    # predict normalized param
    try:
        y_hat = reg.predict(X_s)
        param_norm = float(np.asarray(y_hat).reshape(-1)[0])
    except Exception as e:
        raise RuntimeError(f"Regressor.predict failed for operator '{op}' ({chosen}): {e}") from e

    # inverse log1p if training used it
    if use_log1p:
        param_norm = float(np.expm1(param_norm))

    # denormalize to get actual param_value
    if not np.isfinite(scale) or scale <= 0:
        param_value = param_norm
    else:
        param_value = float(param_norm * scale)

    # param_name: keep it simple & consistent (distance vs area)
    param_name = "distance" if chosen == "distance" else "area"

    return Prediction(
        operator=op,
        param_name=param_name,
        param_value=param_value,
        confidence=confidence,
    )
