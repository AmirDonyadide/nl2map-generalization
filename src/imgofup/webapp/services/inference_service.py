# src/imgofup/webapp/services/inference_service.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Set

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
# Preproc handling
# -----------------------------------------------------------------------------
def _expected_dim_from_preproc_meta(preproc_raw: Any, feature_mode: str) -> Optional[int]:
    """
    If preproc_raw is a metadata dict that contains map_dim/prompt_dim, infer expected X dimension.
    """
    if not isinstance(preproc_raw, dict):
        return None

    map_dim = preproc_raw.get("map_dim", None)
    prompt_dim = preproc_raw.get("prompt_dim", None)
    if map_dim is None or prompt_dim is None:
        return None

    try:
        map_dim_i = int(map_dim)
        prompt_dim_i = int(prompt_dim)
    except Exception:
        return None

    fm = (feature_mode or "").strip().lower()
    if fm == "prompt_only":
        return prompt_dim_i
    if fm == "map_only":
        return map_dim_i
    if fm == "prompt_plus_map":
        return map_dim_i + prompt_dim_i
    return None


def _apply_map_preproc_if_present(preproc_raw: Any, map_vec: np.ndarray) -> np.ndarray:
    """
    Your preproc.joblib for map-based models contains a dict with 'map_preproc' which is fitted.
    We apply it if it exposes .transform().
    """
    if map_vec.size == 0:
        return map_vec
    if not isinstance(preproc_raw, dict):
        return map_vec

    mp = preproc_raw.get("map_preproc", None)
    if mp is None:
        return map_vec

    if hasattr(mp, "transform"):
        try:
            out = mp.transform(map_vec.reshape(1, -1))
            return np.asarray(out, dtype=np.float32).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"map_preproc.transform failed: {e}") from e

    return map_vec


# -----------------------------------------------------------------------------
# Artifact format handling: classifier + regressors
# -----------------------------------------------------------------------------
def _pick_classifier(bundle: Any, model_dir: Path, classifier_name: str) -> Any:
    """
    Your audit shows classifier.joblib is a dict (not sklearn). The real classifier is in bundle['classifier'].
    So we prefer bundle classifier and only accept classifier.joblib if it has .predict().
    """
    clf = _bundle_get(bundle, "classifier", None)
    if clf is not None and hasattr(clf, "predict"):
        return clf

    cand_path = model_dir / classifier_name
    if cand_path.exists():
        cand = _load_joblib(str(cand_path))
        if hasattr(cand, "predict"):
            return cand

    raise RuntimeError("No usable classifier found. Expected bundle.classifier to be an sklearn estimator.")


def _unpack_regressor_entry(entry: Any, op: str) -> Tuple[Any, Optional[Any]]:
    """
    regressors_by_class[op] is typically:
      [MLPRegressor, StandardScaler]   # scaler is for y (target)
    Return (regressor, y_scaler).
    """
    if isinstance(entry, (list, tuple)):
        if len(entry) == 2:
            reg, y_scaler = entry
            return reg, y_scaler
        if len(entry) == 1:
            return entry[0], None
    return entry, None


def _predict_param_from_regressor(reg: Any, y_scaler: Optional[Any], X_s: Any, op: str) -> float:
    """
    1) reg.predict(X_s)
    2) if y_scaler has inverse_transform -> apply it (y-space)
    """
    if not hasattr(reg, "predict"):
        raise RuntimeError(f"Regressor for '{op}' has no .predict(). Got type={type(reg)}")

    try:
        y_hat = reg.predict(X_s)
        y_hat_arr = np.asarray(y_hat, dtype=np.float64).reshape(-1, 1)
    except Exception as e:
        raise RuntimeError(f"Regressor.predict failed for operator '{op}': {e}") from e

    if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
        try:
            y_hat_arr = np.asarray(y_scaler.inverse_transform(y_hat_arr), dtype=np.float64).reshape(-1, 1)
        except Exception as e:
            raise RuntimeError(f"Target inverse scaling failed for operator '{op}': {e}") from e

    return float(y_hat_arr.reshape(-1)[0])


def _get_area_distance_ops(cfg: Dict[str, Any], bundle: Any) -> Tuple[Set[str], Set[str]]:
    """
    Determine which operators use distance vs area for denormalization.

    Preference:
      1) cfg['distance_ops'] / cfg['area_ops'] (if you add them in config.json)
      2) bundle['distance_ops'] / bundle['area_ops'] (currently None in your artifacts)
      3) fallback defaults aligned with your current webapp operators:
         - distance: simplify, aggregate, displace
         - area:     select
    """
    cfg_dist = set(map(str, _as_list(cfg.get("distance_ops", []))))
    cfg_area = set(map(str, _as_list(cfg.get("area_ops", []))))
    if cfg_dist or cfg_area:
        return {s.strip().lower() for s in cfg_dist}, {s.strip().lower() for s in cfg_area}

    b_dist_raw = _bundle_get(bundle, "distance_ops", None)
    b_area_raw = _bundle_get(bundle, "area_ops", None)
    b_dist = set(map(str, _as_list(b_dist_raw))) if b_dist_raw else set()
    b_area = set(map(str, _as_list(b_area_raw))) if b_area_raw else set()
    if b_dist or b_area:
        return {s.strip().lower() for s in b_dist}, {s.strip().lower() for s in b_area}

    # ✅ IMPORTANT: match your operator implementations in generalize_service.py
    return {"simplify", "aggregate", "displace"}, {"select"}


def _normalize_op_label(raw_label: Any, class_names: Optional[list]) -> str:
    """
    Convert classifier output to a normalized operator string.
    Handles:
      - string labels ("simplify", ...)
      - numeric indices (0..K-1) that map into class_names
      - numeric strings ("0","1",...) that map into class_names
    """
    if isinstance(raw_label, (int, np.integer)) or (isinstance(raw_label, str) and raw_label.isdigit()):
        if not class_names:
            raise RuntimeError(
                f"Classifier predicted numeric label '{raw_label}' but no class_names found in bundle/meta."
            )
        idx = int(raw_label)
        if idx < 0 or idx >= len(class_names):
            raise RuntimeError(
                f"Classifier predicted class index {idx} out of range for class_names (len={len(class_names)})."
            )
        return str(class_names[idx]).strip().lower()

    return str(raw_label).strip().lower()


# -----------------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------------
def predict_operator_and_param(
    model: ModelHandle,
    prompt: str,
    geojson: Dict[str, Any],
) -> Prediction:
    """
    Compatible with your current model artifacts.

    Artifacts (as per your audit):
      - preproc.joblib: dict (metadata only OR includes 'map_preproc')
      - classifier.joblib: dict (NOT usable)
      - cls_plus_regressors.joblib: dict with:
          * classifier (sklearn estimator)
          * regressors_by_class: op -> [regressor, y_scaler]
          * class_names
          * use_log1p (optional)
      - classifier_meta.json: contains class_names and feature_mode

    Steps:
      1) Embed map/prompt based on config.feature_mode and config.prompt_encoder
      2) Apply map_preproc if present (map models)
      3) X_raw = [map | prompt]
      4) Sanity-check X dimension if preproc has metadata dims
      5) classifier -> operator (mapped via class_names if needed)
      6) regressor -> param_norm (inverse y scaling if scaler exists)
      7) expm1 if use_log1p
      8) denormalize by extent diag (distance) or extent area (area)
    """
    model_dir = Path(model.model_dir)
    cfg = model.config or {}

    artifacts = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts", {}), dict) else {}
    preproc_name = artifacts.get("preprocessor", "preproc.joblib")
    meta_name = artifacts.get("meta", "classifier_meta.json")
    bundle_name = artifacts.get("regressors", "cls_plus_regressors.joblib")
    classifier_name = artifacts.get("classifier", "classifier.joblib")

    preproc_path = _resolve_artifact(model_dir, preproc_name)
    meta_path = _resolve_artifact(model_dir, meta_name)
    bundle_path = _resolve_artifact(model_dir, bundle_name)

    preproc_raw = _load_joblib(str(preproc_path))
    meta = _load_json(str(meta_path))
    bundle = _load_joblib(str(bundle_path))

    classifier = _pick_classifier(bundle=bundle, model_dir=model_dir, classifier_name=classifier_name)

    feature_mode = str(cfg.get("feature_mode", meta.get("feature_mode", "")) or "").strip().lower()
    prompt_encoder = str(cfg.get("prompt_encoder", cfg.get("prompt_encoder_kind", "")) or "").strip().lower()

    # normalize encoder labels used in your repo
    if prompt_encoder == "use":
        prompt_encoder = "dan"  # your USE model key is "dan"

    # ---- embeddings ----
    map_vec = np.array([], dtype=np.float32)
    extent_refs: Dict[str, float] = {}

    if feature_mode in {"map_only", "prompt_plus_map"}:
        map_vec, extent_refs = embed_geojson_map(geojson)
        map_vec = _apply_map_preproc_if_present(preproc_raw, map_vec)

    prompt_vec = np.array([], dtype=np.float32)
    if feature_mode in {"prompt_only", "prompt_plus_map"}:
        if not prompt_encoder:
            prompt_encoder = "dan"
        repo_root = _infer_repo_root_from_model_dir(model_dir)
        data_dir = repo_root / "data"
        prompt_vec = embed_prompt(prompt, encoder_kind=prompt_encoder, data_dir=data_dir)

    X_raw = np.hstack([map_vec, prompt_vec]).astype(np.float32, copy=False).reshape(1, -1)

    expected_dim = _expected_dim_from_preproc_meta(preproc_raw, feature_mode)
    if expected_dim is not None and X_raw.shape[1] != expected_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: got X_raw.shape[1]={X_raw.shape[1]} but expected {expected_dim} "
            f"(feature_mode={feature_mode})."
        )

    # No global X-transformer exists in your artifacts; map preproc already applied
    X_s = X_raw

    # ---- operator prediction ----
    try:
        op_pred = classifier.predict(X_s)
        raw_label = op_pred[0]
    except Exception as e:
        raise RuntimeError(f"Classifier.predict failed: {e}") from e

    class_names = _bundle_get(bundle, "class_names", None) or meta.get("class_names", None)
    op = _normalize_op_label(raw_label, class_names)

    # ---- confidence ----
    confidence: Optional[float] = None
    try:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(X_s)
            confidence = float(np.max(proba))
    except Exception:
        confidence = None

    # ---- regressors ----
    regressors_by_class = _bundle_get(bundle, "regressors_by_class", None)
    if not isinstance(regressors_by_class, dict):
        raise RuntimeError("Bundle missing or invalid 'regressors_by_class' (expected dict).")

    if op not in regressors_by_class:
        # Helpful debugging: show available keys
        keys_preview = ", ".join(sorted(map(str, regressors_by_class.keys())))
        raise RuntimeError(
            f"No regressors found for predicted operator '{op}' in regressors_by_class. "
            f"Available: [{keys_preview}]"
        )

    reg_entry = regressors_by_class[op]
    reg, y_scaler = _unpack_regressor_entry(reg_entry, op=op)
    param_norm = _predict_param_from_regressor(reg, y_scaler, X_s, op=op)

    use_log1p = bool(_bundle_get(bundle, "use_log1p", False))
    if use_log1p:
        param_norm = float(np.expm1(param_norm))

    # ---- denormalization (distance vs area) ----
    distance_ops, area_ops = _get_area_distance_ops(cfg, bundle)

    if op in area_ops:
        chosen = "area"
        scale = float(extent_refs.get(EXTENT_AREA_COL, float("nan")))
    else:
        chosen = "distance"
        scale = float(extent_refs.get(EXTENT_DIAG_COL, float("nan")))

    if not np.isfinite(scale) or scale <= 0:
        param_value = float(param_norm)
    else:
        param_value = float(param_norm * scale)

    param_name = "distance" if chosen == "distance" else "area"

    return Prediction(
        operator=op,
        param_name=param_name,
        param_value=param_value,
        confidence=confidence,
    )
