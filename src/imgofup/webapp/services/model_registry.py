# src/imgofup/webapp/services/model_registry.py
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from imgofup.webapp.schemas import ModelInfo


@dataclass(frozen=True)
class ModelHandle:
    """
    Handle returned by `load_model(...)`.

    Includes:
      - model folder + config
      - resolved artifact paths (validated)
    """
    model_id: str
    model_dir: Path
    config: Dict[str, Any]
    artifacts: Dict[str, Path]


# -----------------------------
# Helpers
# -----------------------------
_DEFAULT_ARTIFACTS = {
    "preprocessor": "preproc.joblib",
    "classifier": "classifier.joblib",
    "regressors": "cls_plus_regressors.joblib",
    "meta": "classifier_meta.json",
    "config": "config.json",
}


def _safe_read_json(p: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _resolve_artifact(model_dir: Path, rel: str) -> Path:
    p = (model_dir / rel).resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _normalize_model_id(model_id: str) -> str:
    return str(model_id).strip()


# -----------------------------
# Public API
# -----------------------------
def list_models(models_dir: Path) -> List[ModelInfo]:
    """
    Discover models under: <repo>/models/<model_id>/

    Convention:
      - each model is a subfolder under models/
      - config.json provides name/description (optional but recommended)
    """
    if not models_dir.exists():
        return []

    items: List[ModelInfo] = []
    for model_dir in sorted([d for d in models_dir.iterdir() if d.is_dir()]):
        model_id = model_dir.name
        cfg_path = model_dir / _DEFAULT_ARTIFACTS["config"]

        name = model_id
        desc: Optional[str] = None

        if cfg_path.exists():
            cfg = _safe_read_json(cfg_path)
            name = str(cfg.get("name", name))
            desc = cfg.get("description", desc)

        items.append(ModelInfo(id=model_id, name=name, description=desc))

    return items


@lru_cache(maxsize=64)
def load_model(models_dir: Path, model_id: str) -> ModelHandle:
    """
    Load a model by id and validate that the required artifacts exist.

    This does NOT load joblib objects yet; inference_service loads and caches those
    (so model_registry stays lightweight and fast).

    Required (by default):
      - config.json (recommended; if missing, we still work with defaults)
      - preproc.joblib
      - cls_plus_regressors.joblib
      - classifier_meta.json
      - classifier.joblib (optional; bundle usually includes classifier, but we validate if present)
    """
    model_id = _normalize_model_id(model_id)
    model_dir = (Path(models_dir) / model_id).resolve()

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    # Read config (optional)
    cfg_path = model_dir / _DEFAULT_ARTIFACTS["config"]
    config: Dict[str, Any] = _safe_read_json(cfg_path) if cfg_path.exists() else {}

    artifacts_cfg = config.get("artifacts", {})
    if not isinstance(artifacts_cfg, dict):
        artifacts_cfg = {}

    # Resolve artifacts using config overrides (fallback to defaults)
    def _name(key: str) -> str:
        return str(artifacts_cfg.get(key, _DEFAULT_ARTIFACTS[key]))

    required_keys = ["preprocessor", "regressors", "meta"]
    optional_keys = ["classifier"]  # optional, because bundle often contains it

    artifacts: Dict[str, Path] = {}

    # Required
    missing: List[str] = []
    for key in required_keys:
        rel = _name(key)
        try:
            artifacts[key] = _resolve_artifact(model_dir, rel)
        except FileNotFoundError:
            missing.append(f"{key} -> {rel}")

    # Optional
    for key in optional_keys:
        rel = _name(key)
        p = (model_dir / rel).resolve()
        if p.exists():
            artifacts[key] = p

    # Always include config path if it exists (useful for debugging)
    if cfg_path.exists():
        artifacts["config"] = cfg_path

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model '{model_id}' is missing required artifacts in {str(model_dir)}: "
                + ", ".join(missing)
            ),
        )

    return ModelHandle(model_id=model_id, model_dir=model_dir, config=config, artifacts=artifacts)


def clear_model_cache() -> None:
    """
    Useful during development if you change files under models/<id>/.
    """
    load_model.cache_clear()
