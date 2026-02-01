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
    A lightweight handle returned by `load_model(...)`.

    For now it only stores where the model lives and its config.
    Later you can extend this to hold:
      - loaded sklearn pipeline
      - torch model + tokenizer
      - embedding configs
      - normalization stats, etc.
    """
    model_id: str
    model_dir: Path
    config: Dict[str, Any]


def list_models(models_dir: Path) -> List[ModelInfo]:
    """
    Discover models under: <repo>/models/<model_id>/

    Convention:
      - each model has its own folder under models/
      - optional models/<model_id>/config.json provides "name" and "description"
    """
    if not models_dir.exists():
        return []

    items: List[ModelInfo] = []
    for model_dir in sorted([d for d in models_dir.iterdir() if d.is_dir()]):
        model_id = model_dir.name
        cfg_path = model_dir / "config.json"

        name = model_id
        desc: Optional[str] = None

        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                name = str(cfg.get("name", name))
                desc = cfg.get("description", desc)
            except Exception:
                # Keep model listing resilient (broken config shouldn't kill the app)
                pass

        items.append(ModelInfo(id=model_id, name=name, description=desc))

    return items


@lru_cache(maxsize=32)
def load_model(models_dir: Path, model_id: str) -> ModelHandle:
    """
    Load a model by id.

    Current behavior (safe default):
      - validates the directory exists
      - reads config.json if present
      - returns a ModelHandle (cached)

    Later:
      - actually load weights/artifacts here (torch / sklearn / tf)
      - store them inside ModelHandle or a separate object referenced by it
    """
    model_dir = models_dir / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    cfg_path = model_dir / "config.json"
    config: Dict[str, Any] = {}

    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                config = {}
        except Exception:
            # If config is broken, keep going but store empty config.
            config = {}

    return ModelHandle(model_id=model_id, model_dir=model_dir, config=config)


def clear_model_cache() -> None:
    """
    Useful during development if you change files under models/<id>/.
    You can call this from a dev-only endpoint if you want.
    """
    load_model.cache_clear()
