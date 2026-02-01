# src/imgofup/webapp/services/inference_service.py
from __future__ import annotations

from typing import Any, Dict

from imgofup.webapp.schemas import Prediction
from imgofup.webapp.services.model_registry import ModelHandle


def predict_operator_and_param(
    model: ModelHandle,
    prompt: str,
    geojson: Dict[str, Any],
) -> Prediction:
    """
    Predict the generalization operator + parameter/value from (prompt, geojson)
    using the selected trained model.

    This file is intentionally a thin wrapper so you can plug in your *real*
    thesis pipeline without changing the web API.

    Expected future behavior:
      - parse/validate geojson (FeatureCollection)
      - compute map embedding (hand-crafted features or graph embedding)
      - compute text embedding (sentence encoder / LLM embedding)
      - run MLP forward pass
      - decode outputs into {operator, param_name, param_value, confidence}

    For now, it provides a deterministic placeholder so the app works end-to-end.
    Replace the placeholder section with your real inference call.
    """
    # -----------------------------
    # TEMP PLACEHOLDER IMPLEMENTATION
    # -----------------------------
    text = (prompt or "").lower().strip()

    # optionally, you can use model.config to vary behavior across model types:
    # model_type = (model.config.get("type") or "").lower()

    if "simpl" in text:
        return Prediction(
            operator="simplification",
            param_name="tolerance",
            param_value=10.0,
            confidence=0.5,
        )

    if any(k in text for k in ["aggreg", "merge", "cluster"]):
        return Prediction(
            operator="aggregation",
            param_name="distance",
            param_value=25.0,
            confidence=0.5,
        )

    if any(k in text for k in ["displac", "move"]):
        return Prediction(
            operator="displacement",
            param_name="min_sep",
            param_value=5.0,
            confidence=0.5,
        )

    return Prediction(
        operator="unknown",
        param_name=None,
        param_value=None,
        confidence=0.2,
    )


# -----------------------------------------------------------------------------
# Optional helper hook (useful once you integrate real inference)
# -----------------------------------------------------------------------------
def decode_model_output(
    raw_output: Any,
) -> Prediction:
    """
    Hook for decoding your model's raw output (logits, dict, tuple, etc.)
    into the standardized Prediction schema.

    You can delete this if you don't need it, but it's a nice place to keep
    operator/parameter decoding logic centralized.

    Expected patterns you might implement here later:
      - argmax over operator classes
      - regression head for param_value
      - mapping operator -> allowed param_name
      - confidence extraction / calibration
    """
    raise NotImplementedError("Implement decode_model_output(...) when wiring the real model.")
