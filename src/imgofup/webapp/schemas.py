# src/imgofup/webapp/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GeneralizeRequest(BaseModel):
    """
    Request payload for both /api/predict and /api/generalize.
    """
    model_id: str = Field(..., description="Identifier of the selected trained model")
    prompt: str = Field(..., description="User prompt describing the desired generalization")
    geojson: Dict[str, Any] = Field(..., description="Input GeoJSON FeatureCollection")


class Prediction(BaseModel):
    """
    Inference result decoded from the model output.
    """
    operator: str = Field(..., description="Predicted generalization operator (e.g., simplification)")
    param_name: Optional[str] = Field(
        default=None,
        description="Name of the predicted parameter (e.g., tolerance, distance).",
    )
    param_value: Optional[float] = Field(
        default=None,
        description="Predicted numeric parameter value.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence/probability score for the prediction.",
    )


class GeneralizeResponse(BaseModel):
    """
    Response payload for /api/generalize:
    includes prediction + generalized output geojson.
    """
    prediction: Prediction
    output_geojson: Dict[str, Any] = Field(..., description="Generalized GeoJSON FeatureCollection")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g., invalid geometry fixed).",
    )


class ModelInfo(BaseModel):
    """
    Metadata shown in the frontend model dropdown.
    """
    id: str = Field(..., description="Model identifier (folder name under /models)")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(default=None, description="Optional longer model description")
