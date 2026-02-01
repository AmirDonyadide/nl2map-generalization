# src/imgofup/webapp/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter

from imgofup.webapp.schemas import (
    GeneralizeRequest,
    GeneralizeResponse,
    ModelInfo,
    Prediction,
)
from imgofup.webapp.services.model_registry import ModelHandle, list_models, load_model
from imgofup.webapp.services.inference_service import predict_operator_and_param
from imgofup.webapp.services.generalize_service import apply_generalization


router = APIRouter(prefix="/api", tags=["api"])

# -----------------------------------------------------------------------------
# Router factory (lets app.py pass paths cleanly)
# -----------------------------------------------------------------------------
def create_api_router(models_dir: Path) -> APIRouter:
    """
    Returns a router with access to the models directory via closure.
    Keeps api.py reusable and testable.
    """

    @router.get("/models", response_model=List[ModelInfo])
    def get_models() -> List[ModelInfo]:
        return list_models(models_dir)

    @router.post("/predict", response_model=Prediction)
    def predict(req: GeneralizeRequest) -> Prediction:
        model = load_model(models_dir, req.model_id)
        return predict_operator_and_param(model=model, prompt=req.prompt, geojson=req.geojson)

    @router.post("/generalize", response_model=GeneralizeResponse)
    def generalize(req: GeneralizeRequest) -> GeneralizeResponse:
        model = load_model(models_dir, req.model_id)

        pred = predict_operator_and_param(model=model, prompt=req.prompt, geojson=req.geojson)

        out_geojson, warnings = apply_generalization(
            geojson=req.geojson,
            operator=pred.operator,
            param_name=pred.param_name,
            param_value=pred.param_value,
        )

        return GeneralizeResponse(
            prediction=pred,
            output_geojson=out_geojson,
            warnings=warnings,
        )

    return router
