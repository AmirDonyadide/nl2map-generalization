# src/imgofup/webapp/api.py
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from imgofup.webapp.schemas import (
    GeneralizeRequest,
    GeneralizeResponse,
    ModelInfo,
    Prediction,
)
from imgofup.webapp.services.model_registry import list_models, load_model
from imgofup.webapp.services.inference_service import predict_operator_and_param
from imgofup.webapp.services.generalize_service import apply_generalization


# -----------------------------------------------------------------------------
# Router factory (lets app.py pass paths cleanly)
# -----------------------------------------------------------------------------
def create_api_router(models_dir: Path) -> APIRouter:
    """
    Returns a NEW router with access to the models directory via closure.
    (Important: do NOT reuse a global router here, otherwise reloads/tests can duplicate routes.)
    """
    router = APIRouter(prefix="/api", tags=["api"])

    @router.get("/models", response_model=List[ModelInfo])
    def get_models() -> List[ModelInfo]:
        return list_models(models_dir)

    @router.post("/predict", response_model=Prediction)
    def predict(req: GeneralizeRequest) -> Prediction:
        try:
            model = load_model(models_dir, req.model_id)
            return predict_operator_and_param(model=model, prompt=req.prompt, geojson=req.geojson)
        except FileNotFoundError as e:
            # missing artifacts / missing model folder
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (ValueError, TypeError) as e:
            # incompatible formats, bad config, etc.
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            # inference pipeline issues (dims mismatch, missing classifier in bundle, etc.)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            # unexpected
            raise HTTPException(status_code=500, detail=f"Unexpected error during /predict: {e}") from e

    @router.post("/generalize", response_model=GeneralizeResponse)
    def generalize(req: GeneralizeRequest) -> GeneralizeResponse:
        try:
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

        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            # includes inference errors and "no regressors for class" etc.
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error during /generalize: {e}") from e

    return router
