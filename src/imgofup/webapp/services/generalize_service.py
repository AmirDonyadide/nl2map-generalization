# src/imgofup/webapp/services/generalize_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def apply_generalization(
    geojson: Dict[str, Any],
    operator: str,
    param_name: Optional[str],
    param_value: Optional[float],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply a generalization operator to an input GeoJSON FeatureCollection.

    This is a thin wrapper around your existing generalization code in
    `src/imgofup/generalization/*`.

    For now, it returns the input unchanged (so the web app pipeline works).
    Replace the placeholder section with calls to your real operators.

    Returns:
      (output_geojson, warnings)
    """
    warnings: List[str] = []

    # -----------------------------
    # TEMP PLACEHOLDER IMPLEMENTATION
    # -----------------------------
    if operator == "unknown":
        warnings.append("Operator could not be inferred reliably; returning input unchanged.")
        return geojson, warnings

    warnings.append(
        f"Generalization not yet wired (operator={operator}, {param_name}={param_value}); returning input unchanged."
    )
    return geojson, warnings


# -----------------------------------------------------------------------------
# Optional future helpers (recommended once you wire real geometry ops)
# -----------------------------------------------------------------------------
def validate_geojson_featurecollection(geojson: Dict[str, Any]) -> None:
    """
    Basic structural check for a GeoJSON FeatureCollection.
    Raise ValueError if invalid.

    You can call this at the start of apply_generalization once you're ready.
    """
    if not isinstance(geojson, dict):
        raise ValueError("geojson must be a dict.")
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson.type must be 'FeatureCollection'.")
    feats = geojson.get("features")
    if not isinstance(feats, list):
        raise ValueError("geojson.features must be a list.")


def operator_supported(operator: str) -> bool:
    """
    Keep a central list of supported operators once your implementation is ready.
    """
    return operator in {
        "simplification",
        "aggregation",
        "displacement",
        # add more as you implement them
    }
