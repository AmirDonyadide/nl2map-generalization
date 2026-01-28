# src/train/save_bundle.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib


@dataclass(frozen=True)
class BundleSaveResult:
    bundle_path: str


def save_cls_plus_regressors_bundle(
    *,
    exp_name: str,
    out_dir: Path,
    classifier: Any,
    regressors_by_class: Mapping[str, Tuple[Any, Any]],
    class_names: Sequence[str],
    use_log1p: bool,
    cv_summary: Dict[str, Any],
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
    diag_col: str = "extent_diag_m",
    area_col: str = "extent_area_m2",
    save_name: str | None = None,  # default: {exp_name}__cls_plus_regressors.joblib
) -> BundleSaveResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "classifier": classifier,
        "regressors_by_class": dict(regressors_by_class),
        "class_names": [str(x) for x in class_names],
        "use_log1p": bool(use_log1p),
        "target": "param_norm",
        "normalization": {
            "type": "dynamic_extent",
            "distance_ops": list(distance_ops),
            "area_ops": list(area_ops),
            "distance_ref_col": diag_col,
            "area_ref_col": area_col,
        },
        "cv_summary": cv_summary,
    }

    fname = save_name or f"{exp_name}__cls_plus_regressors.joblib"
    bundle_path = out_dir / fname

    joblib.dump(bundle, bundle_path)

    return BundleSaveResult(bundle_path=str(bundle_path))
