#src/train/save_bundle.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib

from .utils._bundle_utils import (
    build_cls_plus_regressors_bundle,
    validate_regressors_bundle_inputs,
    resolve_bundle_path,
)


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
    save_name: str | None = None,
) -> BundleSaveResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_regressors_bundle_inputs(
        class_names=class_names,
        regressors_by_class=regressors_by_class,
        distance_ops=distance_ops,
        area_ops=area_ops,
    )

    bundle = build_cls_plus_regressors_bundle(
        classifier=classifier,
        regressors_by_class=regressors_by_class,
        class_names=class_names,
        use_log1p=use_log1p,
        cv_summary=cv_summary,
        distance_ops=distance_ops,
        area_ops=area_ops,
        diag_col=diag_col,
        area_col=area_col,
    )

    bundle_path = resolve_bundle_path(out_dir, exp_name=exp_name, save_name=save_name)
    joblib.dump(bundle, bundle_path)

    return BundleSaveResult(bundle_path=str(bundle_path))
