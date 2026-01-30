from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


def validate_regressors_bundle_inputs(
    *,
    class_names: Sequence[str],
    regressors_by_class: Mapping[str, Tuple[Any, Any]],
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
) -> None:
    cls_set = {str(c) for c in class_names}
    reg_keys = {str(k) for k in regressors_by_class.keys()}

    unknown = sorted(reg_keys - cls_set)
    if unknown:
        raise ValueError(f"regressors_by_class contains keys not in class_names: {unknown}")

    # optional sanity: distance/area ops should be subsets of class_names
    dist_bad = sorted({str(x) for x in distance_ops} - cls_set)
    area_bad = sorted({str(x) for x in area_ops} - cls_set)
    if dist_bad or area_bad:
        raise ValueError(f"distance_ops/area_ops contain unknown classes: dist={dist_bad}, area={area_bad}")

    overlap = sorted(set(map(str, distance_ops)) & set(map(str, area_ops)))
    if overlap:
        raise ValueError(f"distance_ops and area_ops overlap: {overlap}")


def build_cls_plus_regressors_bundle(
    *,
    classifier: Any,
    regressors_by_class: Mapping[str, Tuple[Any, Any]],
    class_names: Sequence[str],
    use_log1p: bool,
    cv_summary: Dict[str, Any],
    distance_ops: Sequence[str],
    area_ops: Sequence[str],
    diag_col: str,
    area_col: str,
) -> Dict[str, Any]:
    return {
        "bundle_version": 1,
        "classifier": classifier,
        "regressors_by_class": dict(regressors_by_class),
        "class_names": [str(x) for x in class_names],
        "use_log1p": bool(use_log1p),
        "target": "param_norm",
        "normalization": {
            "type": "dynamic_extent",
            "distance_ops": list(distance_ops),
            "area_ops": list(area_ops),
            "distance_ref_col": str(diag_col),
            "area_ref_col": str(area_col),
        },
        "cv_summary": cv_summary,
    }


def resolve_bundle_path(out_dir: Path, *, exp_name: str, save_name: str | None) -> Path:
    fname = save_name or f"{exp_name}__cls_plus_regressors.joblib"
    return out_dir / fname
