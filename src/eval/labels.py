# src/eval/labels.py
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _norm_name(x: str) -> str:
    return str(x).strip().lower()


def build_name_lut(src_names: Sequence[str], dst_names: Sequence[str]) -> Dict[int, int]:
    """
    Build a lookup table mapping:
      src_index (position in src_names) -> dst_index (position in dst_names)
    """
    src = [_norm_name(x) for x in src_names]
    dst = [_norm_name(x) for x in dst_names]
    dst_pos = {name: j for j, name in enumerate(dst)}

    lut: Dict[int, int] = {}
    for i, s in enumerate(src):
        j = dst_pos.get(s)
        if j is not None:
            lut[i] = j
    return lut


def remap_to_bundle_order(
    op_idx: np.ndarray,
    *,
    src_names: Sequence[str],
    dst_names: Sequence[str],
    strict: bool = True,
) -> np.ndarray:
    """
    Remap integer labels from src_names order to dst_names order.

    op_idx values are assumed to be integer codes where:
      0 corresponds to src_names[0], 1 to src_names[1], etc.

    If strict=True:
      - raise if any code is out of range for src_names
      - raise if any src class name is not found in dst_names
    If strict=False:
      - invalid or unmapped codes become -1
    """
    op_idx = np.asarray(op_idx, dtype=int).reshape(-1)
    lut = build_name_lut(src_names, dst_names)

    out = np.full(op_idx.shape, -1, dtype=int)

    # validate bounds
    bad_oob = np.where((op_idx < 0) | (op_idx >= len(src_names)))[0]
    if strict and bad_oob.size > 0:
        example = bad_oob[:10].tolist()
        raise ValueError(
            "op_idx contains out-of-range label codes for src_names. "
            f"src_names size={len(src_names)}. Example bad positions: {example}."
        )

    for i, code in enumerate(op_idx):
        if code < 0 or code >= len(src_names):
            continue
        out[i] = lut.get(int(code), -1)

    if strict and not (out >= 0).all():
        bad = np.where(out < 0)[0][:10].tolist()
        raise ValueError(
            "Class-name mismatch between src_names and dst_names. "
            f"Example bad positions: {bad}. "
            "Ensure FIXED_CLASSES / bundle class_names are consistent."
        )

    return out
