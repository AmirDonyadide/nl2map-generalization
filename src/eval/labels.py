# src/eval/labels.py
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def build_name_lut(src_names: Sequence[str], dst_names: Sequence[str]) -> Dict[int, int]:
    src = [str(x) for x in src_names]
    dst = [str(x) for x in dst_names]
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

    Parameters
    ----------
    op_idx : np.ndarray
        Integer labels (e.g., produced by pd.Categorical(...).codes) using src_names ordering.
    src_names : Sequence[str]
        The ordering used to create op_idx.
    dst_names : Sequence[str]
        The desired ordering (e.g., bundle["class_names"]).
    strict : bool
        If True, raise if any label cannot be remapped (mismatch / missing class).
        If False, unmapped labels become -1.

    Returns
    -------
    np.ndarray
        Remapped labels in dst order.
    """
    op_idx = np.asarray(op_idx).astype(int, copy=False).reshape(-1)
    lut = build_name_lut(src_names, dst_names)

    out = np.full_like(op_idx, -1)
    for i, k in enumerate(op_idx):
        out[i] = lut.get(int(k), -1)

    if strict and not (out >= 0).all():
        bad = np.where(out < 0)[0][:10].tolist()
        raise ValueError(
            "Class-name mismatch between src_names and dst_names. "
            f"Example bad indices: {bad}. "
            "Ensure FIXED_CLASSES / bundle class_names are consistent."
        )

    return out
