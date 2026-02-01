# src/imgofup/userstudy/operators.py
from __future__ import annotations

from typing import Literal, Union, cast
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.strtree import STRtree
from shapely.ops import nearest_points


JoinStyle = Literal["round", "mitre", "bevel"]
CapStyle = Literal["round", "flat", "square"]


def _coerce_join_style(v: Union[int, str]) -> JoinStyle:
    """
    Shapely 2 prefers string join styles.
    Accepts legacy int codes:
      1 -> round, 2 -> mitre, 3 -> bevel
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("round", "mitre", "bevel"):
            return cast(JoinStyle, s)
        raise ValueError(
            f"Invalid join_style {v!r}. Use 'round'|'mitre'|'bevel'."
        )

    m = {1: "round", 2: "mitre", 3: "bevel"}
    return cast(JoinStyle, m.get(int(v), "mitre"))


def _coerce_cap_style(v: Union[int, str]) -> CapStyle:
    """
    Shapely 2 prefers string cap styles.
    Accepts legacy int codes:
      1 -> round, 2 -> flat, 3 -> square
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("round", "flat", "square"):
            return cast(CapStyle, s)
        raise ValueError(
            f"Invalid cap_style {v!r}. Use 'round'|'flat'|'square'."
        )

    m = {1: "round", 2: "flat", 3: "square"}
    return cast(CapStyle, m.get(int(v), "flat"))


def aggregate_buildings(
    gdf: gpd.GeoDataFrame,
    *,
    dist: float,
    join_style: Union[int, JoinStyle] = "mitre",
    mitre_limit: float = 5.0,
    cap_style: Union[int, CapStyle] = "flat",
    resolution: int = 1,
) -> gpd.GeoDataFrame:
    """
    Aggregation operator:
      buffer(+dist) → dissolve → buffer(-dist)

    Notes
    -----
    - Uses Shapely buffer() join/cap styles in a type-safe way.
    - Accepts both legacy int codes and modern string styles.
    """
    if gdf is None or gdf.empty:
        return gdf.copy() if gdf is not None else gpd.GeoDataFrame(geometry=[], crs=None)

    dist = float(dist)
    if dist <= 0:
        out = gdf.copy()
        out = out[~out.geometry.is_empty & out.geometry.is_valid]
        return out.reset_index(drop=True)

    js = _coerce_join_style(join_style)
    cs = _coerce_cap_style(cap_style)

    buff = gdf.geometry.buffer(
        dist,
        join_style=js,
        mitre_limit=float(mitre_limit),
        cap_style=cs,
        resolution=int(resolution),
    )

    merged = unary_union(buff)
    out = gpd.GeoDataFrame(geometry=[merged], crs=gdf.crs)

    out["geometry"] = out.geometry.buffer(
        -dist,
        join_style=js,
        mitre_limit=float(mitre_limit),
        cap_style=cs,
        resolution=int(resolution),
    )

    out = out[~out.geometry.is_empty & out.geometry.is_valid]
    out = out.explode(index_parts=False).reset_index(drop=True)
    return out


# (keep your simplify/select/displace as-is below)
def simplify_buildings(gdf: gpd.GeoDataFrame, *, eps: float) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf.copy() if gdf is not None else gpd.GeoDataFrame(geometry=[], crs=None)

    eps = float(eps)
    if eps <= 0:
        out = gdf.copy()
        out = out[~out.geometry.is_empty & out.geometry.is_valid]
        return out.reset_index(drop=True)

    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(eps, preserve_topology=True)
    out = out[~out.geometry.is_empty & out.geometry.is_valid]
    return out.reset_index(drop=True)


def select_buildings(gdf: gpd.GeoDataFrame, *, area_threshold: float) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf.copy() if gdf is not None else gpd.GeoDataFrame(geometry=[], crs=None)

    thr = float(area_threshold)
    if thr <= 0:
        out = gdf.copy()
        out = out[~out.geometry.is_empty & out.geometry.is_valid]
        return out.reset_index(drop=True)

    out = gdf.copy()
    out = out[out.geometry.area >= thr]
    out = out[~out.geometry.is_empty & out.geometry.is_valid]
    return out.reset_index(drop=True)


def displace_buildings(
    gdf: gpd.GeoDataFrame,
    *,
    clearance: float,
    iters: int = 40,
    step: float = 1.2,
    max_total: float = 10.0,
    small_moves_more: bool = True,
    area_ref: float = 120.0,
) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf.copy() if gdf is not None else gpd.GeoDataFrame(geometry=[], crs=None)

    clearance = float(clearance)
    if clearance <= 0:
        out = gdf.copy()
        out = out[~out.geometry.is_empty & out.geometry.is_valid]
        return out.reset_index(drop=True)

    out = gdf.copy()
    geoms = list(out.geometry.values)

    areas = np.maximum(1.0, np.array([g.area for g in geoms], dtype=float))
    if small_moves_more:
        weights = np.clip((float(area_ref) / areas), 0.5, 3.0)
    else:
        weights = np.ones_like(areas, dtype=float)

    offsets = np.zeros((len(geoms), 2), dtype=float)

    for _ in range(int(iters)):
        tree = STRtree(geoms)
        moved = np.zeros_like(offsets)
        any_push = False

        for i, gi in enumerate(geoms):
            env = gi.buffer(clearance).envelope
            cand_idx = [j for j in tree.query(env) if j != i]
            if not cand_idx:
                continue

            vi = np.array([0.0, 0.0], dtype=float)

            for j in cand_idx:
                gj = geoms[j]
                d = gi.distance(gj)
                if d >= clearance or d == 0:
                    continue

                pi, pj = nearest_points(gi, gj)
                dir_vec = np.array([pi.x - pj.x, pi.y - pj.y], dtype=float)
                nrm = float(np.linalg.norm(dir_vec))
                if nrm == 0:
                    continue
                dir_vec /= nrm

                deficit = clearance - d
                push_i = 0.5 * deficit * (weights[i] / (weights[i] + weights[j] + 1e-6))
                vi += dir_vec * push_i
                any_push = True

            nrm = float(np.linalg.norm(vi))
            if nrm > 0:
                vi = (vi / nrm) * min(float(step), nrm)

            moved[i] = vi

        offsets += moved
        norms = np.linalg.norm(offsets, axis=1)
        too_far = norms > float(max_total)
        if np.any(too_far):
            scale_f = (float(max_total) / (norms[too_far] + 1e-6)).reshape(-1, 1)
            offsets[too_far] *= scale_f

        geoms = [translate(g, xoff=float(dx), yoff=float(dy)) for g, (dx, dy) in zip(geoms, moved)]

        if (not any_push) or (np.max(np.linalg.norm(moved, axis=1)) < 1e-3):
            break

    out["geometry"] = geoms
    out = out[~out.geometry.is_empty & out.geometry.is_valid]
    return out.reset_index(drop=True)
