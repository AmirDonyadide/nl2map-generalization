# src/imgofup/userstudy/sample_generation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd

from imgofup.config.constants import (
    # schema
    USERSTUDY_TILE_ID_COL_DEFAULT,
    OPERATOR_COL,
    INTENSITY_COL,
    PARAM_VALUE_COL,
    # id formatting
    MAP_ID_WIDTH,
    # operators / label space
    FIXED_OPERATOR_CLASSES,
    # userstudy defaults
    USERSTUDY_INTENSITIES_DEFAULT,
    USERSTUDY_SEED_DEFAULT,
    USERSTUDY_TOP_K_TILES_DEFAULT,
    USERSTUDY_SAMPLES_DIR_DEFAULT,
    USERSTUDY_METADATA_DIR_DEFAULT,
    USERSTUDY_META_CSV_NAME_DEFAULT,
    USERSTUDY_META_XLSX_NAME_DEFAULT,
    USERSTUDY_INPUT_GEOJSON_SUFFIX_DEFAULT,
    USERSTUDY_TARGET_GEOJSON_SUFFIX_DEFAULT,
    USERSTUDY_INPUT_PNG_PREFIX_DEFAULT,
    USERSTUDY_TARGET_PNG_PREFIX_DEFAULT,
    USERSTUDY_RENDER_PNG_DEFAULT,
    USERSTUDY_SHOW_ROADS_DEFAULT,
)

from imgofup.userstudy.tiling import (
    TileGrid,
    make_grid,
    assign_tile_ids_by_centroid,
    top_tiles_by_count,
    filter_grid_to_tile_ids,
)

from imgofup.userstudy.operators import (
    aggregate_buildings,
    simplify_buildings,
    displace_buildings,
    select_buildings,
)

from imgofup.userstudy.param_search import (
    choose_aggregate_param_for_tile,
    choose_simplify_param_for_tile,
    choose_displace_param_for_tile,
    choose_select_param_for_tile,
)

from imgofup.userstudy.rendering import (
    render_single_frame,
    compute_tolerance_for_rendering,
)


@dataclass(frozen=True)
class UserStudySamplePaths:
    """
    Where the user study sample artifacts are written.

    samples_dir layout:
      samples_dir/
        0001/
          0001_input.geojson
          0001_generalized.geojson
          input_0001.png
          generalized_0001.png
        0002/
          ...
    """
    samples_dir: Path
    metadata_dir: Path

    def ensure(self) -> "UserStudySamplePaths":
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class SampleGenResult:
    """
    Summary returned by generate_userstudy_samples().
    """
    n_tiles_total: int
    n_tiles_selected: int
    samples_dir: str
    meta_csv: str
    meta_xlsx: str


def _balanced_split(ids: Sequence[int], groups: Sequence[str]) -> Dict[str, List[int]]:
    """
    Split ids into len(groups) chunks with sizes differing by at most 1.
    Returns dict[group -> list[int]].
    """
    ids = list(ids)
    n = len(ids)
    g = len(groups)
    base = n // g
    rem = n % g
    out: Dict[str, List[int]] = {}
    start = 0
    for i, grp in enumerate(groups):
        size = base + (1 if i < rem else 0)
        out[grp] = ids[start : start + size]
        start += size
    return out


def _format_tile_id(tid: int) -> str:
    return str(int(tid)).zfill(int(MAP_ID_WIDTH))


def generate_userstudy_samples(
    *,
    buildings: gpd.GeoDataFrame,
    tile_size_m: float,
    out: Optional[UserStudySamplePaths] = None,
    roads: Optional[gpd.GeoDataFrame] = None,
    top_k_tiles: int = USERSTUDY_TOP_K_TILES_DEFAULT,
    operators: Sequence[str] = FIXED_OPERATOR_CLASSES,
    intensities: Sequence[str] = USERSTUDY_INTENSITIES_DEFAULT,
    seed: int = USERSTUDY_SEED_DEFAULT,
    render_png: bool = USERSTUDY_RENDER_PNG_DEFAULT,
    show_roads: bool = USERSTUDY_SHOW_ROADS_DEFAULT,
    simplify_for_render: bool = True,
) -> SampleGenResult:
    """
    High-level orchestration: create tile grid, pick top-K tiles, assign operator×intensity,
    compute per-tile parameter (via param_search), apply operator, save artifacts.

    Requirements
    ------------
    - buildings must be in a *metric CRS* (meters) because parameters are in meters / m².
    - roads (if provided) must match buildings CRS.
    """
    if buildings is None or buildings.empty:
        raise ValueError("buildings is empty.")
    if buildings.crs is None:
        raise ValueError("buildings.crs is None. Reproject buildings to a metric CRS first.")
    if roads is not None:
        if roads.crs is None:
            raise ValueError("roads.crs is None.")
        if str(roads.crs) != str(buildings.crs):
            raise ValueError("roads CRS must match buildings CRS.")

    out = out or UserStudySamplePaths(
        samples_dir=Path(USERSTUDY_SAMPLES_DIR_DEFAULT),
        metadata_dir=Path(USERSTUDY_METADATA_DIR_DEFAULT),
    )
    out = out.ensure()

    rng = np.random.RandomState(int(seed))

    # --- tile grid ---
    grid_pack: TileGrid = make_grid(
        buildings,
        tile_size_m=float(tile_size_m),
        tile_id_col=USERSTUDY_TILE_ID_COL_DEFAULT,
        snap_origin=True,
    )
    grid = grid_pack.grid

    # assign tile_id to buildings by centroid
    bldg = assign_tile_ids_by_centroid(
        buildings,
        grid,
        tile_id_col=USERSTUDY_TILE_ID_COL_DEFAULT,
        predicate="within",
    )

    # pick top K tiles by building count
    top_ids = top_tiles_by_count(
        bldg,
        tile_id_col=USERSTUDY_TILE_ID_COL_DEFAULT,
        top_k=int(top_k_tiles),
    )
    if not top_ids:
        raise RuntimeError("No tiles selected. Check bbox/tiling inputs.")

    bldg_top = bldg[bldg[USERSTUDY_TILE_ID_COL_DEFAULT].isin(top_ids)].copy()
    grid_top = filter_grid_to_tile_ids(
        grid,
        top_ids,
        tile_id_col=USERSTUDY_TILE_ID_COL_DEFAULT,
    )

    # optional: prep roads tiled
    roads_in_tiles: Optional[gpd.GeoDataFrame] = None
    if roads is not None and show_roads:
        grid_pad = grid_top.copy()
        grid_pad["geometry"] = grid_pad.geometry.buffer(0.5)
        roads_in_tiles = gpd.sjoin(
            roads[["geometry"] + (["highway"] if "highway" in roads.columns else [])],
            grid_pad[[USERSTUDY_TILE_ID_COL_DEFAULT, "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right"])
        if "highway" not in roads_in_tiles.columns:
            roads_in_tiles["highway"] = None

    # simplify for rendering (optional)
    if simplify_for_render:
        tol = compute_tolerance_for_rendering(tile_size_m=float(tile_size_m))
        bldg_top = bldg_top.set_geometry(bldg_top.geometry.simplify(tol, preserve_topology=True))
        if roads_in_tiles is not None and not roads_in_tiles.empty:
            roads_in_tiles = roads_in_tiles.set_geometry(
                roads_in_tiles.geometry.simplify(tol * 1.2, preserve_topology=True)
            )

    # --- assignment of tiles -> operator x intensity ---
    tile_ids = sorted(bldg_top[USERSTUDY_TILE_ID_COL_DEFAULT].unique().astype(int).tolist())
    ids_shuffled = tile_ids.copy()
    rng.shuffle(ids_shuffled)

    op_groups = _balanced_split(ids_shuffled, [str(o) for o in operators])

    assignments: List[dict] = []
    for op, ids_for_op in op_groups.items():
        ids_for_op = list(ids_for_op)
        rng.shuffle(ids_for_op)

        inten_groups = _balanced_split(ids_for_op, [str(i) for i in intensities])
        for inten, ids_for_int in inten_groups.items():
            for tid in ids_for_int:
                assignments.append(
                    {
                        USERSTUDY_TILE_ID_COL_DEFAULT: int(tid),
                        OPERATOR_COL: str(op).strip().lower(),
                        INTENSITY_COL: str(inten).strip().lower(),
                        PARAM_VALUE_COL: np.nan,
                    }
                )

    assign_df = (
        pd.DataFrame(assignments)
        .sort_values([OPERATOR_COL, INTENSITY_COL, USERSTUDY_TILE_ID_COL_DEFAULT])
        .reset_index(drop=True)
    )

    # --- per-tile generation loop ---
    meta_rows: List[dict] = []

    for _, row in assign_df.iterrows():
        tid = int(row[USERSTUDY_TILE_ID_COL_DEFAULT])
        op = str(row[OPERATOR_COL]).strip().lower()
        inten = str(row[INTENSITY_COL]).strip().lower()

        sid = _format_tile_id(tid)
        sample_dir = out.samples_dir / sid
        sample_dir.mkdir(parents=True, exist_ok=True)

        # subset buildings for this tile
        g_tile = bldg_top.loc[bldg_top[USERSTUDY_TILE_ID_COL_DEFAULT] == tid, ["geometry"]].copy()

        # tile polygon for rendering normalization
        tile_poly = grid_top.loc[grid_top[USERSTUDY_TILE_ID_COL_DEFAULT] == tid].geometry.iloc[0]

        # ---- choose param + compute output ----
        if op == "aggregate":
            # param_search requires op_fn(gdf, dist) -> gdf_out
            param_val, out_gdf = choose_aggregate_param_for_tile(
                g_tile,
                inten,
                op_fn=lambda gdf, dist: aggregate_buildings(gdf, dist=float(dist)),
            )
            unit = "m"

        elif op == "simplify":
            # param_search requires op_fn(gdf, eps) -> gdf_out
            param_val, out_gdf = choose_simplify_param_for_tile(
                g_tile,
                inten,
                op_fn=lambda gdf, eps: simplify_buildings(gdf, eps=float(eps)),
            )
            unit = "m"

        elif op == "displace":
            # param_search requires op_fn(gdf, clearance) -> gdf_out
            param_val, out_gdf = choose_displace_param_for_tile(
                g_tile,
                inten,
                op_fn=lambda gdf, clearance: displace_buildings(gdf, clearance=float(clearance)),
            )
            unit = "m"

        elif op == "select":
            # choose_select returns ONLY the threshold; we must apply the operator ourselves
            param_val = choose_select_param_for_tile(g_tile, inten)
            out_gdf = select_buildings(g_tile, area_threshold=float(param_val))
            unit = "m^2"

        else:
            raise ValueError(f"Unknown operator '{op}'. Allowed: {list(operators)}")

        # --- write GeoJSONs ---
        in_geo = sample_dir / f"{sid}{USERSTUDY_INPUT_GEOJSON_SUFFIX_DEFAULT}"
        out_geo = sample_dir / f"{sid}{USERSTUDY_TARGET_GEOJSON_SUFFIX_DEFAULT}"

        if not in_geo.exists():
            g_tile.to_file(in_geo, driver="GeoJSON")
        out_gdf.to_file(out_geo, driver="GeoJSON")

        # --- render PNGs ---
        if render_png:
            in_png = sample_dir / f"{USERSTUDY_INPUT_PNG_PREFIX_DEFAULT}{sid}.png"
            out_png = sample_dir / f"{USERSTUDY_TARGET_PNG_PREFIX_DEFAULT}{sid}.png"

            render_single_frame(
                tile_id=tid,
                buildings_gdf=g_tile,
                tile_poly=tile_poly,
                roads_in_tiles=roads_in_tiles,
                out_path=in_png,
                show_roads=bool(show_roads),
                target_size_m=float(tile_size_m),
            )
            render_single_frame(
                tile_id=tid,
                buildings_gdf=out_gdf,
                tile_poly=tile_poly,
                roads_in_tiles=roads_in_tiles,
                out_path=out_png,
                show_roads=bool(show_roads),
                target_size_m=float(tile_size_m),
            )

        meta_rows.append(
            {
                "sample_id": sid,
                USERSTUDY_TILE_ID_COL_DEFAULT: int(tid),
                OPERATOR_COL: op,
                INTENSITY_COL: inten,
                PARAM_VALUE_COL: round(float(param_val), 3),
                "param_unit": unit,
                "n_input_polys": int(g_tile.shape[0]),
                "n_target_polys": int(out_gdf.shape[0]),
                "ratio": round(float(out_gdf.shape[0] / max(1, g_tile.shape[0])), 2),
                "is_target_empty": bool(out_gdf.empty),
                "input_geojson": str(in_geo),
                "target_geojson": str(out_geo),
            }
        )

    meta_df = pd.DataFrame(meta_rows)

    # write metadata
    meta_csv = out.metadata_dir / USERSTUDY_META_CSV_NAME_DEFAULT
    meta_xlsx = out.metadata_dir / USERSTUDY_META_XLSX_NAME_DEFAULT

    meta_df.to_csv(meta_csv, index=False)
    try:
        meta_df.to_excel(meta_xlsx, index=False)
    except Exception:
        pass

    return SampleGenResult(
        n_tiles_total=int(len(tile_ids)),
        n_tiles_selected=int(len(assign_df)),
        samples_dir=str(out.samples_dir),
        meta_csv=str(meta_csv),
        meta_xlsx=str(meta_xlsx),
    )
