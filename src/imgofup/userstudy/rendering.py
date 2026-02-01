# src/imgofup/userstudy/rendering.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.affinity import translate, scale

from imgofup.config.constants import (
    # render defaults
    USERSTUDY_RENDER_FIG_INCH_DEFAULT,
    USERSTUDY_RENDER_DPI_DEFAULT,
    USERSTUDY_SHOW_ROADS_DEFAULT,
    USERSTUDY_ROAD_COLOR_DEFAULT,
    USERSTUDY_BUILDING_FACE_DEFAULT,
    USERSTUDY_BUILDING_EDGE_DEFAULT,
    USERSTUDY_ROAD_WIDTH_MAP_DEFAULT,
    USERSTUDY_TILE_SIZE_M_DEFAULT,
)


def _normalize_gdf_to_square(
    gdf: gpd.GeoDataFrame,
    *,
    xmin: float,
    ymin: float,
    sx: float,
    sy: float,
) -> gpd.GeoDataFrame:
    """
    Shift and scale geometries so that the tile bbox maps to a square [0..T]x[0..T].
    """
    if gdf is None or gdf.empty:
        return gdf

    out = gdf.copy()
    out["geometry"] = out["geometry"].apply(
        lambda geom: scale(
            translate(geom, xoff=-xmin, yoff=-ymin),
            xfact=sx,
            yfact=sy,
            origin=(0, 0),
        )
    )
    return out


def _meter_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, pos: f"{int(v)} m")


def render_single_frame(
    *,
    tile_id: int,
    buildings_gdf: gpd.GeoDataFrame,
    tile_poly,
    out_path: Path,
    roads_in_tiles: Optional[gpd.GeoDataFrame] = None,
    show_roads: bool = USERSTUDY_SHOW_ROADS_DEFAULT,
    fig_inch: float = USERSTUDY_RENDER_FIG_INCH_DEFAULT,
    dpi: int = USERSTUDY_RENDER_DPI_DEFAULT,
    target_size_m: float = float(USERSTUDY_TILE_SIZE_M_DEFAULT),
    road_color: str = USERSTUDY_ROAD_COLOR_DEFAULT,
    building_face: str = USERSTUDY_BUILDING_FACE_DEFAULT,
    building_edge: str = USERSTUDY_BUILDING_EDGE_DEFAULT,
    road_width_map: Optional[dict[str, float]] = None,
) -> None:
    """
    Render one tile frame (input or generalized) to a PNG.

    Behavior
    --------
    - Normalizes tile bbox into a square 0..target_size_m in both axes.
    - Labels axes in meters (0 m .. target_size_m).
    - Optional road layer (if roads_in_tiles provided).
    - Buildings are filled polygons.

    Parameters
    ----------
    tile_id:
        Tile identifier used for filtering roads_in_tiles (expects a 'tile_id' column).
    buildings_gdf:
        Polygons to draw (buildings).
    tile_poly:
        Tile polygon defining bbox (must have .bounds).
    out_path:
        Output PNG path.
    roads_in_tiles:
        Optional roads GeoDataFrame with column 'tile_id' and optionally 'highway'.
    show_roads:
        If True and roads_in_tiles is not None, draw roads.
    target_size_m:
        Final normalized axis size (typically 400).
    road_width_map:
        Optional mapping highway-class -> linewidth. Uses constants default if None.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    road_width_map = road_width_map or USERSTUDY_ROAD_WIDTH_MAP_DEFAULT

    fig, ax = plt.subplots(figsize=(float(fig_inch), float(fig_inch)))

    xmin, ymin, xmax, ymax = tile_poly.bounds
    width = float(xmax - xmin)
    height = float(ymax - ymin)

    # avoid division by zero on degenerate tiles
    width = max(width, 1e-12)
    height = max(height, 1e-12)

    T = float(target_size_m)
    sx = T / width
    sy = T / height

    # --- optional roads layer ---
    if show_roads and roads_in_tiles is not None and not roads_in_tiles.empty:
        if "tile_id" not in roads_in_tiles.columns:
            raise KeyError("roads_in_tiles must contain a 'tile_id' column when show_roads=True.")

        r_tile = roads_in_tiles[roads_in_tiles["tile_id"] == int(tile_id)]
        r_tile_n = _normalize_gdf_to_square(r_tile, xmin=xmin, ymin=ymin, sx=sx, sy=sy)

        if r_tile_n is not None and not r_tile_n.empty:
            if "highway" in r_tile_n.columns:
                for cls, df in r_tile_n.groupby("highway"):
                    lw = float(road_width_map.get(str(cls), 0.8))
                    df.plot(ax=ax, color=str(road_color), linewidth=lw, zorder=2)
            else:
                r_tile_n.plot(ax=ax, color=str(road_color), linewidth=0.9, zorder=2)

    # --- buildings layer ---
    bldg_n = _normalize_gdf_to_square(buildings_gdf, xmin=xmin, ymin=ymin, sx=sx, sy=sy)
    if bldg_n is not None and not bldg_n.empty:
        bldg_n.plot(
            ax=ax,
            facecolor=str(building_face),
            edgecolor=str(building_edge),
            linewidth=0.3,
            zorder=3,
        )

    # --- axis styling ---
    ax.set_xlim(0, T)
    ax.set_ylim(0, T)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    nticks = 5
    ticks = np.linspace(0, T, nticks)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    mf = _meter_formatter()
    ax.xaxis.set_major_formatter(mf)
    ax.yaxis.set_major_formatter(mf)

    # light grid lines near axes (small “ruler” look)
    grid_height = T * 0.05
    for x in ticks:
        ax.plot([x, x], [0, grid_height], color="0.85", lw=0.5, zorder=1)

    grid_width = T * 0.05
    for y in ticks:
        ax.plot([0, grid_width], [y, y], color="0.85", lw=0.5, zorder=1)

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def compute_tolerance_for_rendering(
    *,
    tile_size_m: float = float(USERSTUDY_TILE_SIZE_M_DEFAULT),
    fig_inch: float = USERSTUDY_RENDER_FIG_INCH_DEFAULT,
    dpi: int = USERSTUDY_RENDER_DPI_DEFAULT,
    factor: float = 0.75,
) -> float:
    """
    Compute a simplification tolerance tuned to rendering resolution.

    Idea
    ----
    meters_per_pixel = tile_size_m / (fig_inch * dpi)
    tolerance ≈ meters_per_pixel * factor

    Use this to simplify buildings/roads *only for rendering speed/clarity*.

    Returns
    -------
    float
        Simplification epsilon in meters.
    """
    px = float(fig_inch) * float(dpi)
    mpp = float(tile_size_m) / max(px, 1.0)
    return float(mpp * float(factor))
