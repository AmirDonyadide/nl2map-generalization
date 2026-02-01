# src/imgofup/userstudy/tiling.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from imgofup.config.constants import (
    # tiling defaults
    USERSTUDY_TILE_SIZE_M_DEFAULT,
    USERSTUDY_TILE_ID_COL_DEFAULT,
)


@dataclass(frozen=True)
class TileGrid:
    """
    A uniform square grid covering the extent of a GeoDataFrame.

    Attributes
    ----------
    grid:
        GeoDataFrame of tiles with columns [tile_id, geometry].
    tile_size_m:
        Tile edge length in CRS units (expected meters).
    tile_id_col:
        Column name used for tile ids (default: "tile_id").
    """
    grid: gpd.GeoDataFrame
    tile_size_m: float
    tile_id_col: str = USERSTUDY_TILE_ID_COL_DEFAULT


def make_grid(
    gdf: gpd.GeoDataFrame,
    *,
    tile_size_m: float = float(USERSTUDY_TILE_SIZE_M_DEFAULT),
    tile_id_col: str = USERSTUDY_TILE_ID_COL_DEFAULT,
    snap_origin: bool = True,
) -> TileGrid:
    """
    Create a square tiling grid over `gdf.total_bounds`.

    Parameters
    ----------
    gdf:
        GeoDataFrame whose bounds define the grid extent.
    tile_size_m:
        Tile width/height in CRS units (meters recommended).
    tile_id_col:
        Column name for tile ids.
    snap_origin:
        If True, snap xmin/ymin down to a multiple of tile_size_m
        to make grids stable across runs.

    Returns
    -------
    TileGrid
        A dataclass containing the grid.
    """
    if gdf is None or gdf.empty:
        raise ValueError("make_grid: input GeoDataFrame is empty.")

    if gdf.crs is None:
        raise ValueError("make_grid: input GeoDataFrame has no CRS. Set CRS before tiling.")

    size = float(tile_size_m)
    if size <= 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m}")

    xmin, ymin, xmax, ymax = map(float, gdf.total_bounds)

    if snap_origin:
        xmin0 = np.floor(xmin / size) * size
        ymin0 = np.floor(ymin / size) * size
    else:
        xmin0, ymin0 = xmin, ymin

    # ensure at least one tile if bounds are degenerate
    xmax = max(xmax, xmin0 + size)
    ymax = max(ymax, ymin0 + size)

    xs = np.arange(xmin0, xmax, size, dtype=float)
    ys = np.arange(ymin0, ymax, size, dtype=float)

    polys = [box(x, y, x + size, y + size) for x in xs for y in ys]
    ids = np.arange(len(polys), dtype=int)

    grid = gpd.GeoDataFrame({tile_id_col: ids}, geometry=polys, crs=gdf.crs)
    return TileGrid(grid=grid, tile_size_m=size, tile_id_col=tile_id_col)


def assign_tile_ids_by_centroid(
    gdf: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    *,
    tile_id_col: str = USERSTUDY_TILE_ID_COL_DEFAULT,
    predicate: str = "within",
) -> gpd.GeoDataFrame:
    """
    Assign each feature in `gdf` to a tile in `grid` based on centroid location.

    Notes
    -----
    - Uses a spatial join between centroids and tile polygons.
    - Features whose centroid falls outside any tile get tile_id = NA and are dropped.

    Parameters
    ----------
    gdf:
        Input geometries (e.g. buildings polygons).
    grid:
        Tile GeoDataFrame produced by make_grid().
    tile_id_col:
        Name of tile id column in `grid` and output.
    predicate:
        Spatial predicate for join (default "within").

    Returns
    -------
    GeoDataFrame
        Copy of gdf with a tile_id column added, and rows without tile assignment removed.
    """
    if gdf is None or gdf.empty:
        return gdf.copy()

    if gdf.crs is None or grid.crs is None:
        raise ValueError("assign_tile_ids_by_centroid: CRS missing on gdf or grid.")
    if str(gdf.crs) != str(grid.crs):
        raise ValueError("assign_tile_ids_by_centroid: CRS mismatch between gdf and grid.")

    if tile_id_col not in grid.columns:
        raise KeyError(f"grid is missing tile_id_col={tile_id_col!r}")

    # centroid join
    cent = gdf.copy()
    cent["geometry"] = cent.geometry.centroid

    joined = gpd.sjoin(
        cent,
        grid[[tile_id_col, "geometry"]],
        how="left",
        predicate=predicate,
    )

    out = gdf.copy()
    out[tile_id_col] = joined[tile_id_col].values

    out = out.dropna(subset=[tile_id_col]).copy()
    out[tile_id_col] = out[tile_id_col].astype(int)
    return out


def top_tiles_by_count(
    gdf: gpd.GeoDataFrame,
    *,
    tile_id_col: str = USERSTUDY_TILE_ID_COL_DEFAULT,
    top_k: int = 100,
) -> list[int]:
    """
    Return the tile ids with the most features assigned.

    Useful for: selecting "dense" tiles for user study samples.
    """
    if gdf is None or gdf.empty:
        return []
    if tile_id_col not in gdf.columns:
        raise KeyError(f"gdf missing tile_id_col={tile_id_col!r}")
    vc = gdf.groupby(tile_id_col).size().sort_values(ascending=False)
    return [int(x) for x in vc.head(int(top_k)).index.tolist()]


def filter_grid_to_tile_ids(
    grid: gpd.GeoDataFrame,
    tile_ids: Sequence[int],
    *,
    tile_id_col: str = USERSTUDY_TILE_ID_COL_DEFAULT,
) -> gpd.GeoDataFrame:
    """
    Keep only tiles with ids in `tile_ids`.
    """
    if grid is None or grid.empty:
        return grid.copy()
    if tile_id_col not in grid.columns:
        raise KeyError(f"grid missing tile_id_col={tile_id_col!r}")
    keep = set(int(x) for x in tile_ids)
    return grid[grid[tile_id_col].isin(keep)].copy()
