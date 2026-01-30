# src/config.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --------------------------- helpers ---------------------------

def env_path(key: str, default: Path) -> Path:
    val = os.getenv(key)
    return Path(val) if val else default

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def try_infer_dims(prompt_npz: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    If prompt embeddings exist, infer PROMPT_DIM.
    MAP_DIM is set elsewhere (after map embedding) — keep None here.
    """
    try:
        import numpy as np
        if prompt_npz.exists():
            z = np.load(prompt_npz, allow_pickle=True)
            E = z["E"]
            prm_dim = int(E.shape[1])
            return None, prm_dim
    except Exception:
        pass
    return None, None

def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# --------------------------- operator groups (Solution 1) ---------------------------

# Distance-based operators (param_value in meters)
DISTANCE_OPS = ("aggregate", "displace", "simplify")

# Area-based operators (param_value in square meters)
AREA_OPS = ("select",)


# --------------------------- extent reference columns ---------------------------
# These names must match what map_embeddings.py writes into maps.parquet
# and what concat_embeddings.py merges into train_pairs.parquet.

EXTENT_MINX_COL   = "extent_minx"
EXTENT_MINY_COL   = "extent_miny"
EXTENT_MAXX_COL   = "extent_maxx"
EXTENT_MAXY_COL   = "extent_maxy"
EXTENT_W_COL      = "extent_width_m"
EXTENT_H_COL      = "extent_height_m"
EXTENT_DIAG_COL   = "extent_diag_m"
EXTENT_AREA_COL   = "extent_area_m2"
EXTENT_CRS_COL    = "extent_crs"   # optional; only if you add it


# --------------------------- normalization behavior ---------------------------

# Primary behavior: use per-map dynamic extents from GeoJSON (recommended)
USE_DYNAMIC_EXTENT_REFS: bool = env_bool("USE_DYNAMIC_EXTENT_REFS", True)

# What to do if a map has missing/degenerate extent (diag/area NaN or <=0):
# - If True: fall back to default tile constants below
# - If False: drop those rows (you handle drop in notebook/code)
ALLOW_FALLBACK_EXTENT: bool = env_bool("ALLOW_FALLBACK_EXTENT", True)


# --------------------------- param estimation strategy ---------------------------
# Choose how to estimate param_value at inference:
# - "mlp": your current per-operator MLPRegressor (predict param_norm -> unnormalize)
# - "hybrid": rule-based parsing (numbers/%/small/large) + optional ML fallback
PARAM_STRATEGY: str = os.getenv("PARAM_STRATEGY", "mlp").strip().lower()
if PARAM_STRATEGY not in {"mlp", "hybrid"}:
    raise ValueError(f"PARAM_STRATEGY must be 'mlp' or 'hybrid', got: {PARAM_STRATEGY}")


# --------------------------- hybrid rules: qualitative -> quantile ---------------------------
# Used when prompt has "small/medium/large" but no number.
# You can tune these in thesis experiments.
QUAL_TO_QUANTILE = {
    "very_small": 0.10,
    "small": 0.25,
    "medium": 0.50,
    "large": 0.75,
    "very_large": 0.90,
}

# Words that map to the above categories (lowercased match)
QUAL_SYNONYMS = {
    "very_small": ["very small", "tiny", "minuscule", "very little"],
    "small":      ["small", "smaller", "minor", "little"],
    "medium":     ["medium", "moderate", "average", "normal"],
    "large":      ["large", "bigger", "big", "major"],
    "very_large": ["very large", "huge", "massive", "giant"],
}


# --------------------------- hybrid rules: unit aliases ---------------------------
# Normalize units found in text.
UNIT_ALIASES = {
    # distance
    "m":  ["m", "meter", "meters", "metre", "metres"],
    "km": ["km", "kilometer", "kilometers", "kilometre", "kilometres"],
    # area
    "m2": ["m2", "m^2", "sqm", "sq m", "sq. m", "square meter", "square meters",
           "square metre", "square metres", "meter^2", "metre^2"],
    "km2": ["km2", "km^2", "square kilometer", "square kilometers", "square kilometre", "square kilometres"],
    # percent
    "%":  ["%", "percent", "percentage", "per cent"],
}

# If prompt has *no* usable number/percent/qual word, use these defaults (in original units).
# Keep conservative so you don't explode errors.
DEFAULT_PARAM_BY_OPERATOR = {
    "aggregate": 5.0,   # meters (example default)
    "displace":  5.0,   # meters
    "simplify":  5.0,   # meters
    "select":   50.0,   # m² (example default)
}

# --------------------------- config ----------------------------

@dataclass(frozen=True)
class ProjectPaths:
    PROJ_ROOT: Path = Path(os.getenv("PROJ_ROOT", "../")).resolve()

    # Data
    DATA_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data")
    INPUT_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "input")
    OUTPUT_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output")

    # ----------------------- Inputs (User Study Excel) -----------------------
    USER_STUDY_XLSX: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "userstudy" / "UserStudy.xlsx"
    )

    RESPONSES_SHEET: str = os.getenv("RESPONSES_SHEET", "Responses")

    TILE_ID_COL: str = os.getenv("TILE_ID_COL", "tile_id")
    COMPLETE_COL: str = os.getenv("COMPLETE_COL", "complete")
    REMOVE_COL: str = os.getenv("REMOVE_COL", "remove")
    TEXT_COL: str = os.getenv("TEXT_COL", "cleaned_text")
    PROMPT_ID_COL: str = os.getenv("PROMPT_ID_COL", "prompt_id")
    PARAM_VALUE_COL: str = os.getenv("PARAM_VALUE_COL", "param_value")
    OPERATOR_COL: str = os.getenv("OPERATOR_COL", "operator")
    INTENSITY_COL: str = os.getenv("INTENSITY_COL", "intensity")

    ONLY_COMPLETE: bool = env_bool("ONLY_COMPLETE", True)
    EXCLUDE_REMOVED: bool = env_bool("EXCLUDE_REMOVED", True)

    PROMPT_ID_PREFIX: str = os.getenv("PROMPT_ID_PREFIX", "r")
    PROMPT_ID_WIDTH: int = int(os.getenv("PROMPT_ID_WIDTH", "8"))

    SPLIT_BY: str = os.getenv("SPLIT_BY", "tile")

    # ----------------------- Map inputs -----------------------
    MAPS_ROOT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "input" / "samples" / "pairs"
    )
    INPUT_MAPS_PATTERN: str = os.getenv("INPUT_MAPS_PATTERN", "*_input.geojson")

    # ----------------------- Outputs -----------------------
    PROMPT_OUT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "prompt_out"
    )
    MAP_OUT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "map_out"
    )
    TRAIN_OUT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "train_out"
    )
    MODEL_OUT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "models"
    )
    SPLIT_OUT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "train_out" / "splits"
    )

    PRM_NPZ: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "prompt_out" / "prompts_embeddings.npz"
    )

    # NEW: useful canonical artifacts
    MAPS_PARQUET: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "map_out" / "maps.parquet"
    )
    TRAIN_PAIRS_PARQUET: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "train_out" / "train_pairs.parquet"
    )

    def ensure_outputs(self) -> "ProjectPaths":
        from os import makedirs
        makedirs(self.OUTPUT_DIR, exist_ok=True)
        for p in (self.PROMPT_OUT, self.MAP_OUT, self.TRAIN_OUT, self.MODEL_OUT, self.SPLIT_OUT):
            makedirs(p, exist_ok=True)
        return self

    def clean_outputs(self) -> None:
        import shutil
        for d in [self.PROMPT_OUT, self.MAP_OUT, self.TRAIN_OUT, self.MODEL_OUT, self.SPLIT_OUT]:
            if d.exists():
                print(f"🧹 Removing old directory: {d}")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print("✅ All output folders cleaned and recreated fresh.\n")


@dataclass(frozen=True)
class ModelConfig:
    """
    Model hyper-parameters and prompt encoder config.
    """
    PROMPT_ENCODER: str = os.getenv("PROMPT_ENCODER", "openai-small")

    MAP_DIM: int = int(os.getenv("MAP_DIM", "165"))
    PROMPT_DIM: int = int(os.getenv("PROMPT_DIM", "512"))
    FUSED_DIM: int = 0

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "512"))

    VAL_RATIO: float = float(os.getenv("VAL_RATIO", "0.15"))
    TEST_RATIO: float = float(os.getenv("TEST_RATIO", "0.15"))
    SEED: int = int(os.getenv("SEED", "42"))

    # Fallback tile scale (ONLY used if dynamic extents are missing/degenerate)
    DEFAULT_TILE_WIDTH_M: float = float(os.getenv("DEFAULT_TILE_WIDTH_M", "400"))
    DEFAULT_TILE_HEIGHT_M: float = float(os.getenv("DEFAULT_TILE_HEIGHT_M", "400"))

    DEFAULT_TILE_DIAG_M: float = 0.0
    DEFAULT_TILE_AREA_M2: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "FUSED_DIM", self.MAP_DIM + self.PROMPT_DIM)

        diag = math.sqrt(self.DEFAULT_TILE_WIDTH_M**2 + self.DEFAULT_TILE_HEIGHT_M**2)
        area = self.DEFAULT_TILE_WIDTH_M * self.DEFAULT_TILE_HEIGHT_M
        object.__setattr__(self, "DEFAULT_TILE_DIAG_M", float(diag))
        object.__setattr__(self, "DEFAULT_TILE_AREA_M2", float(area))


# --------------------------- public API ------------------------

PATHS = ProjectPaths().ensure_outputs()

_, inferred_prm_dim = try_infer_dims(PATHS.PRM_NPZ)
if inferred_prm_dim is not None:
    CFG = ModelConfig(PROMPT_DIM=inferred_prm_dim)
else:
    CFG = ModelConfig()


def print_summary():
    print("=== CONFIG SUMMARY ===")
    print("PROJ_ROOT  :", PATHS.PROJ_ROOT)
    print("DATA_DIR   :", PATHS.DATA_DIR)
    print("INPUT_DIR  :", PATHS.INPUT_DIR)
    print("OUTPUT_DIR :", PATHS.OUTPUT_DIR)
    print("MAPS_ROOT  :", PATHS.MAPS_ROOT)
    print("INPUT PAT. :", PATHS.INPUT_MAPS_PATTERN)

    print("--- User Study ---")
    print("USER_STUDY_XLSX :", PATHS.USER_STUDY_XLSX)
    print("RESPONSES_SHEET :", PATHS.RESPONSES_SHEET)
    print("TILE_ID_COL     :", PATHS.TILE_ID_COL)
    print("COMPLETE_COL    :", PATHS.COMPLETE_COL)
    print("REMOVE_COL      :", PATHS.REMOVE_COL)
    print("TEXT_COL        :", PATHS.TEXT_COL)
    print("PARAM_VALUE_COL :", PATHS.PARAM_VALUE_COL)
    print("OPERATOR_COL    :", PATHS.OPERATOR_COL)
    print("INTENSITY_COL   :", PATHS.INTENSITY_COL)

    print("--- Filters / IDs / Split ---")
    print("ONLY_COMPLETE   :", PATHS.ONLY_COMPLETE)
    print("EXCLUDE_REMOVED :", PATHS.EXCLUDE_REMOVED)
    print("PROMPT_ID_COL   :", PATHS.PROMPT_ID_COL)
    print("PROMPT_ID_RULE  :", "Read from Excel (no generation).")
    print("SPLIT_BY        :", PATHS.SPLIT_BY)

    print("--- Outputs ---")
    print("PROMPT_OUT :", PATHS.PROMPT_OUT)
    print("MAP_OUT    :", PATHS.MAP_OUT)
    print("TRAIN_OUT  :", PATHS.TRAIN_OUT)
    print("MODEL_OUT  :", PATHS.MODEL_OUT)
    print("SPLIT_OUT  :", PATHS.SPLIT_OUT)
    print("PRM_NPZ    :", PATHS.PRM_NPZ)
    print("MAPS_PQ    :", PATHS.MAPS_PARQUET)
    print("PAIRS_PQ   :", PATHS.TRAIN_PAIRS_PARQUET)

    print("--- Model ---")
    print("PROMPT_ENCODER:", CFG.PROMPT_ENCODER)
    print("MAP_DIM       :", CFG.MAP_DIM)
    print("PROMPT_DIM    :", CFG.PROMPT_DIM)
    print("FUSED_DIM     :", CFG.FUSED_DIM)
    print("BATCH_SIZE    :", CFG.BATCH_SIZE)
    print("VAL/TEST      :", CFG.VAL_RATIO, CFG.TEST_RATIO)
    print("SEED          :", CFG.SEED)

    print("--- Normalization behavior ---")
    print("USE_DYNAMIC_EXTENT_REFS :", USE_DYNAMIC_EXTENT_REFS)
    print("ALLOW_FALLBACK_EXTENT   :", ALLOW_FALLBACK_EXTENT)

    print("--- Fallback tile scale (ONLY if dynamic refs missing) ---")
    print("DEFAULT_TILE_W/H (m) :", CFG.DEFAULT_TILE_WIDTH_M, CFG.DEFAULT_TILE_HEIGHT_M)
    print("DEFAULT_TILE_DIAG_M  :", CFG.DEFAULT_TILE_DIAG_M)
    print("DEFAULT_TILE_AREA_M2 :", CFG.DEFAULT_TILE_AREA_M2)

    print("--- Extent columns ---")
    print("EXTENT_DIAG_COL :", EXTENT_DIAG_COL)
    print("EXTENT_AREA_COL :", EXTENT_AREA_COL)

    print("--- Operator groups ---")
    print("DISTANCE_OPS :", DISTANCE_OPS)
    print("AREA_OPS     :", AREA_OPS)
    print("--- Operator groups ---")
    print("DISTANCE_OPS :", DISTANCE_OPS)
    print("AREA_OPS     :", AREA_OPS)

    print("--- Param estimation ---")
    print("PARAM_STRATEGY :", PARAM_STRATEGY)
    print("QUAL_TO_QUANTILE:", QUAL_TO_QUANTILE)
    print("DEFAULT_PARAM_BY_OPERATOR:", DEFAULT_PARAM_BY_OPERATOR)

