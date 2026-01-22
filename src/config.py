# src/config.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --------------------------- helpers ---------------------------

def env_path(key: str, default: Path) -> Path:
    """Read a path from env (string), else use default."""
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
    """Parse common truthy/falsey env values."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# --------------------------- operator groups (Solution 1) ---------------------------

# Distance-based operators (param_value in meters)
DISTANCE_OPS = ("aggregate", "displace", "simplify")

# Area-based operators (param_value in square meters)
AREA_OPS = ("select",)


# --------------------------- config ----------------------------

@dataclass(frozen=True)
class ProjectPaths:
    # Project root that contains `src/` and `data/`
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

    # Excel sheet name
    RESPONSES_SHEET: str = os.getenv("RESPONSES_SHEET", "Responses")

    # Required columns inside the Responses sheet
    TILE_ID_COL: str = os.getenv("TILE_ID_COL", "tile_id")        # number
    COMPLETE_COL: str = os.getenv("COMPLETE_COL", "complete")      # True / False
    REMOVE_COL: str = os.getenv("REMOVE_COL", "remove")            # True / False
    TEXT_COL: str = os.getenv("TEXT_COL", "cleaned_text")          # text
    PARAM_VALUE_COL: str = os.getenv("PARAM_VALUE_COL", "param_value")  # float
    OPERATOR_COL: str = os.getenv("OPERATOR_COL", "operator")           # text
    INTENSITY_COL: str = os.getenv("INTENSITY_COL", "intensity")        # text

    # Training inclusion filters (keep consistent across scripts)
    ONLY_COMPLETE: bool = env_bool("ONLY_COMPLETE", True)       # keep only complete==True
    EXCLUDE_REMOVED: bool = env_bool("EXCLUDE_REMOVED", True)   # keep only remove==False

    # Prompt ID scheme (must match prompt_embeddings + concat)
    PROMPT_ID_PREFIX: str = os.getenv("PROMPT_ID_PREFIX", "r")
    PROMPT_ID_WIDTH: int = int(os.getenv("PROMPT_ID_WIDTH", "8"))

    # Split strategy (avoid leakage!)
    # Recommended: 'tile' means group split by TILE_ID_COL.
    SPLIT_BY: str = os.getenv("SPLIT_BY", "tile")  # 'tile' or 'row'

    # ----------------------- Map inputs -----------------------
    MAPS_ROOT: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "input" / "samples" / "pairs"
    )

    # File patterns
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

    # Precomputed embeddings
    PRM_NPZ: Path = (
        Path(os.getenv("PROJ_ROOT", "../")).resolve()
        / "data" / "output" / "prompt_out" / "prompts_embeddings.npz"
    )

    # -------------------------------------------------------------
    def ensure_outputs(self) -> "ProjectPaths":
        """Create output directories if missing."""
        from os import makedirs
        makedirs(self.OUTPUT_DIR, exist_ok=True)
        for p in (self.PROMPT_OUT, self.MAP_OUT, self.TRAIN_OUT, self.MODEL_OUT, self.SPLIT_OUT):
            makedirs(p, exist_ok=True)
        return self

    def clean_outputs(self) -> None:
        """Remove and recreate all output directories (use with caution)."""
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

    PROMPT_ENCODER:
      - 'dan', 'transformer'  → USE variants
      - 'openai-small', 'openai-large' → OpenAI embedding models
    """

    # Prompt encoder name (generic, not just USE)
    PROMPT_ENCODER: str = "openai-small"
    # PROMPT_ENCODER: str = "dan"

    # Dimensions (can be overridden by env or inferred later)
    MAP_DIM: int = int(os.getenv("MAP_DIM", "165"))
    PROMPT_DIM: int = int(os.getenv("PROMPT_DIM", "512"))

    # Will be set in __post_init__
    FUSED_DIM: int = 0

    # Training
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "512"))

    # Splits
    VAL_RATIO: float = float(os.getenv("VAL_RATIO", "0.15"))
    TEST_RATIO: float = float(os.getenv("TEST_RATIO", "0.15"))
    SEED: int = int(os.getenv("SEED", "42"))

    # ----------------------- Tile scale (Solution 1 normalization) -----------------------
    # Tile width/height in meters (set via env for other datasets)
    TILE_WIDTH_M: float = float(os.getenv("TILE_WIDTH_M", "400"))
    TILE_HEIGHT_M: float = float(os.getenv("TILE_HEIGHT_M", "400"))

    # Derived (computed in __post_init__)
    TILE_DIAG_M: float = 0.0
    TILE_AREA_M2: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "FUSED_DIM", self.MAP_DIM + self.PROMPT_DIM)

        diag = math.sqrt(self.TILE_WIDTH_M**2 + self.TILE_HEIGHT_M**2)
        area = self.TILE_WIDTH_M * self.TILE_HEIGHT_M
        object.__setattr__(self, "TILE_DIAG_M", float(diag))
        object.__setattr__(self, "TILE_AREA_M2", float(area))


# --------------------------- public API ------------------------

PATHS = ProjectPaths().ensure_outputs()

# Try to infer PROMPT_DIM from existing embeddings, if available
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
    print("PROMPT_ID       :", f"{PATHS.PROMPT_ID_PREFIX}{{i:0{PATHS.PROMPT_ID_WIDTH}d}}")
    print("SPLIT_BY        :", PATHS.SPLIT_BY)

    print("--- Outputs ---")
    print("PROMPT_OUT :", PATHS.PROMPT_OUT)
    print("MAP_OUT    :", PATHS.MAP_OUT)
    print("TRAIN_OUT  :", PATHS.TRAIN_OUT)
    print("MODEL_OUT  :", PATHS.MODEL_OUT)
    print("SPLIT_OUT  :", PATHS.SPLIT_OUT)
    print("PRM_NPZ    :", PATHS.PRM_NPZ)

    print("--- Model ---")
    print("PROMPT_ENCODER:", CFG.PROMPT_ENCODER)
    print("MAP_DIM       :", CFG.MAP_DIM)
    print("PROMPT_DIM    :", CFG.PROMPT_DIM)
    print("FUSED_DIM     :", CFG.FUSED_DIM)
    print("BATCH_SIZE    :", CFG.BATCH_SIZE)
    print("VAL/TEST      :", CFG.VAL_RATIO, CFG.TEST_RATIO)
    print("SEED          :", CFG.SEED)

    print("--- Tile scale (Solution 1) ---")
    print("TILE_W/H (m)  :", CFG.TILE_WIDTH_M, CFG.TILE_HEIGHT_M)
    print("TILE_DIAG_M   :", CFG.TILE_DIAG_M)
    print("TILE_AREA_M2  :", CFG.TILE_AREA_M2)

    print("--- Operator groups (Solution 1) ---")
    print("DISTANCE_OPS  :", DISTANCE_OPS)
    print("AREA_OPS      :", AREA_OPS)
