from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Import all “global knobs” from constants (single source of truth)
from imgofup.config.constants import (
    # core schema
    OPERATOR_COL,
    INTENSITY_COL,
    MAPS_ID_COL,
    PROMPTS_TILE_ID_COL,
    PROMPTS_PROMPT_ID_COL,
    PROMPTS_TEXT_COL,
    PARAM_VALUE_COL,
    # extents
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    # filenames
    PROMPT_EMBEDDINGS_NPZ_NAME,
    PROMPTS_PARQUET_NAME,
    MAP_EMBEDDINGS_NPZ_NAME,
    MAPS_PARQUET_NAME,
    TRAIN_PAIRS_SINGLE_NAME,
    # defaults (env-overridable)
    PROMPT_ENCODER_DEFAULT,
    MAP_DIM_DEFAULT,
    PROMPT_DIM_DEFAULT,
    BATCH_SIZE_DEFAULT,
    VAL_RATIO_DEFAULT,
    TEST_RATIO_DEFAULT,
    SEED_DEFAULT,
    DISTANCE_OPS_DEFAULT,
    AREA_OPS_DEFAULT,
    USE_DYNAMIC_EXTENT_REFS_DEFAULT,
    ALLOW_FALLBACK_EXTENT_DEFAULT,
    DEFAULT_TILE_WIDTH_M_DEFAULT,
    DEFAULT_TILE_HEIGHT_M_DEFAULT,
    PARAM_STRATEGY_DEFAULT,
    QUAL_TO_QUANTILE_DEFAULT,
    QUAL_SYNONYMS_DEFAULT,
    UNIT_ALIASES_DEFAULT,
    DEFAULT_PARAM_BY_OPERATOR_DEFAULT,
    # repo path policy
    DEFAULT_DATA_DIRNAME,
)

# --------------------------- helpers ----------------------------

def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_path(key: str) -> Optional[Path]:
    """
    Read an env var path safely:
    - empty / unset -> None
    - set -> expanded + resolved Path
    Avoids the common trap Path("") == Path(".").
    """
    v = os.getenv(key, "").strip()
    return Path(v).expanduser().resolve() if v else None


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find a repo marker: src/imgofup.
    This makes paths robust even if Jupyter sets CWD weirdly.
    """
    for p in [start, *start.parents]:
        if (p / "src" / "imgofup").is_dir():
            return p
    return start


DEFAULT_PROJ_ROOT: Path = find_repo_root(Path.cwd().resolve())


def try_infer_dims(prompt_npz: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    If prompt embeddings exist, infer PROMPT_DIM.
    MAP_DIM is not inferred here (depends on your map embedding pipeline).
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


# ======================================================
# Operator groups (used for dynamic normalization of param_norm)
# ======================================================

DISTANCE_OPS = DISTANCE_OPS_DEFAULT
AREA_OPS = AREA_OPS_DEFAULT


# ======================================================
# Normalization behavior (runtime override via env)
# ======================================================

USE_DYNAMIC_EXTENT_REFS: bool = env_bool("USE_DYNAMIC_EXTENT_REFS", USE_DYNAMIC_EXTENT_REFS_DEFAULT)
ALLOW_FALLBACK_EXTENT: bool = env_bool("ALLOW_FALLBACK_EXTENT", ALLOW_FALLBACK_EXTENT_DEFAULT)


# ======================================================
# Param estimation strategy (inference-time policy)
# ======================================================

PARAM_STRATEGY: str = os.getenv("PARAM_STRATEGY", PARAM_STRATEGY_DEFAULT).strip().lower()
if PARAM_STRATEGY not in {"mlp", "hybrid"}:
    raise ValueError(f"PARAM_STRATEGY must be 'mlp' or 'hybrid', got: {PARAM_STRATEGY}")

QUAL_TO_QUANTILE = dict(QUAL_TO_QUANTILE_DEFAULT)
QUAL_SYNONYMS = dict(QUAL_SYNONYMS_DEFAULT)
UNIT_ALIASES = dict(UNIT_ALIASES_DEFAULT)
DEFAULT_PARAM_BY_OPERATOR = dict(DEFAULT_PARAM_BY_OPERATOR_DEFAULT)


# ======================================================
# Project paths (only filesystem + dataset schema overrides)
# ======================================================

@dataclass(frozen=True)
class ProjectPaths:
    """
    ProjectPaths controls:
    - where inputs live (Excel, map tiles)
    - where outputs are written (prompt/map/train/models)
    - dataset column names (if your Excel schema differs)
    - filtering flags (complete/remove)

    All values are env-overridable to make the repo portable.
    """

    # Repo root (env override wins). Otherwise auto-detected.
    PROJ_ROOT: Path = env_path("PROJ_ROOT") or DEFAULT_PROJ_ROOT

    # Data folders
    DATA_DIR: Path = env_path("DATA_DIR") or (PROJ_ROOT / DEFAULT_DATA_DIRNAME)
    INPUT_DIR: Path = env_path("INPUT_DIR") or (PROJ_ROOT / DEFAULT_DATA_DIRNAME / "input")
    OUTPUT_DIR: Path = env_path("OUTPUT_DIR") or (PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output")

    # ----------------------- Inputs (User Study Excel) -----------------------
    USER_STUDY_XLSX: Path = env_path("USER_STUDY_XLSX") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "userstudy" / "UserStudy.xlsx"
    )

    # Excel sheet name
    RESPONSES_SHEET: str = os.getenv("RESPONSES_SHEET", "Responses")

    # Excel schema columns (env-overridable)
    TILE_ID_COL: str = os.getenv("TILE_ID_COL", PROMPTS_TILE_ID_COL)
    COMPLETE_COL: str = os.getenv("COMPLETE_COL", "complete")
    REMOVE_COL: str = os.getenv("REMOVE_COL", "remove")
    TEXT_COL: str = os.getenv("TEXT_COL", "cleaned_text")
    PROMPT_ID_COL: str = os.getenv("PROMPT_ID_COL", PROMPTS_PROMPT_ID_COL)

    # Label columns in Excel (match training schema)
    PARAM_VALUE_COL: str = os.getenv("PARAM_VALUE_COL", PARAM_VALUE_COL)
    OPERATOR_COL: str = os.getenv("OPERATOR_COL", OPERATOR_COL)
    INTENSITY_COL: str = os.getenv("INTENSITY_COL", INTENSITY_COL)

    # Filtering flags
    ONLY_COMPLETE: bool = env_bool("ONLY_COMPLETE", True)
    EXCLUDE_REMOVED: bool = env_bool("EXCLUDE_REMOVED", True)

    # ----------------------- Map inputs -----------------------
    MAPS_ROOT: Path = env_path("MAPS_ROOT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "input" / "samples" / "pairs"
    )
    INPUT_MAPS_PATTERN: str = os.getenv("INPUT_MAPS_PATTERN", "*_input.geojson")

    # ----------------------- Outputs (pipelines) -----------------------
    PROMPT_OUT: Path = env_path("PROMPT_OUT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output" / "prompt_out"
    )
    MAP_OUT: Path = env_path("MAP_OUT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output" / "map_out"
    )
    TRAIN_OUT: Path = env_path("TRAIN_OUT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output" / "train_out"
    )
    MODEL_OUT: Path = env_path("MODEL_OUT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output" / "models"
    )
    SPLIT_OUT: Path = env_path("SPLIT_OUT") or (
        PROJ_ROOT / DEFAULT_DATA_DIRNAME / "output" / "train_out" / "splits"
    )

    # Canonical “single” artifacts (optional; many pipelines are experiment-scoped)
    PRM_NPZ: Path = env_path("PRM_NPZ") or (PROMPT_OUT / PROMPT_EMBEDDINGS_NPZ_NAME)
    PROMPTS_PARQUET: Path = env_path("PROMPTS_PARQUET") or (PROMPT_OUT / PROMPTS_PARQUET_NAME)

    MAPS_NPZ: Path = env_path("MAPS_NPZ") or (MAP_OUT / MAP_EMBEDDINGS_NPZ_NAME)
    MAPS_PARQUET: Path = env_path("MAPS_PARQUET") or (MAP_OUT / MAPS_PARQUET_NAME)

    TRAIN_PAIRS_PARQUET: Path = env_path("TRAIN_PAIRS_PARQUET") or (TRAIN_OUT / TRAIN_PAIRS_SINGLE_NAME)

    def ensure_outputs(self) -> "ProjectPaths":
        for p in (self.OUTPUT_DIR, self.PROMPT_OUT, self.MAP_OUT, self.TRAIN_OUT, self.MODEL_OUT, self.SPLIT_OUT):
            ensure_dir(Path(p))
        return self

    def clean_outputs(self) -> None:
        import shutil
        for d in [self.PROMPT_OUT, self.MAP_OUT, self.TRAIN_OUT, self.MODEL_OUT, self.SPLIT_OUT]:
            d = Path(d)
            if d.exists():
                print(f"🧹 Removing old directory: {d}")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print("✅ All output folders cleaned and recreated fresh.\n")


@dataclass(frozen=True)
class ModelConfig:
    """
    ModelConfig controls:
    - embedding backend selection
    - embedding dimensionalities
    - batching and split defaults
    - fallback tile scale (only used if dynamic extents are missing/degenerate)
    """

    PROMPT_ENCODER: str = os.getenv("PROMPT_ENCODER", PROMPT_ENCODER_DEFAULT)

    MAP_DIM: int = int(os.getenv("MAP_DIM", str(MAP_DIM_DEFAULT)))
    PROMPT_DIM: int = int(os.getenv("PROMPT_DIM", str(PROMPT_DIM_DEFAULT)))
    FUSED_DIM: int = 0

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", str(BATCH_SIZE_DEFAULT)))

    VAL_RATIO: float = float(os.getenv("VAL_RATIO", str(VAL_RATIO_DEFAULT)))
    TEST_RATIO: float = float(os.getenv("TEST_RATIO", str(TEST_RATIO_DEFAULT)))
    SEED: int = int(os.getenv("SEED", str(SEED_DEFAULT)))

    # Fallback tile scale (ONLY used if dynamic extents are missing/degenerate)
    DEFAULT_TILE_WIDTH_M: float = float(os.getenv("DEFAULT_TILE_WIDTH_M", str(DEFAULT_TILE_WIDTH_M_DEFAULT)))
    DEFAULT_TILE_HEIGHT_M: float = float(os.getenv("DEFAULT_TILE_HEIGHT_M", str(DEFAULT_TILE_HEIGHT_M_DEFAULT)))

    DEFAULT_TILE_DIAG_M: float = 0.0
    DEFAULT_TILE_AREA_M2: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "FUSED_DIM", int(self.MAP_DIM) + int(self.PROMPT_DIM))

        diag = math.sqrt(float(self.DEFAULT_TILE_WIDTH_M) ** 2 + float(self.DEFAULT_TILE_HEIGHT_M) ** 2)
        area = float(self.DEFAULT_TILE_WIDTH_M) * float(self.DEFAULT_TILE_HEIGHT_M)
        object.__setattr__(self, "DEFAULT_TILE_DIAG_M", float(diag))
        object.__setattr__(self, "DEFAULT_TILE_AREA_M2", float(area))


# --------------------------- public API ------------------------

PATHS = ProjectPaths().ensure_outputs()

_, inferred_prm_dim = try_infer_dims(PATHS.PRM_NPZ)
if inferred_prm_dim is not None:
    CFG = ModelConfig(PROMPT_DIM=inferred_prm_dim)
else:
    CFG = ModelConfig()


def print_summary() -> None:
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
    print("PROMPT_ID_COL   :", PATHS.PROMPT_ID_COL)
    print("PARAM_VALUE_COL :", PATHS.PARAM_VALUE_COL)
    print("OPERATOR_COL    :", PATHS.OPERATOR_COL)
    print("INTENSITY_COL   :", PATHS.INTENSITY_COL)

    print("--- Filters ---")
    print("ONLY_COMPLETE   :", PATHS.ONLY_COMPLETE)
    print("EXCLUDE_REMOVED :", PATHS.EXCLUDE_REMOVED)

    print("--- Outputs ---")
    print("PROMPT_OUT :", PATHS.PROMPT_OUT)
    print("MAP_OUT    :", PATHS.MAP_OUT)
    print("TRAIN_OUT  :", PATHS.TRAIN_OUT)
    print("MODEL_OUT  :", PATHS.MODEL_OUT)
    print("SPLIT_OUT  :", PATHS.SPLIT_OUT)
    print("PRM_NPZ    :", PATHS.PRM_NPZ)
    print("PROMPTS_PQ :", PATHS.PROMPTS_PARQUET)
    print("MAPS_NPZ   :", PATHS.MAPS_NPZ)
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

    print("--- Normalization ---")
    print("USE_DYNAMIC_EXTENT_REFS :", USE_DYNAMIC_EXTENT_REFS)
    print("ALLOW_FALLBACK_EXTENT   :", ALLOW_FALLBACK_EXTENT)
    print("Extent cols             :", EXTENT_DIAG_COL, EXTENT_AREA_COL)

    print("--- Fallback tile scale (ONLY if dynamic refs missing) ---")
    print("DEFAULT_TILE_W/H (m) :", CFG.DEFAULT_TILE_WIDTH_M, CFG.DEFAULT_TILE_HEIGHT_M)
    print("DEFAULT_TILE_DIAG_M  :", CFG.DEFAULT_TILE_DIAG_M)
    print("DEFAULT_TILE_AREA_M2 :", CFG.DEFAULT_TILE_AREA_M2)

    print("--- Operator groups ---")
    print("DISTANCE_OPS :", DISTANCE_OPS)
    print("AREA_OPS     :", AREA_OPS)

    print("--- Param estimation (inference policy) ---")
    print("PARAM_STRATEGY :", PARAM_STRATEGY)
    if PARAM_STRATEGY == "hybrid":
        print("QUAL_TO_QUANTILE:", QUAL_TO_QUANTILE)
        print("DEFAULT_PARAM_BY_OPERATOR:", DEFAULT_PARAM_BY_OPERATOR)
