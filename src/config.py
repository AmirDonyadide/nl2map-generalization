# src/config.py
from __future__ import annotations

import os
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

# --------------------------- config ----------------------------

@dataclass(frozen=True)
class ProjectPaths:
    # Project root that contains `src/` and `data/`
    PROJ_ROOT: Path = Path(os.getenv("PROJ_ROOT", "../")).resolve()

    # Data
    DATA_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data")
    INPUT_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "input")
    OUTPUT_DIR: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output")

    # Inputs
    PROMPTS_CSV: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "input" / "prompts.csv")  # columns: prompt_id,text
    PAIRS_CSV: Path   = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "input" / "pairs.csv")    # columns: map_id,prompt_id
    MAPS_ROOT: Path   = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "input" / "samples" / "pairs")

    # File patterns
    INPUT_MAPS_PATTERN: str = os.getenv("INPUT_MAPS_PATTERN", "*_input.geojson")

    # Outputs
    PROMPT_OUT: Path = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "prompt_out")
    MAP_OUT: Path    = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "map_out")
    TRAIN_OUT: Path  = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "train_out")
    MODEL_OUT: Path  = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "models")
    SPLIT_OUT: Path  = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "train_out" / "splits")

    # Precomputed embeddings
    PRM_NPZ: Path    = (Path(os.getenv("PROJ_ROOT", "../")).resolve() / "data" / "output" / "prompt_out" / "prompts_embeddings.npz")

    # -------------------------------------------------------------
    def ensure_outputs(self) -> "ProjectPaths":
        """Create output directories if missing."""
        from pathlib import Path
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
      - 'openai-small', 'openai-large', etc. → LLM embeddings
      - or any other string you use in prompt_embeddings.py --model
    """
    # Prompt encoder name (generic, not just USE)
    PROMPT_ENCODER: str = "openai-small"
    #PROMPT_ENCODER: str = "dan"

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

    def __post_init__(self):
        # Keep FUSED_DIM consistent with MAP_DIM and PROMPT_DIM
        object.__setattr__(self, "FUSED_DIM", self.MAP_DIM + self.PROMPT_DIM)




# --------------------------- public API ------------------------

# Instantiate paths, ensure outputs exist
PATHS = ProjectPaths().ensure_outputs()

# Try to infer PROMPT_DIM from existing embeddings, if available
_, inferred_prm_dim = try_infer_dims(PATHS.PRM_NPZ)

if inferred_prm_dim is not None:
    CFG = ModelConfig(PROMPT_DIM=inferred_prm_dim)
else:
    CFG = ModelConfig()


# Optional: quick sanity warnings (no hard failure)
def print_summary():
    print("=== CONFIG SUMMARY ===")
    print("PROJ_ROOT  :", PATHS.PROJ_ROOT)
    print("DATA_DIR   :", PATHS.DATA_DIR)
    print("INPUT_DIR  :", PATHS.INPUT_DIR)
    print("OUTPUT_DIR :", PATHS.OUTPUT_DIR)
    print("MAPS_ROOT  :", PATHS.MAPS_ROOT)
    print("INPUT PAT. :", PATHS.INPUT_MAPS_PATTERN)
    print("PROMPTS_CSV:", PATHS.PROMPTS_CSV)
    print("PAIRS_CSV  :", PATHS.PAIRS_CSV)
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
    print("BATCH_SIZE :", CFG.BATCH_SIZE)
    print("VAL/TEST   :", CFG.VAL_RATIO, CFG.TEST_RATIO)
    print("SEED       :", CFG.SEED)