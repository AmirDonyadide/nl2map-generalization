# src/imgofup/config/constants.py
from __future__ import annotations

from typing import Final, Literal

"""
IMGOFUP — Central Constants

Goal
----
Single user-facing knob file for the whole repository.
Changing values here should affect the entire pipeline consistently.

What belongs here?
------------------
- Dataset schema (column names)
- Artifact filenames and naming templates
- Feature modes and artifact lookup mappings
- Training / preprocessing defaults and hyperparameter search spaces
- MapVec feature extraction and embedding pipeline defaults
- “paths.py defaults” (values that paths.py may override via environment variables)

What should NOT belong here?
----------------------------
- Concrete filesystem paths (those live in paths.py)
- Code logic (those live in modules)
"""

# ======================================================
# 0) Fundamental numeric safety
# ======================================================

# Small epsilon used throughout to avoid divide-by-zero.
EPS_POSITIVE: Final[float] = 1e-12

# Clamp range for features that are supposed to be bounded in [0,1].
UNIT_INTERVAL_MIN: Final[float] = 0.0
UNIT_INTERVAL_MAX: Final[float] = 1.0

# Numerical stability for L2 normalization (used in several places).
L2_NORM_EPS: Final[float] = 1e-12


# ======================================================
# 1) Dataset schema: core labels and columns
# ======================================================

# Operator label column (classification target).
OPERATOR_COL: Final[str] = "operator"

# Optional intensity column (used for stratified splitting when available).
INTENSITY_COL: Final[str] = "intensity"

# Raw parameter value column from Excel (before normalization).
PARAM_VALUE_COL: Final[str] = "param_value"

# Fixed operator classes (ORDER MATTERS: defines integer class encoding).
FIXED_OPERATOR_CLASSES: Final[tuple[str, ...]] = (
    "simplify",
    "select",
    "aggregate",
    "displace",
)

# Missing-value tokens used when cleaning IDs / string cols.
NA_TOKENS: Final[set[str]] = {"", "nan", "none", "null"}

# ID normalization: zero-padding widths for stable keys.
MAP_ID_WIDTH: Final[int] = 4
PROMPT_ID_WIDTH_DEFAULT: Final[int] = 4


# ======================================================
# 2) Dataset schema: prompts artifacts (prompt embedding outputs)
# ======================================================

PROMPTS_REQUIRED_COLS: Final[list[str]] = ["tile_id", "prompt_id", "text"]

PROMPTS_TILE_ID_COL: Final[str] = "tile_id"      # prompt parquet
PROMPTS_MAP_ID_COL: Final[str] = "map_id"        # renamed downstream
PROMPTS_PROMPT_ID_COL: Final[str] = "prompt_id"
PROMPTS_TEXT_COL: Final[str] = "text"


# ======================================================
# 3) Dataset schema: maps artifacts (map embedding outputs)
# ======================================================

MAPS_ID_COL: Final[str] = "map_id"

# Extent references used for dynamic normalization of param_norm.
EXTENT_DIAG_COL: Final[str] = "extent_diag_m"
EXTENT_AREA_COL: Final[str] = "extent_area_m2"

MAPS_REQUIRED_EXTENT_COLS: Final[list[str]] = [EXTENT_DIAG_COL, EXTENT_AREA_COL]

# Traceability columns
MAP_GEOJSON_COL: Final[str] = "geojson"
MAP_N_POLYGONS_COL: Final[str] = "n_polygons"

# Extra extent columns (optional but useful)
EXTENT_MINX_COL: Final[str] = "extent_minx"
EXTENT_MINY_COL: Final[str] = "extent_miny"
EXTENT_MAXX_COL: Final[str] = "extent_maxx"
EXTENT_MAXY_COL: Final[str] = "extent_maxy"
EXTENT_WIDTH_COL: Final[str] = "extent_width_m"
EXTENT_HEIGHT_COL: Final[str] = "extent_height_m"

# Preferred extent columns to carry into train_pairs_*.parquet
EXTENT_COLS_PREFERRED: Final[list[str]] = [
    MAPS_ID_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    EXTENT_WIDTH_COL,
    EXTENT_HEIGHT_COL,
    EXTENT_MINX_COL,
    EXTENT_MINY_COL,
    EXTENT_MAXX_COL,
    EXTENT_MAXY_COL,
]


# ======================================================
# 4) Task definition & bundle format
# ======================================================

# Name of the normalized regression target column created during training-data load.
PARAM_TARGET_NAME: Final[str] = "param_norm"

# Serialized bundle schema version (increment only if structure changes incompatibly).
BUNDLE_VERSION: Final[int] = 1

# Name of normalization scheme stored in bundle metadata.
NORMALIZATION_TYPE: Final[str] = "dynamic_extent"


# ======================================================
# 5) Feature modes and artifact discovery
# ======================================================

FeatureMode = Literal[
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
]

FEATURE_MODES: Final[tuple[str, ...]] = (
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
)

# Feature mode -> (artifact folder, artifact stem).
# Used by loaders to find X_*.npy and train_pairs_*.parquet without hardcoding.
FEATURE_MODE_TO_ARTIFACTS: Final[dict[str, tuple[str, str]]] = {
    "prompt_only": ("train_out_prompt_only", "prompt_only"),
    "use_map": ("train_out_use", "use_map"),
    "map_only": ("train_out_map_only", "map_only"),
    "openai_map": ("train_out_openai", "openai_map"),
    "prompt_plus_map": ("train_out", "prompt_plus_map"),
}


# ======================================================
# 6) Naming conventions: concat + models + embedding artifacts
# ======================================================

# Concat outputs (experiment-scoped):
CONCAT_X_NAME_TEMPLATE: Final[str] = "X_{exp_name}.npy"
CONCAT_PAIRS_NAME_TEMPLATE: Final[str] = "train_pairs_{exp_name}.parquet"
CONCAT_META_NAME_TEMPLATE: Final[str] = "meta_{exp_name}.json"

# Classifier model filename:
CLS_MODEL_NAME_TEMPLATE: Final[str] = "clf_{exp_name}.joblib"

# Embedding pipeline output filenames:
PROMPT_EMBEDDINGS_NPZ_NAME: Final[str] = "prompts_embeddings.npz"
PROMPTS_PARQUET_NAME: Final[str] = "prompts.parquet"

MAP_EMBEDDINGS_NPZ_NAME: Final[str] = "maps_embeddings.npz"
MAPS_PARQUET_NAME: Final[str] = "maps.parquet"


# ======================================================
# 7) Dataset preparation policies (merge, filtering)
# ======================================================

# Default merge keys for labels onto pairs.
DEFAULT_KEY_COLS: Final[tuple[str, str]] = (MAPS_ID_COL, PROMPTS_PROMPT_ID_COL)

# Minimum fraction of rows that must receive labels after merge.
LABEL_MERGE_MIN_HIT_RATE: Final[float] = 0.5

# Whether training-data loading requires text column by default.
TRAIN_REQUIRE_TEXT_DEFAULT: Final[bool] = True

# Required keys in train_pairs_*.parquet.
PAIRS_REQUIRED_KEY_COLS: Final[tuple[str, str]] = (PROMPTS_PROMPT_ID_COL, MAPS_ID_COL)

# Fallback prompt_id column name in Excel if not configured in paths.
LABELS_EXCEL_PROMPT_ID_FALLBACK: Final[str] = "prompt_id"


# ======================================================
# 8) Weighting policy (classification)
# ======================================================

CLASS_WEIGHT_MODE_DEFAULT: Final[str] = "balanced"

# If True: each map contributes ~1 total weight (1 / count(map_id)).
USE_MAP_WEIGHT_DEFAULT: Final[bool] = True


# ======================================================
# 9) Classifier training defaults (MLP random search)
# ======================================================

HIDDEN_LAYER_CANDIDATES: Final[list[tuple[int, ...]]] = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
    (256, 128, 64),
]

BATCH_CANDIDATES: Final[list[int]] = [16, 32, 64, 128]

CLS_SEARCH_N_ITER_DEFAULT: Final[int] = 50
CLS_SEARCH_N_SPLITS_DEFAULT: Final[int] = 5
CLS_SEARCH_SEED_DEFAULT: Final[int] = 42
CLS_SEARCH_VERBOSE_DEFAULT: Final[bool] = True
CLS_SEARCH_TOPK: Final[int] = 5


# ======================================================
# 10) Regressor training defaults (per-operator MLPRegressor)
# ======================================================

REG_USE_LOG1P_DEFAULT: Final[bool] = False

REG_N_SPLITS_DEFAULT: Final[int] = 5
REG_N_ITER_DEFAULT: Final[int] = 40
REG_RANDOM_STATE_DEFAULT: Final[int] = 42
REG_VERBOSE_DEFAULT: Final[int] = 1

REG_MIN_SAMPLES_PER_CLASS: Final[int] = 10

REG_MLP_BASE_PARAMS: Final[dict] = {
    "activation": "relu",
    "solver": "adam",
    "learning_rate": "adaptive",
    "early_stopping": False,
    "max_iter": 2000,
    "tol": 1e-3,
    "verbose": False,
    "batch_size": "auto",
}

REG_HIDDEN_LAYER_CANDIDATES: Final[list[tuple[int, ...]]] = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
]

REG_ALPHA_BOUNDS: Final[tuple[float, float]] = (1e-6, 3e-2)
REG_LR_INIT_BOUNDS: Final[tuple[float, float]] = (1e-4, 3e-3)

REG_SCORING: Final[str] = "neg_root_mean_squared_error"
REG_N_JOBS: Final[int] = -1

REG_REFIT_MAX_ITER: Final[int] = 2000
REG_REFIT_EARLY_STOPPING: Final[bool] = False
REG_TOL_DEFAULT: Final[float] = 1e-3


# ======================================================
# 11) Preprocessing defaults (map/prompt)
# ======================================================

# Map preprocessing:
MAP_CLIP_Q_DEFAULT: Final[tuple[int, int]] = (5, 95)
MAP_IMPUTE_STRATEGY_DEFAULT: Final[str] = "median"
MAP_ROBUST_QRANGE_DEFAULT: Final[tuple[int, int]] = (5, 95)
MAP_VAR_EPS_DEFAULT: Final[float] = 1e-12


# ======================================================
# 12) Pipeline verbosity defaults
# ======================================================

CONCAT_VERBOSITY_DEFAULT: Final[int] = 1
MAP_EMBED_VERBOSITY_DEFAULT: Final[int] = 1
PROMPT_EMBED_VERBOSITY_DEFAULT: Final[int] = 1

PROMPT_EMBED_L2_NORMALIZE_DEFAULT: Final[bool] = True
PROMPT_EMBED_SAVE_CSV_DEFAULT: Final[bool] = False


# ======================================================
# 13) Dataset splitting defaults / policies
# ======================================================

SPLIT_USE_INTENSITY_FOR_STRAT_DEFAULT: Final[bool] = True
SPLIT_SEED_DEFAULT: Final[int] = 42
SPLIT_VAL_RATIO_DEFAULT: Final[float] = 0.15
SPLIT_TEST_RATIO_DEFAULT: Final[float] = 0.15
SPLIT_MAX_ATTEMPTS_DEFAULT: Final[int] = 500
SPLIT_VERBOSE_DEFAULT: Final[bool] = True

SPLIT_TINY_SINGLE_MAP_THRESHOLD: Final[int] = 2
SPLIT_PROMPTS_PER_MAP_MULTI_THRESHOLD: Final[int] = 1

SPLIT_STRAT_COL_NAME: Final[str] = "_strat"
SPLIT_STRAT_DELIM: Final[str] = "__"
SPLIT_SSS2_SEED_OFFSET: Final[int] = 999
SPLIT_JSON_INDENT: Final[int] = 2


# ======================================================
# 14) MapVec — pooling: per-polygon table → one map embedding
# ======================================================

MAP_POOL_EXCLUDE_COLS_DEFAULT: Final[tuple[str, ...]] = ("id",)
MAP_POOL_STATS_DEFAULT: Final[tuple[str, ...]] = ("mean", "std", "min", "max")
MAP_POOL_QUANTILES_DEFAULT: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)
MAP_POOL_ADD_GLOBALS_DEFAULT: Final[bool] = True
MAP_POOL_QUANTILE_METHOD: Final[str] = "linear"

POLY_CENTROID_X_COL: Final[str] = "centroid_x"
POLY_CENTROID_Y_COL: Final[str] = "centroid_y"
POLY_AREA_COL: Final[str] = "area"

MAP_POOL_GLOBAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "poly_count",
    "poly_spread_w",
    "poly_spread_h",
    "coverage_ratio",
)


# ======================================================
# 15) MapVec — polygon feature extraction (hand-crafted)
# ======================================================

POLY_FEATURE_ID_COL: Final[str] = "id"

POLY_FEATURE_ORDER: Final[tuple[str, ...]] = (
    "area",
    "perimeter",
    "centroid_x",
    "centroid_y",
    "circularity",
    "axis_ratio",
    "convexity",
    "rectangularity",
    "neighbor_count",
    "bbox_width",
    "bbox_height",
    "orient_sin",
    "orient_cos",
    "eq_diameter",
    "eccentricity",
    "has_hole",
    "reflex_ratio",
    "nn_dist_median",
    "knn1",
    "knn3",
    "density_r05",
    "density_r10",
    "extent_fill",
)

POLY_BY_DIAG_FEATURES: Final[tuple[str, ...]] = (
    "perimeter",
    "eq_diameter",
    "nn_dist_median",
    "knn1",
    "knn3",
    "bbox_width",
    "bbox_height",
)

POLY_NORM_MODE_DEFAULT: Final[str] = "extent"
POLY_NORM_FIXED_WH_DEFAULT: Final[tuple[float, float]] = (400.0, 400.0)

POLY_ECC_EPS: Final[float] = 1e-12
POLY_ECC_MAX: Final[float] = 0.999999

POLY_DENSITY_R05_FRAC: Final[float] = 0.05
POLY_DENSITY_R10_FRAC: Final[float] = 0.10

POLY_CLIP_QHI: Final[float] = 0.995

SHAPELY_WARN_MODULE_PREDICATES: Final[str] = r"shapely\.predicates"
SHAPELY_WARN_MODULE_SETOPS: Final[str] = r"shapely\.set_operations"


# ======================================================
# 16) MapVec — map embedding pipeline defaults (GeoJSON → pooled embedding)
# ======================================================

MAP_EMBED_ROOT_DEFAULT: Final[str] = "samples/pairs"
MAP_EMBED_PATTERN_DEFAULT: Final[str] = "*_input.geojson"
MAP_EMBED_OUTDIR_DEFAULT: Final[str] = "map_out"

MAP_EMBED_SAVE_CSV_DEFAULT: Final[bool] = False

MAP_EMBED_PROJECT_IF_GEOGRAPHIC: Final[bool] = True
MAP_EMBED_GEOGRAPHIC_CRS_EPSG: Final[int] = 4326
MAP_EMBED_METRIC_CRS_EPSG: Final[int] = 3857

MAP_POLY_AREA_EPS: Final[float] = 1e-12

MAP_EMBED_WARNINGS_TO_IGNORE: Final[tuple[str, ...]] = (
    "invalid value encountered in within",
    "invalid value encountered in contains",
    "invalid value encountered in buffer",
)

MAPS_CSV_NAME: Final[str] = "maps.csv"
MAPS_META_JSON_NAME: Final[str] = "meta.json"
MAPS_FEATURE_NAMES_JSON_NAME: Final[str] = "feature_names.json"


# ======================================================
# 17) Prompt embedding pipeline defaults (Excel → embeddings)
# ======================================================

PROJECT_ROOT_MARKER_LEVELS_UP: Final[int] = 3

DEFAULT_DATA_DIRNAME: Final[str] = "data"
DEFAULT_OUTPUT_DIRNAME: Final[str] = "output"

MAPVEC_DATA_DIR_ENVVAR: Final[str] = "MAPVEC_DATA_DIR"
DOTENV_FILENAME: Final[str] = ".env"

PROMPT_EMBED_LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

PROMPTS_EXCEL_SHEET_DEFAULT: Final[str] = "Responses"
EXCEL_COMPLETE_COL_DEFAULT: Final[str] = "complete"
EXCEL_REMOVE_COL_DEFAULT: Final[str] = "remove"
EXCEL_TEXT_COL_DEFAULT: Final[str] = "cleaned_text"
EXCEL_ONLY_COMPLETE_DEFAULT: Final[bool] = True
EXCEL_EXCLUDE_REMOVED_DEFAULT: Final[bool] = True

PROMPTS_META_JSON_NAME: Final[str] = "meta.json"
PROMPTS_EMBEDDINGS_CSV_NAME: Final[str] = "embeddings.csv"

PROMPT_ENCODER_CHOICES: Final[tuple[str, ...]] = ("dan", "transformer", "openai-small", "openai-large")

USE_KAGGLE_MODEL_IDS: Final[dict[str, str]] = {
    "dan": "google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2",
    "transformer": "google/universal-sentence-encoder-large/tensorFlow2/universal-sentence-encoder-large/2",
}

USE_MODEL_DIR_DAN: Final[str] = "input/model_dan"
USE_MODEL_DIR_TRANSFORMER: Final[str] = "input/model_transformer"

USE_BATCH_SIZE_DEFAULT: Final[int] = 512
OPENAI_BATCH_SIZE_DEFAULT: Final[int] = 256

OPENAI_MODEL_NAME_SMALL: Final[str] = "text-embedding-3-small"
OPENAI_MODEL_NAME_LARGE: Final[str] = "text-embedding-3-large"

OPENAI_PROMPT_PREFIX: Final[str] = "Cartographic map generalization instruction: "


# ======================================================
# 18) Paths.py defaults (env-overridable runtime configuration)
# ======================================================

# Default prompt encoder used by ModelConfig if not set via env/cfg.
PROMPT_ENCODER_DEFAULT: Final[str] = "openai-small"

# Default embedding dims used if not inferred from artifacts.
MAP_DIM_DEFAULT: Final[int] = 165
PROMPT_DIM_DEFAULT: Final[int] = 512

# Default batch size for embedding calls.
BATCH_SIZE_DEFAULT: Final[int] = 512

# Default split ratios and seed (used by ModelConfig unless overridden).
VAL_RATIO_DEFAULT: Final[float] = 0.15
TEST_RATIO_DEFAULT: Final[float] = 0.15
SEED_DEFAULT: Final[int] = 42

# Operator groups used to normalize PARAM_TARGET_NAME.
DISTANCE_OPS_DEFAULT: Final[tuple[str, ...]] = ("aggregate", "displace", "simplify")
AREA_OPS_DEFAULT: Final[tuple[str, ...]] = ("select",)

# Extent normalization behavior defaults (env override in paths.py).
USE_DYNAMIC_EXTENT_REFS_DEFAULT: Final[bool] = True
ALLOW_FALLBACK_EXTENT_DEFAULT: Final[bool] = True

# Fallback tile scale used only if dynamic extents are missing.
DEFAULT_TILE_WIDTH_M_DEFAULT: Final[float] = 400.0
DEFAULT_TILE_HEIGHT_M_DEFAULT: Final[float] = 400.0

# Param estimation strategy used at inference time.
PARAM_STRATEGY_DEFAULT: Final[str] = "mlp"

QUAL_TO_QUANTILE_DEFAULT: Final[dict[str, float]] = {
    "very_small": 0.10,
    "small": 0.25,
    "medium": 0.50,
    "large": 0.75,
    "very_large": 0.90,
}

QUAL_SYNONYMS_DEFAULT: Final[dict[str, list[str]]] = {
    "very_small": ["very small", "tiny", "minuscule", "very little"],
    "small":      ["small", "smaller", "minor", "little"],
    "medium":     ["medium", "moderate", "average", "normal"],
    "large":      ["large", "bigger", "big", "major"],
    "very_large": ["very large", "huge", "massive", "giant"],
}

UNIT_ALIASES_DEFAULT: Final[dict[str, list[str]]] = {
    "m":  ["m", "meter", "meters", "metre", "metres"],
    "km": ["km", "kilometer", "kilometers", "kilometre", "kilometres"],
    "m2": ["m2", "m^2", "sqm", "sq m", "sq. m", "square meter", "square meters",
           "square metre", "square metres", "meter^2", "metre^2"],
    "km2": ["km2", "km^2", "square kilometer", "square kilometers",
            "square kilometre", "square kilometres"],
    "%":  ["%", "percent", "percentage", "per cent"],
}

DEFAULT_PARAM_BY_OPERATOR_DEFAULT: Final[dict[str, float]] = {
    "aggregate": 5.0,
    "displace":  5.0,
    "simplify":  5.0,
    "select":   50.0,
}

# If you use a single non-experiment train_pairs output, this is the default filename.
TRAIN_PAIRS_SINGLE_NAME: Final[str] = "train_pairs.parquet"


# ======================================================
# 19) Legacy concat CLI script (optional)
# ======================================================
# Keep only if you still use the old mapvec/concat CLI.

CONCAT_X_CONCAT_NAME: Final[str] = "X_concat.npy"
CONCAT_TRAIN_PAIRS_NAME: Final[str] = "train_pairs.parquet"
CONCAT_META_JSON_NAME: Final[str] = "meta.json"

CONCAT_SAVE_BLOCKS_NAMES: Final[dict[str, str]] = {
    "X_map": "X_map.npy",
    "X_prompt": "X_prompt.npy",
    "map_ids": "map_ids.npy",
    "prompt_ids": "prompt_ids.npy",
}

CONCAT_FAIL_ON_MISSING_DEFAULT: Final[bool] = False
CONCAT_DROP_DUPES_DEFAULT: Final[bool] = False
CONCAT_L2_PROMPT_DEFAULT: Final[bool] = False
CONCAT_SAVE_BLOCKS_DEFAULT: Final[bool] = False
CONCAT_L2_EPS: Final[float] = 1e-12
CONCAT_PAD_NUMERIC_PROMPT_IDS_DEFAULT: Final[bool] = False


# ======================================================
# 20) Backwards compatibility aliases (temporary)
# ======================================================
# Some older modules might still import FEATURE_MODE_TO_TRAIN_FOLDER.
# Keep this alias until you update all imports, then remove it.
FEATURE_MODE_TO_TRAIN_FOLDER = FEATURE_MODE_TO_ARTIFACTS

# ======================================================
# 21) User-study sample generation (OSM → tile pairs)
# ======================================================
# These constants control the *dataset creation* notebook / scripts that:
#   - read a .osm.pbf extract
#   - create a tile grid
#   - pick the densest tiles (TOP_K)
#   - apply one generalization operator (aggregate/simplify/displace/select)
#   - render input/target PNGs
#   - write per-tile metadata (operator, intensity, param_value, etc.)
#
# IMPORTANT:
# - These are NOT used during model training/inference directly.
# - They exist so you can reproduce the user-study dataset generation exactly.

# ----------------------
# 21.1 Rendering defaults
# ----------------------

# DPI used for rendered PNG frames in the user-study dataset.
USERSTUDY_RENDER_DPI_DEFAULT: Final[int] = 300

# Figure size (in inches) used for rendering.
# Together with TILE_SIZE_M, this determines meters-per-pixel.
USERSTUDY_RENDER_FIG_INCH_DEFAULT: Final[float] = 6.0

# Whether to draw a tile frame / boundary (if your renderer supports it).
USERSTUDY_DRAW_TILE_FRAME_DEFAULT: Final[bool] = True

# Whether to include road layer in the rendered PNGs.
# Keep False if the user study is building-only to reduce visual clutter.
USERSTUDY_SHOW_ROADS_DEFAULT: Final[bool] = False

# Visual styling (matplotlib / geopandas plotting)
USERSTUDY_ROAD_COLOR_DEFAULT: Final[str] = "black"
USERSTUDY_BUILDING_FACE_DEFAULT: Final[str] = "gray"
USERSTUDY_BUILDING_EDGE_DEFAULT: Final[str] = "white"


# ----------------------
# 21.2 Spatial defaults
# ----------------------

# CRS used for user-study tile creation (must be metric; meters).
# Example: EPSG:25832 for Germany UTM 32N.
USERSTUDY_TARGET_CRS_DEFAULT: Final[str] = "EPSG:25832"

# Tile size in meters (width == height).
# This is the *ground truth* scale used during dataset generation.
USERSTUDY_TILE_SIZE_M_DEFAULT: Final[int] = 400

# When selecting tiles from a grid, keep only the TOP_K tiles by building count.
# This is a dataset-size knob (affects how many samples you create).
USERSTUDY_TOPK_TILES_DEFAULT: Final[int] = 824

# Random seed used for:
# - shuffling tile ids
# - balancing assignment to operator/intensity
# - any stochastic operator behavior (if you use random anywhere)
USERSTUDY_SEED_DEFAULT: Final[int] = 42


# ----------------------
# 21.3 OSM feature filters (optional roads)
# ----------------------
# Only relevant if USERSTUDY_SHOW_ROADS_DEFAULT=True.

# OSM road classes considered "important" for display.
# You can extend or shrink this set depending on what you want participants to see.
USERSTUDY_IMPORTANT_ROADS_DEFAULT: Final[set[str]] = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "residential", "unclassified",
}

# Stroke width per OSM highway class when rendering (matplotlib linewidth units).
USERSTUDY_ROAD_WIDTH_MAP_DEFAULT: Final[dict[str, float]] = {
    "motorway": 2.0, "trunk": 1.8, "primary": 1.6,
    "secondary": 1.4, "tertiary": 1.2,
    "residential": 0.8, "unclassified": 0.8,
    "motorway_link": 1.4, "trunk_link": 1.3,
    "primary_link": 1.2, "secondary_link": 1.1, "tertiary_link": 1.0,
}


# ----------------------
# 21.4 Operator / intensity design
# ----------------------

# Operators used in the user study generation.
# These must align with FIXED_OPERATOR_CLASSES (same spelling) if you train on them.
USERSTUDY_OPERATORS_DEFAULT: Final[tuple[str, ...]] = (
    "aggregate",
    "simplify",
    "displace",
    "select",
)

# Intensity levels for the user study.
USERSTUDY_INTENSITIES_DEFAULT: Final[tuple[str, ...]] = (
    "low",
    "medium",
    "high",
)

# Target effects per operator/intensity
# These targets are used by your parameter-search helpers to pick a param_value
# that produces roughly the intended amount of change.
#
# Interpretation:
# - select: fraction of polygons removed (by area threshold)
# - aggregate: fraction of polygons merged (count reduction proxy)
# - simplify: fraction of polygons that changed “visibly” (area-change heuristic)
# - displace: fraction of polygons moved beyond a distance tolerance
USERSTUDY_SELECT_REMOVAL_TARGET_DEFAULT: Final[dict[str, float]] = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}

USERSTUDY_AGG_MERGE_TARGET_DEFAULT: Final[dict[str, float]] = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}

USERSTUDY_SIMPLIFY_CHANGE_TARGET_DEFAULT: Final[dict[str, float]] = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}

USERSTUDY_DISPLACE_CHANGE_TARGET_DEFAULT: Final[dict[str, float]] = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}


# ----------------------
# 21.5 Parameter-search / heuristic tuning
# ----------------------
# These constants control the internal search loops that choose a per-tile
# param_value that best matches the targets above.

# Number of steps in the grid search when choosing aggregate parameter per tile.
USERSTUDY_AGG_PARAM_SEARCH_STEPS_DEFAULT: Final[int] = 8

# Number of steps in the grid search when choosing simplify parameter per tile.
USERSTUDY_SIMPLIFY_PARAM_SEARCH_STEPS_DEFAULT: Final[int] = 6

# Number of steps in the grid search when choosing displace parameter per tile.
USERSTUDY_DISPLACE_PARAM_SEARCH_STEPS_DEFAULT: Final[int] = 6

# Simplify “changed” heuristic threshold:
# A polygon is counted as "changed" if relative area difference exceeds this value.
USERSTUDY_SIMPLIFY_AREA_REL_TOL_DEFAULT: Final[float] = 0.01

# Displace “changed” heuristic threshold (meters):
# A polygon is counted as moved if centroid displacement > move_tol.
USERSTUDY_DISPLACE_MOVE_TOL_M_DEFAULT: Final[float] = 0.30

# Displace operator solver defaults (these affect how strong the displacement is).
USERSTUDY_DISPLACE_ITERS_DEFAULT: Final[int] = 15
USERSTUDY_DISPLACE_STEP_DEFAULT: Final[float] = 0.60
USERSTUDY_DISPLACE_MAX_TOTAL_DEFAULT: Final[float] = 10.0

# Aggregate operator buffering defaults:
# join_style=2 (mitre), cap_style=2 (flat), resolution=1 are "angular" defaults.
USERSTUDY_AGG_JOIN_STYLE_DEFAULT: Final[int] = 2
USERSTUDY_AGG_CAP_STYLE_DEFAULT: Final[int] = 2
USERSTUDY_AGG_MITRE_LIMIT_DEFAULT: Final[float] = 5.0
USERSTUDY_AGG_BUFFER_RESOLUTION_DEFAULT: Final[int] = 1


# ----------------------
# 21.6 Derived helper defaults
# ----------------------

# When simplifying before rendering, you compute a tolerance based on pixel size:
# meters_per_pixel = TILE_SIZE_M / (FIG_INCH * DPI)
# and then use something like meters_per_pixel * factor.
#
# This factor controls how aggressively you simplify for rendering speed/clarity.
USERSTUDY_RENDER_TOL_FACTOR_DEFAULT: Final[float] = 0.75

# ======================================================
# User study — tiling & sampling defaults
# ======================================================

# Column name used to identify tiles throughout user-study generation
# This column is added during tiling and used consistently in metadata,
# GeoJSONs, and downstream notebooks.
USERSTUDY_TILE_ID_COL_DEFAULT: Final[str] = "tile_id"

# ============================================================
# User Study — sample generation & rendering defaults
# ============================================================



# these already exist elsewhere in your repo, but we re-export conceptually
# OPERATOR_COL = "operator"
# INTENSITY_COL = "intensity"
# PARAM_VALUE_COL = "param_value"

# ---- sampling / balancing ----
USERSTUDY_TOP_K_TILES_DEFAULT = 800


# ---- output locations (relative to repo root / data) ----
USERSTUDY_SAMPLES_DIR_DEFAULT = "data/input/samples/pairs"
USERSTUDY_METADATA_DIR_DEFAULT = "data/input/samples/metadata"

# ---- metadata filenames ----
USERSTUDY_META_CSV_NAME_DEFAULT = "meta.csv"
USERSTUDY_META_XLSX_NAME_DEFAULT = "meta.xlsx"

# ---- per-sample file naming ----
USERSTUDY_INPUT_GEOJSON_SUFFIX_DEFAULT = "_input.geojson"
USERSTUDY_TARGET_GEOJSON_SUFFIX_DEFAULT = "_generalized.geojson"

USERSTUDY_INPUT_PNG_PREFIX_DEFAULT = "input_"
USERSTUDY_TARGET_PNG_PREFIX_DEFAULT = "generalized_"

# ---- rendering defaults ----
USERSTUDY_RENDER_PNG_DEFAULT = True

# ---- map normalization ----
USERSTUDY_RENDER_TARGET_SIZE_M = 400.0

# ==================================================
# User-study (sample generation) defaults
# ==================================================

USERSTUDY_PBF_PATH_DEFAULT = "../data/input/koeln-regbez-250927.osm.pbf"
USERSTUDY_BBOX_DEFAULT = [7.00, 50.65, 7.20, 50.82]

USERSTUDY_TOP_K_TILES_DEFAULT = 824


USERSTUDY_SAMPLES_DIR_DEFAULT = "../data/input/samples/pairs_new"
USERSTUDY_METADATA_DIR_DEFAULT = "../data/input/samples/metadata_new"

USERSTUDY_META_CSV_NAME_DEFAULT = "meta.csv"
USERSTUDY_META_XLSX_NAME_DEFAULT = "meta.xlsx"
