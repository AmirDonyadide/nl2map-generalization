# src/config/constants.py
from __future__ import annotations
from typing import Final, Literal

"""
Central configuration file for the thesis repository.

Goal
----
All user-adjustable constants live here. Other modules should only IMPORT from this file.
Changing values here should affect the whole pipeline consistently.

Conventions
-----------
- *_COL constants define column names used across dataframes / parquet outputs.
- *_NAME / *_TEMPLATE constants define file naming conventions.
- *_DEFAULT constants define default behavior for training / preprocessing / pipelines.
"""

# ======================================================
# 1) Core label space (classification task)
# ======================================================

# Column name for the operator label in training dataframes.
OPERATOR_COL = "operator"

# Optional column used for stratification / analysis (if present).
# Used in splitting to stratify by operator × intensity when enabled.
INTENSITY_COL = "intensity"

# Fixed operator classes for the classifier (ORDER MATTERS!).
# This ordering defines the integer encoding of classes across the whole project.
FIXED_OPERATOR_CLASSES = (
    "simplify",
    "select",
    "aggregate",
    "displace",
)

# ======================================================
# 2) Target definition & bundle format
# ======================================================

# Name of the normalized regression target produced during training data loading.
# If you change this, you must regenerate training data / bundles that rely on it.
PARAM_TARGET_NAME = "param_norm"

# Version for serialized bundles produced by the training pipeline.
# Increase if you change bundle structure in an incompatible way.
BUNDLE_VERSION = 1

# Name of the normalization scheme stored in the bundle metadata.
# This is informational unless other code branches on it.
NORMALIZATION_TYPE = "dynamic_extent"

# ======================================================
# 3) Missing-value & ID normalization policy
# ======================================================

# Tokens treated as "missing" when cleaning ID / string columns.
# Extend this if your Excel/CSV exports contain other missing markers (e.g., "na", "-", "n/a").
NA_TOKENS = {"", "nan", "none", "null"}

# Zero-padding width for map identifiers (e.g., 1 -> "0001").
# Must match how map tiles are keyed in your artifacts and study files.
MAP_ID_WIDTH = 4

# Zero-padding width for prompt identifiers (e.g., 7 -> "0007").
# Must match how prompt IDs appear in prompts.parquet and embedding IDs.
PROMPT_ID_WIDTH_DEFAULT = 4

# ======================================================
# 4) Input schema: prompts artifacts (produced by prompt embedding pipeline)
# ======================================================

# Required columns expected in prompts.parquet produced by the prompt embedding pipeline.
PROMPTS_REQUIRED_COLS = ["tile_id", "prompt_id", "text"]

# Column names in prompts.parquet / merged training data.
# tile_id is renamed to map_id during processing.
PROMPTS_TILE_ID_COL = "tile_id"
PROMPTS_MAP_ID_COL = "map_id"
PROMPTS_PROMPT_ID_COL = "prompt_id"
PROMPTS_TEXT_COL = "text"

# ======================================================
# 5) Input schema: maps artifacts (produced by map embedding pipeline)
# ======================================================

# Canonical map id column name used in maps.parquet and merged frames.
MAPS_ID_COL = "map_id"

# Extent reference columns (must exist for dynamic normalization).
# Used to compute PARAM_TARGET_NAME (distance ops normalized by diag, area ops by area).
EXTENT_DIAG_COL = "extent_diag_m"
EXTENT_AREA_COL = "extent_area_m2"

# Required map extent columns for training (minimum needed).
MAPS_REQUIRED_EXTENT_COLS = [EXTENT_DIAG_COL, EXTENT_AREA_COL]

# Path column stored with each map row (for traceability/debugging).
MAP_GEOJSON_COL = "geojson"

# Optional column added during map embedding run for metadata/analysis.
MAP_N_POLYGONS_COL = "n_polygons"

# ======================================================
# 6) Output naming conventions (concat + training artifacts)
# ======================================================

# Concat outputs (feature matrices + join table + metadata).
# Change to enforce different naming conventions across experiments.
CONCAT_X_NAME_TEMPLATE = "X_{exp_name}.npy"
CONCAT_PAIRS_NAME_TEMPLATE = "train_pairs_{exp_name}.parquet"
CONCAT_META_NAME_TEMPLATE = "meta_{exp_name}.json"

# Classifier model artifact naming.
CLS_MODEL_NAME_TEMPLATE = "clf_{exp_name}.joblib"

# ======================================================
# 7) Artifact discovery / folder conventions
# ======================================================

# Folder mapping used when resolving artifacts for a given feature_mode.
# Tuple = (folder_name, stem_used_for_files)
FEATURE_MODE_TO_TRAIN_FOLDER = {
    "prompt_only": ("train_out_prompt_only", "prompt_only"),
    "use_map": ("train_out_use", "use_map"),
    "map_only": ("train_out_map_only", "map_only"),
    "openai_map": ("train_out_openai", "openai_map"),
    "prompt_plus_map": ("train_out", "prompt_plus_map"),
}

# Default keys used to merge labels onto pairs.
DEFAULT_KEY_COLS = (MAPS_ID_COL, PROMPTS_PROMPT_ID_COL)

# Minimum fraction of rows that must successfully receive labels after merging.
# If you lower this, you may silently train on many unlabeled rows; if you increase it,
# you may get errors more often when keys mismatch between Excel and parquet.
LABEL_MERGE_MIN_HIT_RATE = 0.5

# ======================================================
# 8) Training-data loading policy
# ======================================================

# Whether the training loader requires the "text" column to exist.
# Set False if you want to run map-only experiments without prompt text.
TRAIN_REQUIRE_TEXT_DEFAULT = True

# Required join keys in train_pairs_*.parquet.
PAIRS_REQUIRED_KEY_COLS = (PROMPTS_PROMPT_ID_COL, MAPS_ID_COL)

# Fallback name for prompt id column in Excel labels file if paths.PROMPT_ID_COL is not set.
LABELS_EXCEL_PROMPT_ID_FALLBACK = "prompt_id"

# ======================================================
# 9) Weighting policy (classification training)
# ======================================================

# Default mode for sklearn.compute_class_weight.
# "balanced" compensates for operator class imbalance in TRAIN.
CLASS_WEIGHT_MODE_DEFAULT = "balanced"

# Whether to apply map-level sample weighting by default.
# If True: each map contributes ~1 total weight (1/count(map_id)).
USE_MAP_WEIGHT_DEFAULT = True

# ======================================================
# 10) MLP classifier random search space (classification)
# ======================================================

# Candidate hidden layer configurations for MLPClassifier random search.
HIDDEN_LAYER_CANDIDATES = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
    (256, 128, 64),
]

# Candidate batch sizes for MLPClassifier random search.
BATCH_CANDIDATES = [16, 32, 64, 128]

# Defaults controlling the classifier search procedure (train_classifier.py).
CLS_SEARCH_N_ITER_DEFAULT = 50       # number of random parameter samples to evaluate
CLS_SEARCH_N_SPLITS_DEFAULT = 5      # number of folds for StratifiedGroupKFold
CLS_SEARCH_SEED_DEFAULT = 42         # RNG seed for reproducibility
CLS_SEARCH_VERBOSE_DEFAULT = True    # print progress + reports
CLS_SEARCH_TOPK = 5                 # how many top candidates to print/save

# ======================================================
# 11) Regressor training defaults (per-operator MLPRegressor)
# ======================================================

# Whether to apply log1p transform to the regression target before training.
# Only valid if the target is non-negative.
REG_USE_LOG1P_DEFAULT = False

# RandomizedSearchCV defaults.
REG_N_SPLITS_DEFAULT = 5
REG_N_ITER_DEFAULT = 40
REG_RANDOM_STATE_DEFAULT = 42
REG_VERBOSE_DEFAULT = 1

# Minimum samples per operator to train a regressor; otherwise the operator is skipped.
REG_MIN_SAMPLES_PER_CLASS = 10

# Base regressor fixed params (except random_state).
# These define the training behavior for each operator regressor.
REG_MLP_BASE_PARAMS = {
    "activation": "relu",
    "solver": "adam",
    "learning_rate": "adaptive",
    "early_stopping": False,
    "max_iter": 2000,
    "tol": 1e-3,
    "verbose": False,
    "batch_size": "auto",
}

# Hyperparameter search space for MLPRegressor (RandomizedSearchCV).
REG_HIDDEN_LAYER_CANDIDATES = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
]

# loguniform bounds (low, high) for alpha and learning_rate_init.
REG_ALPHA_BOUNDS = (1e-6, 3e-2)
REG_LR_INIT_BOUNDS = (1e-4, 3e-3)

# RandomizedSearchCV scoring and parallelism.
REG_SCORING = "neg_root_mean_squared_error"
REG_N_JOBS = -1  # -1 uses all cores

# Refit settings (applied when training final regressor after selecting best params).
REG_REFIT_MAX_ITER = 2000
REG_REFIT_EARLY_STOPPING = False
REG_TOL_DEFAULT = 1e-3

# ======================================================
# 12) Preprocessing defaults (prompt/map)
# ======================================================

# Numerical stability for L2 normalization (prevents division by zero).
L2_NORM_EPS = 1e-12

# Map preprocessing defaults:
# - MAP_CLIP_Q_DEFAULT: percentile clipping per feature (train-derived)
# - MAP_IMPUTE_STRATEGY_DEFAULT: SimpleImputer strategy for missing map features
# - MAP_ROBUST_QRANGE_DEFAULT: RobustScaler quantile range
# - MAP_VAR_EPS_DEFAULT: remove near-constant map features (std <= eps)
MAP_CLIP_Q_DEFAULT = (5, 95)
MAP_IMPUTE_STRATEGY_DEFAULT = "median"
MAP_ROBUST_QRANGE_DEFAULT = (5, 95)
MAP_VAR_EPS_DEFAULT = 1e-12

# ======================================================
# 13) Concat pipeline defaults (building X from map/prompt embeddings)
# ======================================================

# Logging verbosity for concat pipeline (passed to concat_embeddings.setup_logging).
CONCAT_VERBOSITY_DEFAULT = 1

# Filenames produced by embedding pipelines (used by concat & resolution).
PROMPT_EMBEDDINGS_NPZ_NAME = "prompts_embeddings.npz"
PROMPTS_PARQUET_NAME = "prompts.parquet"
MAP_EMBEDDINGS_NPZ_NAME = "maps_embeddings.npz"
MAPS_PARQUET_NAME = "maps.parquet"

# Preferred extent columns to carry forward from maps.parquet into training pairs.
# Must include MAPS_ID_COL, EXTENT_DIAG_COL, EXTENT_AREA_COL.
EXTENT_COLS_PREFERRED = [
    MAPS_ID_COL,
    EXTENT_DIAG_COL,
    EXTENT_AREA_COL,
    "extent_width_m",
    "extent_height_m",
    "extent_minx",
    "extent_miny",
    "extent_maxx",
    "extent_maxy",
]

# ======================================================
# 14) Map embedding pipeline defaults
# ======================================================

# Logging verbosity for map embedding pipeline.
MAP_EMBED_VERBOSITY_DEFAULT = 1

# Default normalization kind passed to map embedder (depends on your mapvec implementation).
MAP_EMBED_NORM_DEFAULT = "extent"

# ======================================================
# 15) Prompt embedding pipeline defaults
# ======================================================

# Logging verbosity for prompt embedding pipeline.
PROMPT_EMBED_VERBOSITY_DEFAULT = 1

# Whether to L2-normalize prompt embeddings (if supported by embedder).
PROMPT_EMBED_L2_NORMALIZE_DEFAULT = True

# Whether to also write CSV files alongside the binary artifacts.
PROMPT_EMBED_SAVE_CSV_DEFAULT = False

# ======================================================
# 16) Dataset splitting defaults / policies
# ======================================================

# Use operator×intensity stratification for single-prompt maps when possible.
# If some operator×intensity groups are too rare, code falls back to operator-only.
SPLIT_USE_INTENSITY_FOR_STRAT_DEFAULT = True

SPLIT_SEED_DEFAULT = 42
SPLIT_VAL_RATIO_DEFAULT = 0.15
SPLIT_TEST_RATIO_DEFAULT = 0.15
SPLIT_MAX_ATTEMPTS_DEFAULT = 500
SPLIT_VERBOSE_DEFAULT = True

# If fewer than this many single-prompt maps exist, fallback assigns all rows to TRAIN.
SPLIT_TINY_SINGLE_MAP_THRESHOLD = 2

# Multi-prompt map threshold: a map is "multi-prompt" if prompt_count > threshold.
# Default threshold=1 means ">=2 prompts" are multi-prompt and forced into TRAIN.
SPLIT_PROMPTS_PER_MAP_MULTI_THRESHOLD = 1

# Internal stratification label settings (column name + delimiter).
SPLIT_STRAT_COL_NAME = "_strat"
SPLIT_STRAT_DELIM = "__"

# Seed offset used for the second shuffle split stage (train/val inside trainval).
SPLIT_SSS2_SEED_OFFSET = 999

# JSON formatting indent when saving split files.
SPLIT_JSON_INDENT = 2

# ======================================================
# Feature modes (global experiment modes)
# ======================================================
# Type used throughout the repo for feature-mode values.
FeatureMode = Literal[
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
]

# Allowed feature modes (runtime list).
FEATURE_MODES: Final[tuple[str, ...]] = (
    "prompt_only",
    "prompt_plus_map",
    "use_map",
    "openai_map",
    "map_only",
)

# Maps feature mode -> (artifact folder, artifact filename stem).
# Used when resolving output artifacts (X_*.npy, train_pairs_*.parquet, etc.).
FEATURE_MODE_TO_ARTIFACTS: Final[dict[str, tuple[str, str]]] = {
    "prompt_only": ("train_out_prompt_only", "prompt_only"),
    "use_map": ("train_out_use", "use_map"),
    "map_only": ("train_out_map_only", "map_only"),
    "openai_map": ("train_out_openai", "openai_map"),
    "prompt_plus_map": ("train_out", "prompt_plus_map"),
}

# ======================================================
# Mapvec concat script defaults / repo path policy
# ======================================================

# How many levels up from this file to reach project root
# (concat_embeddings.py is under src/mapvec/concat/, so parents[3] reaches repo root).
PROJECT_ROOT_MARKER_LEVELS_UP = 3

# Default data folder name under repo root
DEFAULT_DATA_DIRNAME = "data"

# Default output folder name under data/
DEFAULT_OUTPUT_DIRNAME = "output"

# ======================================================
# Mapvec concat script output filenames
# ======================================================

# Main concatenated output
CONCAT_X_CONCAT_NAME = "X_concat.npy"
CONCAT_TRAIN_PAIRS_NAME = "train_pairs.parquet"
CONCAT_META_JSON_NAME = "meta.json"

# Optional debug outputs (when --save-blocks is enabled)
CONCAT_SAVE_BLOCKS_NAMES = {
    "X_map": "X_map.npy",
    "X_prompt": "X_prompt.npy",
    "map_ids": "map_ids.npy",
    "prompt_ids": "prompt_ids.npy",
}

# ======================================================
# Mapvec concat script behavior defaults
# ======================================================

CONCAT_VERBOSE_DEFAULT = 1
CONCAT_FAIL_ON_MISSING_DEFAULT = False
CONCAT_DROP_DUPES_DEFAULT = False
CONCAT_L2_PROMPT_DEFAULT = False
CONCAT_SAVE_BLOCKS_DEFAULT = False

# Numerical stability for L2-normalizing prompt embeddings inside concat script
CONCAT_L2_EPS = 1e-12

# If True, numeric-looking prompt_ids will be zero-padded to prompt_id_width in matching.
CONCAT_PAD_NUMERIC_PROMPT_IDS_DEFAULT = False

# Column name for raw parameter values from Excel / labels (before normalization).
PARAM_VALUE_COL = "param_value"

# ======================================================
# Mapvec: pooling per-polygon features into one map embedding
# ======================================================

# Which per-polygon columns should NOT be pooled into stats.
# Keep IDs / non-feature identifiers out of the aggregated embedding.
MAP_POOL_EXCLUDE_COLS_DEFAULT = ("id",)

# Which aggregate stats to compute per feature column.
# Options supported by current implementation: "mean", "std", "min", "max"
MAP_POOL_STATS_DEFAULT = ("mean", "std", "min", "max")

# Robust statistics: which quantiles to include per feature column.
# Set to () to disable quantiles entirely.
MAP_POOL_QUANTILES_DEFAULT = (0.25, 0.50, 0.75)

# Whether to append extra global scalars (polygon count, spreads, coverage).
MAP_POOL_ADD_GLOBALS_DEFAULT = True

# NumPy nanquantile method/interpolation setting.
# "linear" is stable across versions (fallback uses interpolation=...).
MAP_POOL_QUANTILE_METHOD = "linear"

# Input column names expected in df_polys for the optional global features.
POLY_CENTROID_X_COL = "centroid_x"
POLY_CENTROID_Y_COL = "centroid_y"
POLY_AREA_COL = "area"

# Names of the appended global features (must match the order used in code).
MAP_POOL_GLOBAL_FEATURE_NAMES = (
    "poly_count",
    "poly_spread_w",
    "poly_spread_h",
    "coverage_ratio",
)

# Common clamps / bounded-feature policy
UNIT_INTERVAL_MIN = 0.0
UNIT_INTERVAL_MAX = 1.0

# ======================================================
# Mapvec: polygon feature extraction (hand-crafted)
# ======================================================

# Output schema
POLY_FEATURE_ID_COL = "id"

# Stable feature order for output columns (id is added separately).
# Changing this changes the column order in the output dataframe.
POLY_FEATURE_ORDER = (
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

# Features that are normalized by the map diagonal during stabilization.
POLY_BY_DIAG_FEATURES = (
    "perimeter",
    "eq_diameter",
    "nn_dist_median",
    "knn1",
    "knn3",
    "bbox_width",
    "bbox_height",
)

# Normalization mode for polygon features.
# - "extent": normalize within true map extent
# - "fixed": normalize within a fixed W×H frame (pads if smaller)
POLY_NORM_MODE_DEFAULT = "extent"

# Default fixed width/height (only used when norm_mode="fixed" and fixed_wh is None).
POLY_NORM_FIXED_WH_DEFAULT = (400.0, 400.0)

# Numeric stability epsilon used for widths/heights/bbox safety (avoid division by zero).
EPS_POSITIVE = 1e-12

# Eccentricity computation constants:
# - POLY_ECC_EPS: tolerance for treating r≈1 as circular
# - POLY_ECC_MAX: clamp max eccentricity to avoid extreme values
POLY_ECC_EPS = 1e-12
POLY_ECC_MAX = 0.999999

# Density radii defined as fractions of the map diagonal.
# Changing these affects density_* features.
POLY_DENSITY_R05_FRAC = 0.05
POLY_DENSITY_R10_FRAC = 0.10

# Optional per-map clipping of extreme feature values (upper quantile).
# Set to 1.0 to disable clipping.
POLY_CLIP_QHI = 0.995

# Warning suppression regex modules (used to silence GEOS runtime warnings).
SHAPELY_WARN_MODULE_PREDICATES = r"shapely\.predicates"
SHAPELY_WARN_MODULE_SETOPS = r"shapely\.set_operations"

# ======================================================
# Mapvec: map embeddings pipeline (GeoJSON -> pooled embedding)
# ======================================================

# Default input root relative to data/ for the CLI (map tiles folder).
MAP_EMBED_ROOT_DEFAULT = "samples/pairs"

# Default glob pattern for the input geojson within each <map_id>/ folder.
MAP_EMBED_PATTERN_DEFAULT = "*_input.geojson"

# Default output directory relative to data/ for the CLI.
MAP_EMBED_OUTDIR_DEFAULT = "map_out"

# CLI defaults
MAP_EMBED_VERBOSE_DEFAULT = 1
MAP_EMBED_SAVE_CSV_DEFAULT = False

# If True and CRS is geographic (lat/lon), project to a metric CRS before computing extents & polygon measures.
MAP_EMBED_PROJECT_IF_GEOGRAPHIC = True
MAP_EMBED_GEOGRAPHIC_CRS_EPSG = 4326
MAP_EMBED_METRIC_CRS_EPSG = 3857

# Numeric eps used to avoid divide-by-zero / degenerate widths/heights.
# (If you already defined EPS_POSITIVE from polygon_features, reuse it globally.)
EPS_POSITIVE = 1e-12

# Drop polygons with very tiny area (slivers).
MAP_POLY_AREA_EPS = 1e-12

# Warning messages to silence (keeps logs readable on invalid geometries).
MAP_EMBED_WARNINGS_TO_IGNORE = (
    "invalid value encountered in within",
    "invalid value encountered in contains",
    "invalid value encountered in buffer",
)

# Output sidecar filenames produced by map embedding pipeline.
MAPS_CSV_NAME = "maps.csv"
MAPS_META_JSON_NAME = "meta.json"
MAPS_FEATURE_NAMES_JSON_NAME = "feature_names.json"

# Extent column names (used in maps.parquet)
EXTENT_MINX_COL = "extent_minx"
EXTENT_MINY_COL = "extent_miny"
EXTENT_MAXX_COL = "extent_maxx"
EXTENT_MAXY_COL = "extent_maxy"
EXTENT_WIDTH_COL = "extent_width_m"
EXTENT_HEIGHT_COL = "extent_height_m"

# ======================================================
# Mapvec: prompt embedding pipeline (Excel -> embeddings)
# ======================================================

# Environment variable to override where the data folder lives.
MAPVEC_DATA_DIR_ENVVAR = "MAPVEC_DATA_DIR"

# .env file name looked up at project root
DOTENV_FILENAME = ".env"

# Logging date format for prompt embedding script
PROMPT_EMBED_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Default Excel sheet name for prompts
PROMPTS_EXCEL_SHEET_DEFAULT = "Responses"

# Default column names in Excel user study file
EXCEL_COMPLETE_COL_DEFAULT = "complete"
EXCEL_REMOVE_COL_DEFAULT = "remove"
EXCEL_TEXT_COL_DEFAULT = "cleaned_text"

# Default filtering behavior for Excel rows
EXCEL_ONLY_COMPLETE_DEFAULT = True
EXCEL_EXCLUDE_REMOVED_DEFAULT = True

# Output sidecar filenames for prompt embedding pipeline
PROMPTS_META_JSON_NAME = "meta.json"
PROMPTS_EMBEDDINGS_CSV_NAME = "embeddings.csv"

# Available prompt encoder backends
PROMPT_ENCODER_CHOICES = ("dan", "transformer", "openai-small", "openai-large")

# USE models: Kaggle model IDs (used by kagglehub downloader)
USE_KAGGLE_MODEL_IDS = {
    "dan": "google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2",
    "transformer": "google/universal-sentence-encoder-large/tensorFlow2/universal-sentence-encoder-large/2",
}

# Where USE models live under data_dir (relative paths)
USE_MODEL_DIR_DAN = "input/model_dan"
USE_MODEL_DIR_TRANSFORMER = "input/model_transformer"

# Default batch sizes for embedding backends
USE_BATCH_SIZE_DEFAULT = 512
OPENAI_BATCH_SIZE_DEFAULT = 256

# OpenAI embedding model names
OPENAI_MODEL_NAME_SMALL = "text-embedding-3-small"
OPENAI_MODEL_NAME_LARGE = "text-embedding-3-large"

# Optional prefix added before each prompt when embedding with OpenAI.
# This can improve task alignment for your thesis domain, but it's a knob.
OPENAI_PROMPT_PREFIX = "Cartographic map generalization instruction: "
