# src/types.py 
from __future__ import annotations
from typing import Literal, Final

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

FEATURE_MODE_TO_ARTIFACTS: Final[dict[str, tuple[str, str]]] = {
    "prompt_only": ("train_out_prompt_only", "prompt_only"),
    "use_map": ("train_out_use", "use_map"),
    "map_only": ("train_out_map_only", "map_only"),
    "openai_map": ("train_out_openai", "openai_map"),
    "prompt_plus_map": ("train_out", "prompt_plus_map"),
}
