# notebooks/experiments.py
from __future__ import annotations
from pathlib import Path

def make_experiments(repo_root: Path) -> dict:
    data_dir = repo_root / "data"

    return {
        "openai_prompt_only": {
            "train_out": data_dir / "output" / "train_out_openai_prompt_only",
            "model_out": data_dir / "output" / "models" / "exp_openai_prompt_only",
            "feature_mode": "prompt_only",
            "prompt_encoder_kind": "openai-small",
        },
        "use_prompt_only": {
            "train_out": data_dir / "output" / "train_out_use_prompt_only",
            "model_out": data_dir / "output" / "models" / "exp_use_prompt_only",
            "feature_mode": "prompt_only",
            "prompt_encoder_kind": "dan",
        },
        "map_only": {
            "train_out": data_dir / "output" / "train_out_map_only",
            "model_out": data_dir / "output" / "models" / "exp_map_only",
            "feature_mode": "map_only",
        },
        "use_map": {
            "train_out": data_dir / "output" / "train_out_use_map",
            "model_out": data_dir / "output" / "models" / "exp_use_map",
            "feature_mode": "prompt_plus_map",
            "prompt_encoder_kind": "dan",
        },
        "openai_map": {
            "train_out": data_dir / "output" / "train_out_openai_map",
            "model_out": data_dir / "output" / "models" / "exp_openai_map",
            "feature_mode": "prompt_plus_map",
            "prompt_encoder_kind": "openai-small",
        },
    }
