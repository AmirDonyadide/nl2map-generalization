#src/train/run_prompt_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.mapvec.prompts import prompt_embeddings as pe

@dataclass(frozen=True)
class PromptEmbeddingRunMeta:
    n_prompts: int
    model_label: str
    out_dir: str
    embeddings_path: str
    prompts_parquet_path: str

def _require_attrs(obj: Any, attrs: list[str], *, where: str) -> None:
    missing = [a for a in attrs if getattr(obj, a, None) is None]
    if missing:
        raise AttributeError(f"{where} is missing required attributes: {missing}")
    
def run_prompt_embeddings_from_config(
    *,
    input_path: Path,
    out_dir: Path,
    cfg: Any,
    paths: Any,
    verbosity: int = 1,
    l2_normalize: bool = True,
    also_save_embeddings_csv: bool = False,
) -> PromptEmbeddingRunMeta:
    """
    End-to-end prompt embedding pipeline.

    Loads prompts from the user study file, embeds them using cfg.PROMPT_ENCODER,
    and writes outputs to out_dir.

    Required in `paths`:
      - RESPONSES_SHEET, TILE_ID_COL, COMPLETE_COL, REMOVE_COL, TEXT_COL, DATA_DIR
      - optional: PROMPT_ID_COL

    Required in `cfg`:
      - PROMPT_ENCODER, BATCH_SIZE
    """
    _require_attrs(
        paths,
        ["RESPONSES_SHEET", "TILE_ID_COL", "COMPLETE_COL", "REMOVE_COL", "TEXT_COL", "DATA_DIR"],
        where="paths",
    )
    _require_attrs(cfg, ["PROMPT_ENCODER", "BATCH_SIZE"], where="cfg")

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe.setup_logging(verbosity=verbosity)

    prompt_id_col = getattr(paths, "PROMPT_ID_COL", "prompt_id")

    ids, texts, tile_ids, id_colname = pe.load_prompts_from_source(
        input_path=input_path,
        sheet_name=paths.RESPONSES_SHEET,
        tile_id_col=paths.TILE_ID_COL,
        complete_col=paths.COMPLETE_COL,
        remove_col=paths.REMOVE_COL,
        text_col=paths.TEXT_COL,
        prompt_id_col=prompt_id_col,
    )

    if len(texts) == 0:
        raise ValueError("No prompts loaded (after filtering). Check Excel sheet/columns and filters.")

    embed_fn, model_label = pe.get_embedder(
        kind=cfg.PROMPT_ENCODER,
        data_dir=Path(paths.DATA_DIR),
        l2_normalize=l2_normalize,
        batch_size=int(cfg.BATCH_SIZE),
    )

    E = np.asarray(embed_fn(texts))
    if E.ndim != 2 or E.shape[0] != len(texts):
        raise RuntimeError(
            f"Unexpected embedding shape: {E.shape}. Expected (n_prompts, dim)=({len(texts)}, dim)."
        )

    pe.save_outputs(
        out_dir=out_dir,
        ids=ids,
        texts=texts,
        tile_ids=tile_ids,
        E=E,
        model_name=model_label,
        l2_normalized=l2_normalize,
        id_colname=id_colname,
        also_save_embeddings_csv=also_save_embeddings_csv,
    )

    embeddings_path = out_dir / "prompts_embeddings.npz"
    prompts_parquet_path = out_dir / "prompts.parquet"

    return PromptEmbeddingRunMeta(
        n_prompts=int(len(texts)),
        model_label=str(model_label),
        out_dir=str(out_dir),
        embeddings_path=str(embeddings_path),
        prompts_parquet_path=str(prompts_parquet_path),
    )
