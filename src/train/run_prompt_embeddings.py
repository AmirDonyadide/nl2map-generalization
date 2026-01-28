# src/train/run_prompt_embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Your existing embedding module (already supports USE + OpenAI)
from src.mapvec.prompts import prompt_embeddings as pe


@dataclass(frozen=True)
class PromptEmbeddingRunMeta:
    n_prompts: int
    model_label: str
    out_dir: str
    embeddings_path: str
    prompts_parquet_path: str


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

    Loads prompts from the user study file, embeds them using cfg.PROMPT_ENCODER
    (USE or OpenAI), and writes outputs to out_dir in the same format used by
    the rest of your pipeline.

    Required fields expected in `paths`:
      - RESPONSES_SHEET, TILE_ID_COL, COMPLETE_COL, REMOVE_COL, TEXT_COL

    Required fields expected in `cfg`:
      - PROMPT_ENCODER, BATCH_SIZE
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe.setup_logging(verbosity=verbosity)

    ids, texts, tile_ids, id_colname = pe.load_prompts_from_source(
        input_path=input_path,
        sheet_name=paths.RESPONSES_SHEET,
        tile_id_col=paths.TILE_ID_COL,
        complete_col=paths.COMPLETE_COL,
        remove_col=paths.REMOVE_COL,
        text_col=paths.TEXT_COL,
    )

    if len(texts) == 0:
        raise ValueError("No prompts loaded (after filtering). Check Excel sheet/columns and filters.")

    embed_fn, model_label = pe.get_embedder(
        kind=cfg.PROMPT_ENCODER,
        data_dir=Path(paths.DATA_DIR),
        l2_normalize=l2_normalize,
        batch_size=int(cfg.BATCH_SIZE),
    )

    E = embed_fn(texts)
    E = np.asarray(E)

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

    # These filenames are defined by prompt_embeddings.save_outputs
    embeddings_path = out_dir / "prompts_embeddings.npz"
    prompts_parquet_path = out_dir / "prompts.parquet"

    return PromptEmbeddingRunMeta(
        n_prompts=int(len(texts)),
        model_label=str(model_label),
        out_dir=str(out_dir),
        embeddings_path=str(embeddings_path),
        prompts_parquet_path=str(prompts_parquet_path),
    )
