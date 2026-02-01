# src/imgofup/userstudy/prompt_cleaning.py
from __future__ import annotations

"""
IMGOFUP — UserStudy Prompt Cleaning Utilities

This module contains reusable building blocks for cleaning user-study free-text prompts
using a mixture of lightweight heuristics and LLM calls.

Design goals
------------
- Notebook-friendly: you can call these functions from notebooks with minimal glue code.
- Safe + robust: defensive parsing of LLM outputs, predictable return types.
- Optional dependencies: langdetect/tqdm are optional (graceful fallback).
- No I/O in core functions: reading/writing CSV/XLSX should stay in notebooks or pipelines.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import os
import time

import numpy as np
import pandas as pd

# Optional: tqdm progress bars
try:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Optional: language detection
try:  # pragma: no cover
    from langdetect import detect as _langdetect_detect  # type: ignore
    from langdetect import DetectorFactory as _DetectorFactory  # type: ignore

    _DetectorFactory.seed = 42
except Exception:  # pragma: no cover
    _langdetect_detect = None  # type: ignore


# ---------------------------
# OpenAI client helpers
# ---------------------------

def get_openai_client(*, api_key_env: str = "OPENAI_API_KEY"):
    """
    Lazily import and initialize an OpenAI client.

    Raises:
        RuntimeError if OPENAI_API_KEY is missing.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} not set. Create a .env file in repo root and set:\n"
            f"{api_key_env}=sk-...\n"
        )
    from openai import OpenAI  # local import keeps dependency optional until used
    return OpenAI(api_key=api_key)


@dataclass(frozen=True)
class LLMConfig:
    """
    Common OpenAI call settings.
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    retry_sleep_s: float = 0.8


def _sleep_backoff(attempt: int, base: float) -> None:
    time.sleep(base * (1.5 ** max(0, attempt - 1)))


def _chat_text(
    client: Any,
    *,
    system: str,
    user: str,
    cfg: LLMConfig,
) -> str:
    """
    Chat completion returning plain text (no JSON mode).
    Retries transient failures.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, int(cfg.max_retries) + 1):
        try:
            r = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return str(r.choices[0].message.content or "").strip()
        except Exception as e:  # pragma: no cover
            last_err = e
            _sleep_backoff(attempt, cfg.retry_sleep_s)
    raise RuntimeError(f"OpenAI request failed after {cfg.max_retries} tries: {last_err}")  # pragma: no cover


def _chat_json(
    client: Any,
    *,
    system: str,
    user: str,
    cfg: LLMConfig,
) -> Dict[str, Any]:
    """
    Chat completion in JSON-object mode (when supported by the model).
    Retries transient failures.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, int(cfg.max_retries) + 1):
        try:
            r = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = str(r.choices[0].message.content or "").strip()
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return {}
            return obj
        except Exception as e:  # pragma: no cover
            last_err = e
            _sleep_backoff(attempt, cfg.retry_sleep_s)
    raise RuntimeError(f"OpenAI request failed after {cfg.max_retries} tries: {last_err}")  # pragma: no cover


# ---------------------------
# Basic string hygiene
# ---------------------------

DEFAULT_EMPTY_TOKENS = {"", "nan", "none", "null", "nil", "n/a"}


def normalize_empty_to_nan(x: Any, *, empty_tokens: Optional[set[str]] = None) -> Any:
    """
    Convert common empty tokens to np.nan, otherwise return stripped string.
    """
    empty_tokens = empty_tokens or DEFAULT_EMPTY_TOKENS
    if x is None:
        return np.nan
    s = str(x).strip()
    if s.lower() in empty_tokens:
        return np.nan
    return s


# ---------------------------
# Language detection
# ---------------------------

def safe_detect_lang(text: Any) -> str:
    """
    Best-effort language detection.

    Returns:
        ISO code (e.g., 'en') or 'unknown' if detection not available/fails.
    """
    if text is None or not isinstance(text, str):
        return "unknown"
    t = text.strip()
    if not t:
        return "unknown"
    if _langdetect_detect is None:
        return "unknown"
    try:
        return str(_langdetect_detect(t))
    except Exception:
        return "unknown"


# ---------------------------
# Prompt-likeness detection
# ---------------------------

PROMPT_DETECT_SYSTEM = "You are a strict classifier for user-study responses."

PROMPT_DETECT_USER_TEMPLATE = """Decide if the text is a usable user prompt for map generalization.

Return ONLY one token:
- PROMPT
- NOT_PROMPT

Text:
{TEXT}
"""


def looks_like_prompt_llm(
    text: Any,
    *,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> bool:
    """
    Returns True if the LLM classifies the text as a usable prompt.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return False
    user = PROMPT_DETECT_USER_TEMPLATE.format(TEXT=text.strip())
    out = _chat_text(client, system=PROMPT_DETECT_SYSTEM, user=user, cfg=cfg)
    return out.strip() == "PROMPT"


# ---------------------------
# Translation
# ---------------------------

TRANSLATE_SYSTEM = "You translate text to English with high fidelity."

TRANSLATE_USER_TEMPLATE = """Translate the following text into English.

Rules:
- Preserve meaning and tone.
- Keep it as a prompt/command.
- Do NOT add new information.
- Output a single line only.

Text:
{TEXT}
"""


def translate_to_english(
    text: Any,
    *,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> Any:
    """
    Translate to English. Returns np.nan for empty inputs.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return np.nan
    user = TRANSLATE_USER_TEMPLATE.format(TEXT=text.strip())
    out = _chat_text(client, system=TRANSLATE_SYSTEM, user=user, cfg=cfg)
    return out.strip()


# ---------------------------
# Minimal grammar fix
# ---------------------------

GRAMMAR_SYSTEM = "You perform minimal grammatical correction without changing tone or meaning."

GRAMMAR_USER_TEMPLATE = """Correct grammar, punctuation, spacing, and number formatting in the sentence.

Rules:
- Do NOT change meaning.
- Do NOT change tone.
- Do NOT paraphrase.
- Do NOT add/remove content unless grammatically required.
- Keep it as ONE sentence prompt.
- Output ONE line only.

Text:
{TEXT}
"""


def minimal_grammar_fix(
    text: Any,
    *,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> Any:
    """
    Minimal grammar correction. Returns np.nan for empty inputs.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return np.nan
    user = GRAMMAR_USER_TEMPLATE.format(TEXT=text.strip())
    out = _chat_text(client, system=GRAMMAR_SYSTEM, user=user, cfg=cfg)
    return out.strip()


# ---------------------------
# Threshold detection
# ---------------------------

TH_EXIST_SYSTEM = "You are a strict binary classifier for map-generalization prompts."

TH_EXIST_USER_TEMPLATE = """Determine whether the prompt contains a threshold/constraint condition (explicit or implicit).

Return ONLY a JSON object with exactly these keys:
{{
  "threshold_exist": true/false,
  "evidence": "short phrase indicating the threshold, or empty string if none"
}}

Prompt:
{TEXT}
"""


def detect_threshold_llm(
    text: Any,
    *,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> Dict[str, Any]:
    """
    Returns:
        {"threshold_exist": bool, "evidence": str}
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return {"threshold_exist": False, "evidence": ""}
    t = str(text).strip()
    if not t:
        return {"threshold_exist": False, "evidence": ""}

    obj = _chat_json(
        client,
        system=TH_EXIST_SYSTEM,
        user=TH_EXIST_USER_TEMPLATE.format(TEXT=t),
        cfg=cfg,
    )

    threshold_exist = bool(obj.get("threshold_exist", False))
    evidence = obj.get("evidence", "")
    evidence = "" if evidence is None else str(evidence).strip()

    if not threshold_exist:
        evidence = ""

    return {"threshold_exist": threshold_exist, "evidence": evidence}


TH_KNOWN_SYSTEM = "You are a strict classifier for threshold specification in map-generalization prompts."

TH_KNOWN_USER_TEMPLATE = """The prompt below contains a threshold/constraint.

Decide whether the threshold is KNOWN or UNKNOWN.

KNOWN:
- numeric threshold (e.g., 10 meters, 200 m², 5 buildings)
- qualitative but explicit threshold (e.g., very small buildings, closely spaced, short distance)

UNKNOWN:
- placeholder/vague threshold (e.g., X meters, N, some distance, a threshold, less than the threshold)

Return ONLY a JSON object with exactly these keys:
{{
  "threshold_known": true/false,
  "evidence": "If threshold_known is true: copy the short phrase that specifies the threshold. Otherwise: empty string."
}}

Prompt:
{TEXT}
"""


def threshold_known_llm(
    text: Any,
    *,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> Dict[str, Any]:
    """
    Returns:
        {"threshold_known": bool, "evidence": str}
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return {"threshold_known": False, "evidence": ""}
    t = str(text).strip()
    if not t:
        return {"threshold_known": False, "evidence": ""}

    obj = _chat_json(
        client,
        system=TH_KNOWN_SYSTEM,
        user=TH_KNOWN_USER_TEMPLATE.format(TEXT=t),
        cfg=cfg,
    )

    known = bool(obj.get("threshold_known", False))
    evidence = obj.get("evidence", "")
    evidence = "" if evidence is None else str(evidence).strip()
    if not known:
        evidence = ""

    return {"threshold_known": known, "evidence": evidence}


# ---------------------------
# Conflict detection + suggestion
# ---------------------------

CONFLICT_SYSTEM = "You are a strict validator for a map-generalization user-study dataset."

CONFLICT_USER_TEMPLATE = """You are given a prompt and its metadata labels.
Decide if there is a conflict between the prompt text and the labels.

Rules:
- Mark conflict=true if the prompt does NOT match the operator label.
- Mark conflict=true if intensity label is clearly inconsistent with wording.
- Mark conflict=true if threshold_exist is true but the prompt has no threshold/constraint.
- Mark conflict=true if threshold_known is true but the prompt does not specify any threshold.
- Mark conflict=false if the prompt is compatible with the labels, even if it is vague.

Output ONLY JSON with keys:
{{
  "conflict": true/false,
  "reason": "one short sentence"
}}

Metadata:
operator: {operator}
intensity: {intensity}
param_value: {param_value}
threshold_exist: {threshold_exist}
threshold_known: {threshold_known}

Prompt:
{prompt}
"""


def _boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t", "on")


def conflict_llm(
    *,
    prompt: Any,
    operator: Any,
    intensity: Any,
    param_value: Any,
    threshold_exist: Any,
    threshold_known: Any,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> Dict[str, Any]:
    """
    Returns:
        {"conflict": bool, "reason": str}
    """
    p = "" if prompt is None else str(prompt).strip()
    if not p:
        return {"conflict": True, "reason": "prompt is empty"}

    user = CONFLICT_USER_TEMPLATE.format(
        operator=str(operator or "").strip(),
        intensity=str(intensity or "").strip(),
        param_value=str(param_value or "").strip(),
        threshold_exist=str(_boolish(threshold_exist)).lower(),
        threshold_known=str(_boolish(threshold_known)).lower(),
        prompt=p,
    )

    obj = _chat_json(client, system=CONFLICT_SYSTEM, user=user, cfg=cfg)
    conflict = bool(obj.get("conflict", True))
    reason = obj.get("reason", "")
    reason = "" if reason is None else str(reason).strip()

    return {"conflict": conflict, "reason": reason}


SUGGEST_SYSTEM = "You are an expert assistant for cleaning map-generalization user-study prompts."


def suggest_prompt_llm(
    *,
    cleaned_text: Any,
    raw_text: Any,
    conflict_reason: Any,
    operator: Any,
    intensity: Any,
    param_value: Any,
    threshold_exist: Any,
    threshold_known: Any,
    client: Any,
    cfg: LLMConfig = LLMConfig(),
) -> str:
    """
    Suggest a corrected prompt that resolves a detected conflict.
    Returns empty string if cleaned_text is empty.
    """
    cleaned = "" if cleaned_text is None else str(cleaned_text).strip()
    if not cleaned:
        return ""

    raw = "" if raw_text is None else str(raw_text).strip()
    reason = "" if conflict_reason is None else str(conflict_reason).strip()

    th_exist = _boolish(threshold_exist)
    th_known = _boolish(threshold_known)

    user = f"""
You are given a map-generalization prompt and metadata.

Your task:
Rewrite the prompt so that it is CONSISTENT with the metadata,
while staying as close as possible to the original wording.

Rules:
- Do NOT change the intent unless required to fix the conflict
- Do NOT add new operations
- Do NOT add thresholds if threshold_exist is false
- If threshold_known is true, explicitly specify the threshold
- Keep it a single imperative sentence
- Output ONLY the revised prompt text (no explanations)

Metadata:
operator: {str(operator or "").strip()}
intensity: {str(intensity or "").strip()}
param_value: {str(param_value or "").strip()}
threshold_exist: {str(th_exist).lower()}
threshold_known: {str(th_known).lower()}

Conflict reason:
{reason}

Original cleaned prompt:
{cleaned}

Original raw response (for context only):
{raw}
""".strip()

    out = _chat_text(client, system=SUGGEST_SYSTEM, user=user, cfg=cfg)
    return out.strip()


# ---------------------------
# DataFrame-level convenience helpers
# ---------------------------
def apply_series(
    s: pd.Series,
    fn: Callable[[Any], Any],
    *,
    use_tqdm: bool = True,
    desc: str = "Applying",
) -> pd.Series:
    """
    Apply fn to a pandas Series and ALWAYS return a pandas Series with the same index.

    This implementation:
    - does NOT require tqdm.pandas()
    - uses tqdm.auto.tqdm if available
    - preserves the original index exactly
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    if not use_tqdm:
        return s.apply(fn)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None  # type: ignore

    if tqdm is None:
        return s.apply(fn)

    vals = []
    for x in tqdm(s.tolist(), total=len(s), desc=desc):
        vals.append(fn(x))

    return pd.Series(vals, index=s.index)



def clean_prompts_dataframe(
    df: pd.DataFrame,
    *,
    text_col: str,
    client: Any,
    llm_cfg: LLMConfig = LLMConfig(),
    use_tqdm: bool = True,
    do_prompt_detect: bool = True,
    do_translate: bool = True,
    do_grammar_fix: bool = True,
) -> pd.DataFrame:
    """
    High-level convenience function:
      - normalize empties -> NaN
      - (optional) detect prompt-like
      - detect language + translate to English when needed
      - (optional) minimal grammar fix

    Returns a COPY of df with added columns:
      - step1_is_prompt, step1_text
      - step1_lang
      - step2_text
      - cleaned_text
    """
    if text_col not in df.columns:
        raise KeyError(f"df missing text_col '{text_col}'")

    out = df.copy()

    out[text_col] = out[text_col].apply(normalize_empty_to_nan)

    if do_prompt_detect:
        out["step1_is_prompt"] = apply_series(
            out[text_col],
            lambda x: looks_like_prompt_llm(x, client=client, cfg=llm_cfg),
            use_tqdm=use_tqdm,
        )
        out["step1_text"] = out[text_col].where(out["step1_is_prompt"], np.nan)
    else:
        out["step1_is_prompt"] = out[text_col].notna()
        out["step1_text"] = out[text_col]

    out["step1_lang"] = out["step1_text"].apply(lambda x: safe_detect_lang(x) if isinstance(x, str) else "unknown")

    if do_translate:
        need_translate = out["step1_is_prompt"] & (out["step1_lang"] != "en") & out["step1_text"].notna()
        out["step2_text"] = np.nan
        out.loc[out["step1_is_prompt"] & (out["step1_lang"] == "en"), "step2_text"] = out.loc[
            out["step1_is_prompt"] & (out["step1_lang"] == "en"), "step1_text"
        ]
        if need_translate.any():
            out.loc[need_translate, "step2_text"] = apply_series(
                out.loc[need_translate, "step1_text"],
                lambda x: translate_to_english(x, client=client, cfg=llm_cfg),
                use_tqdm=use_tqdm,
            )
    else:
        out["step2_text"] = out["step1_text"]

    if do_grammar_fix:
        out["cleaned_text"] = apply_series(
            out["step2_text"],
            lambda x: minimal_grammar_fix(x, client=client, cfg=llm_cfg),
            use_tqdm=use_tqdm,
        )
    else:
        out["cleaned_text"] = out["step2_text"]

    return out


def add_threshold_columns(
    df: pd.DataFrame,
    *,
    cleaned_text_col: str = "cleaned_text",
    client: Any,
    llm_cfg: LLMConfig = LLMConfig(),
    use_tqdm: bool = True,
) -> pd.DataFrame:
    """
    Add columns:
      - threshold_exist, threshold_evidence
      - threshold_known, threshold_known_evidence

    Uses LLM calls only on non-empty cleaned_text.
    """
    if cleaned_text_col not in df.columns:
        raise KeyError(f"df missing '{cleaned_text_col}'")

    out = df.copy()
    mask = out[cleaned_text_col].notna()

    # threshold_exist
    res_exist = pd.Series([{"threshold_exist": False, "evidence": ""}] * len(out), index=out.index)
    if mask.any():
        res_exist.loc[mask] = apply_series(
            out.loc[mask, cleaned_text_col],
            lambda x: detect_threshold_llm(x, client=client, cfg=llm_cfg),
            use_tqdm=use_tqdm,
        )

    out["threshold_exist"] = res_exist.apply(lambda x: bool(x.get("threshold_exist", False)))
    out["threshold_evidence"] = res_exist.apply(lambda x: str(x.get("evidence", "") or "").strip())

    # threshold_known only when threshold_exist True
    mask2 = out["threshold_exist"] & out[cleaned_text_col].notna()
    out["threshold_known"] = False
    out["threshold_known_evidence"] = ""

    if mask2.any():
        res_known = apply_series(
            out.loc[mask2, cleaned_text_col],
            lambda x: threshold_known_llm(x, client=client, cfg=llm_cfg),
            use_tqdm=use_tqdm,
        )
        out.loc[mask2, "threshold_known"] = res_known.apply(lambda x: bool(x.get("threshold_known", False))).values
        out.loc[mask2, "threshold_known_evidence"] = res_known.apply(lambda x: str(x.get("evidence", "") or "").strip()).values

    # enforce: if known is false => evidence empty
    out.loc[~out["threshold_known"], "threshold_known_evidence"] = ""

    return out
