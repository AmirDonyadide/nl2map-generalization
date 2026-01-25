# src/param_estimation/hybrid_param.py
"""
Hybrid parameter estimation from natural-language prompts.

Goal
----
Given:
  - prompt_text (string)
  - operator (one of: aggregate, displace, simplify, select, ...)
  - map_refs: per-map reference geometry scales (preferred), OR fallback tile scales
      expected keys (any subset):
        * extent_diag_m (meters)         # for distance-based ops
        * extent_area_m2 (square meters) # for area-based ops
        * tile_diag_m / tile_area_m2     # fallback if extents unavailable
  - map_stats: optional, per-map stats/quantiles for more realistic thresholds
      example keys (any subset, you define them upstream):
        * building_area_q: dict like {"q10":..., "q25":..., "q50":..., "q75":..., "q90":...} (m²)
        * road_length_q:   ... (m)
        * feature_<name>_q: ...
  - fallback_model: optional callable for ML fallback when text is ambiguous

Return:
  - param_norm: normalized value (distance / diag OR area / area_ref)
  - param_value: in original units (m or m²)
  - debug: rich dict explaining decisions

Important
---------
This module is rule-based + extensible. It does NOT require CRS columns.
You only need consistent "meters" geometry in your data pipeline (or treat your
stored scales as the source of truth).

Usage (high-level)
------------------
from src.param_estimation.hybrid_param import estimate_param_hybrid

param_norm, param_value, debug = estimate_param_hybrid(
    prompt_text=txt,
    operator=op,
    map_refs={"extent_diag_m": 565.0, "extent_area_m2": 160000.0},
    map_stats={"building_area_q": {"q25": 40, "q50": 80, "q75": 120}},
    fallback_model=None,
)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, Literal
from pathlib import Path
import numpy as np
import geopandas as gpd

FallbackFn = Callable[
    [str, str, Dict[str, Any], Dict[str, Any]],
    Tuple[Optional[float], Optional[float], Dict[str, Any]]
]

@dataclass(frozen=True)
class MapRefs:
    extent_diag_m: float
    extent_area_m2: float


def build_map_refs_from_geojson(geojson_path: str | Path) -> MapRefs:
    p = Path(geojson_path)

    gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # project to meters if geographic
    try:
        if gdf.crs and getattr(gdf.crs, "is_geographic", False):
            gdf = gdf.to_crs(3857)
    except Exception:
        pass

    minx, miny, maxx, maxy = map(float, gdf.total_bounds)
    w = maxx - minx
    h = maxy - miny

    if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
        # fallback minimal safe values
        return MapRefs(extent_diag_m=float("nan"), extent_area_m2=float("nan"))

    diag = float(np.sqrt(w * w + h * h))
    area = float(w * h)
    return MapRefs(extent_diag_m=diag, extent_area_m2=area)


ParamMode = Literal["EXPLICIT_NUMBER", "PERCENT", "QUALITATIVE", "UNSPECIFIED"]

# ----------------------------
# Operator groups (defaults)
# ----------------------------
DEFAULT_DISTANCE_OPS = {"aggregate", "displace", "simplify"}
DEFAULT_AREA_OPS = {"select"}

# ----------------------------
# Regex helpers
# ----------------------------
_NUM = r"(?P<num>(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?)"  # handles 1,000.5 / 1000,5 / 1000
_UNIT = r"(?P<unit>m2|m²|sqm|sq\.?m|square\s*meters?|square\s*metres?|meters?\b|metres?\b|m\b|%)"

# Examples matched:
#  - "100 m2", "100m²", "100 sqm"
#  - "10 m", "10m"
#  - "20%", "20 %", "20 percent"
_NUM_UNIT_RE = re.compile(rf"{_NUM}\s*(?:{_UNIT})", flags=re.IGNORECASE)

# percent variants
_PERCENT_RE = re.compile(rf"{_NUM}\s*(?:%|percent\b|percentage\b)", flags=re.IGNORECASE)

# number without unit (we treat it as operator-dependent fallback)
_BARE_NUM_RE = re.compile(_NUM, flags=re.IGNORECASE)

# qualitative cues
_SMALL_RE = re.compile(r"\b(tiny|very\s*small|small|smaller|minor|low)\b", re.IGNORECASE)
_MED_RE = re.compile(r"\b(medium|moderate|normal|average)\b", re.IGNORECASE)
_LARGE_RE = re.compile(r"\b(large|bigger|big|major|high|very\s*large)\b", re.IGNORECASE)

# explicit-all cues (no number): "all buildings", "everything"
_ALL_RE = re.compile(r"\b(all|everything|entire|whole)\b", re.IGNORECASE)

# unit normalization map
_UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "m2": "m2",
    "m²": "m2",
    "sqm": "m2",
    "sq.m": "m2",
    "sq m": "m2",
    "square meter": "m2",
    "square meters": "m2",
    "square metre": "m2",
    "square metres": "m2",
    "%": "%",
    "percent": "%",
    "percentage": "%",
}


def _to_float(num_str: str) -> float:
    """
    Convert '1,000.5' / '1.000,5' / '1000,5' safely to float.
    Heuristic:
      - if both '.' and ',' appear: assume thousand separators + decimal, pick last as decimal
      - if only ',' appears: treat ',' as decimal
      - else: normal float
    """
    s = num_str.strip()
    if "," in s and "." in s:
        # decimal is the last separator
        if s.rfind(",") > s.rfind("."):
            # comma decimal: "1.000,5" -> "1000.5"
            s = s.replace(".", "").replace(",", ".")
        else:
            # dot decimal: "1,000.5" -> "1000.5"
            s = s.replace(",", "")
    elif "," in s:
        # comma decimal: "1000,5" -> "1000.5"
        s = s.replace(",", ".")
    return float(s)


def _norm_unit(raw: str) -> str:
    r = raw.strip().lower().replace("²", "2")
    r = re.sub(r"\s+", " ", r)
    return _UNIT_ALIASES.get(r, r)


# ----------------------------
# Public API pieces
# ----------------------------
def detect_param_mode(prompt_text: str) -> ParamMode:
    t = (prompt_text or "").strip()
    if not t:
        return "UNSPECIFIED"

    # explicit number with unit OR bare % phrasing
    if _NUM_UNIT_RE.search(t) is not None:
        return "EXPLICIT_NUMBER"
    if _PERCENT_RE.search(t) is not None:
        return "PERCENT"

    # qualitative cues
    if _SMALL_RE.search(t) or _MED_RE.search(t) or _LARGE_RE.search(t):
        return "QUALITATIVE"

    # no usable signal
    return "UNSPECIFIED"


def extract_number_and_unit(prompt_text: str) -> Optional[Tuple[float, Optional[str]]]:
    """
    Return (value, unit) if we can find a number (and maybe a unit).
    unit is normalized to: 'm', 'm2', '%' or None.
    """
    t = (prompt_text or "")
    m = _NUM_UNIT_RE.search(t)
    if m:
        val = _to_float(m.group("num"))
        unit = _norm_unit(m.group("unit"))
        return val, unit

    # no unit - try bare number
    m2 = _BARE_NUM_RE.search(t)
    if m2:
        val = _to_float(m2.group("num"))
        return val, None

    return None


def extract_percent(prompt_text: str) -> Optional[float]:
    """
    Returns percent as 0..100 if present (e.g., "20%" -> 20.0), else None.
    """
    t = (prompt_text or "")
    m = _PERCENT_RE.search(t)
    if not m:
        return None
    pct = _to_float(m.group("num"))
    return float(pct)


def qualitative_to_quantile(prompt_text: str) -> Optional[float]:
    """
    Map qualitative words to a quantile.
      - small  -> 0.25
      - medium -> 0.50
      - large  -> 0.75
    If multiple cues appear, picks the strongest match by priority: small/large/medium.
    """
    t = (prompt_text or "")
    if _SMALL_RE.search(t):
        return 0.25
    if _LARGE_RE.search(t):
        return 0.75
    if _MED_RE.search(t):
        return 0.50
    return None


def map_quantile_to_threshold(
    map_stats: Dict[str, Any],
    q: float,
    operator: str,
    *,
    fallback_area_m2: Optional[float] = None,
    fallback_diag_m: Optional[float] = None,
) -> Optional[float]:
    """
    Convert qualitative quantile -> param_value using map_stats.

    Preferred behavior (you provide map_stats):
      - For select (area): use map_stats['building_area_q'] quantiles
      - For distance ops: use map_stats['distance_q'] or a suitable feature quantile

    Minimal fallback behavior if stats not provided:
      - For select: return q * fallback_area_m2 * 0.001  (tiny fraction of extent area)
      - For distance: return q * fallback_diag_m * 0.05  (few % of diag)

    Returns param_value in original units (m for distance ops, m² for select).
    """
    op = (operator or "").strip().lower()

    # --- user-provided stats route ---
    if op in DEFAULT_AREA_OPS:
        bq = map_stats.get("building_area_q") if isinstance(map_stats, dict) else None
        if isinstance(bq, dict):
            # choose nearest known quantile key
            key = _nearest_quantile_key(bq, q)
            if key is not None and bq.get(key) is not None:
                try:
                    return float(bq[key])
                except Exception:
                    pass

    if op in DEFAULT_DISTANCE_OPS:
        dq = map_stats.get("distance_q") if isinstance(map_stats, dict) else None
        if isinstance(dq, dict):
            key = _nearest_quantile_key(dq, q)
            if key is not None and dq.get(key) is not None:
                try:
                    return float(dq[key])
                except Exception:
                    pass

    # --- minimal fallback route (only if no stats) ---
    if op in DEFAULT_AREA_OPS and fallback_area_m2 and fallback_area_m2 > 0:
        # "small buildings" -> choose a *small fraction* of map area as an area threshold
        # tuned for your stated ranges (select ~17..167 m²), 400x400=160k m² => fraction ~0.0001..0.001
        frac = 0.0002 if q <= 0.25 else (0.0005 if q <= 0.5 else 0.0010)
        return float(fallback_area_m2 * frac)

    if op in DEFAULT_DISTANCE_OPS and fallback_diag_m and fallback_diag_m > 0:
        # distance thresholds are small relative to diag; use a few percent
        frac = 0.01 if q <= 0.25 else (0.02 if q <= 0.5 else 0.03)
        return float(fallback_diag_m * frac)

    return None


def estimate_param_hybrid(
    prompt_text: str,
    operator: str,
    map_refs: Dict[str, Any],
    map_stats: Optional[Dict[str, Any]] = None,
    fallback_model: Optional[FallbackFn] = None,
    *,
    distance_ops: Optional[set[str]] = None,
    area_ops: Optional[set[str]] = None,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Main entry point.

    Returns:
      (param_norm, param_value, debug)

    Normalization:
      - if operator in distance_ops: param_norm = param_value / diag_ref
      - if operator in area_ops:     param_norm = param_value / area_ref

    map_refs accepted keys (any):
      - extent_diag_m, extent_area_m2  (preferred)
      - tile_diag_m, tile_area_m2      (fallback)
      - diag_m, area_m2                (accepted aliases)

    If extraction fails and fallback_model is provided:
      fallback_model(prompt_text, operator, map_refs, map_stats) -> -> (param_norm, param_value, debug_dict)
    """
    op = (operator or "").strip().lower()
    dist_ops = distance_ops or set(DEFAULT_DISTANCE_OPS)
    ar_ops = area_ops or set(DEFAULT_AREA_OPS)

    map_stats = map_stats or {}

    diag_ref = _pick_first_float(map_refs, ["extent_diag_m", "diag_m", "tile_diag_m"])
    area_ref = _pick_first_float(map_refs, ["extent_area_m2", "area_m2", "tile_area_m2"])

    debug: Dict[str, Any] = {
        "operator": op,
        "mode": None,
        "extracted": {},
        "refs": {"diag_ref": diag_ref, "area_ref": area_ref},
        "decision": {},
        "warnings": [],
    }

    mode = detect_param_mode(prompt_text)
    debug["mode"] = mode

    # ----------------------------
    # Case 1: explicit number
    # ----------------------------
    if mode == "EXPLICIT_NUMBER":
        got = extract_number_and_unit(prompt_text)
        if got is None:
            debug["warnings"].append("EXPLICIT_NUMBER mode but could not parse number.")
        else:
            val, unit = got
            debug["extracted"] = {"value": val, "unit": unit}

            # If unit is missing, infer from operator group
            if unit is None:
                unit = "m2" if op in ar_ops else "m"
                debug["decision"]["unit_inferred_from_operator"] = unit

            # If unit conflicts with operator, try to resolve:
            if op in ar_ops and unit == "m":
                # select expects area; if user wrote "m" maybe they meant "m2" but we won't guess strongly.
                debug["warnings"].append("Operator is area-based but unit looks like meters.")
            if op in dist_ops and unit == "m2":
                debug["warnings"].append("Operator is distance-based but unit looks like square meters.")

            # interpret value directly as param_value
            param_value = float(val)

            # normalize
            param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
            return param_norm, param_value, debug

    # ----------------------------
    # Case 2: percent
    # ----------------------------
    if mode == "PERCENT":
        pct = extract_percent(prompt_text)
        debug["extracted"] = {"percent": pct}
        if pct is None:
            debug["warnings"].append("PERCENT mode but could not parse percent.")
        else:
            pct01 = float(pct) / 100.0
            # Mapping percent -> threshold depends on what percent refers to.
            # We implement conservative defaults:
            # - select: percent of *typical* building sizes if available; else fraction of extent area
            # - distance: percent of diag_ref
            if op in ar_ops:
                # If building area distribution exists, we map percent to a quantile threshold
                bq = map_stats.get("building_area_q")
                if isinstance(bq, dict) and len(bq) > 0:
                    # interpret "remove 20%" as "remove smallest 20%" => threshold at q=0.20
                    q = max(0.0, min(1.0, pct01))
                    debug["decision"]["percent_interpretation"] = "select: smallest-percent -> area-quantile"
                    thr = map_quantile_to_threshold(map_stats, q, op, fallback_area_m2=area_ref, fallback_diag_m=diag_ref)
                    if thr is not None:
                        param_value = float(thr)
                        param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                        return param_norm, param_value, debug
                # fallback: fraction of extent area (very small)
                if area_ref and area_ref > 0:
                    debug["decision"]["percent_interpretation"] = "select: fraction-of-extent-area (fallback)"
                    # scale down aggressively so 20% does not become huge; tune constant
                    param_value = float(area_ref * (pct01 * 0.001))
                    param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                    return param_norm, param_value, debug

            if op in dist_ops and diag_ref and diag_ref > 0:
                debug["decision"]["percent_interpretation"] = "distance: percent-of-diagonal"
                param_value = float(diag_ref * pct01)
                param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                return param_norm, param_value, debug

            debug["warnings"].append("Could not map percent to param_value (missing refs/stats).")

    # ----------------------------
    # Case 3: qualitative
    # ----------------------------
    if mode == "QUALITATIVE":
        q = qualitative_to_quantile(prompt_text)
        debug["extracted"] = {"quantile": q}
        if q is None:
            debug["warnings"].append("QUALITATIVE mode but no cue matched.")
        else:
            thr = map_quantile_to_threshold(
                map_stats=map_stats,
                q=float(q),
                operator=op,
                fallback_area_m2=area_ref,
                fallback_diag_m=diag_ref,
            )
            debug["decision"]["qualitative_mapping"] = "map_quantile_to_threshold"
            if thr is not None:
                param_value = float(thr)
                param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                return param_norm, param_value, debug
            debug["warnings"].append("Qualitative cue found but could not map to threshold (no stats/refs).")

    # ----------------------------
    # Case 4: unspecified
    # ----------------------------
    if mode == "UNSPECIFIED":
        # If prompt indicates "all", we can interpret as maximum threshold
        if _ALL_RE.search(prompt_text or ""):
            debug["decision"]["unspecified_interpretation"] = "ALL -> max threshold using map refs"
            if op in ar_ops and area_ref and area_ref > 0:
                # if "all buildings", threshold can be very large; we cap to a reasonable fraction of extent
                param_value = float(area_ref * 0.01)  # 1% of extent area (still could be large; tune)
                param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                return param_norm, param_value, debug
            if op in dist_ops and diag_ref and diag_ref > 0:
                param_value = float(diag_ref * 0.10)  # 10% of diagonal (tune)
                param_norm = _normalize(param_value, op, dist_ops, ar_ops, diag_ref, area_ref, debug)
                return param_norm, param_value, debug

        debug["warnings"].append("No numeric/percent/qualitative signal in prompt.")

    # ----------------------------
    # Fallback to ML model if provided
    # ----------------------------
    if fallback_model is not None:
        try:
            pn, pv, fb_dbg = fallback_model(prompt_text, op, map_refs, map_stats)
            debug["decision"]["fallback_model_used"] = True
            debug["decision"]["fallback_debug"] = fb_dbg
            return pn, pv, debug
        except Exception as e:
            debug["warnings"].append(f"fallback_model failed: {type(e).__name__}: {e}")

    # total failure
    return None, None, debug


# ----------------------------
# Internal helpers
# ----------------------------
def _pick_first_float(d: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if v == v and v > 0:  # not NaN and positive
                    return v
            except Exception:
                continue
    return None


def _normalize(
    param_value: float,
    operator: str,
    dist_ops: set[str],
    area_ops: set[str],
    diag_ref: Optional[float],
    area_ref: Optional[float],
    debug: Dict[str, Any],
) -> Optional[float]:
    op = operator
    if op in dist_ops:
        if not diag_ref or diag_ref <= 0:
            debug["warnings"].append("Missing/invalid diag_ref for distance normalization.")
            return None
        return float(param_value) / float(diag_ref)
    if op in area_ops:
        if not area_ref or area_ref <= 0:
            debug["warnings"].append("Missing/invalid area_ref for area normalization.")
            return None
        return float(param_value) / float(area_ref)

    debug["warnings"].append(f"Unknown operator group for normalization: {op}")
    return None


def _nearest_quantile_key(qdict: Dict[str, Any], q: float) -> Optional[str]:
    """
    Accepts keys like 'q10','q25','q50','q75','q90' and finds the closest to q in [0..1].
    """
    if not qdict:
        return None
    q = float(q)
    best_key = None
    best_dist = 1e9
    for k in qdict.keys():
        m = re.match(r"q(\d+)", str(k).strip().lower())
        if not m:
            continue
        pct = float(m.group(1))
        qq = pct / 100.0
        dist = abs(qq - q)
        if dist < best_dist:
            best_dist = dist
            best_key = str(k)
    return best_key
