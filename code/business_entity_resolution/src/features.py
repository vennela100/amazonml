"""Pairwise feature engineering for entity resolution.

Computes all features for a (S1_record, candidate_record) pair.
Uses normalized views from Stage 2 and retrieval metadata from Stage 3.
No country-specific string features — must generalise to unseen countries.
No external data or learned weights here; purely deterministic transforms.
"""
from __future__ import annotations

import re
import json
from typing import Any

from rapidfuzz import fuzz, distance as rfdist

VERSION = '1.0.0'

# ── Column indices in the 25-column normalized TSV ──────────────────────────
_H = [
    'entity_id',            # 0
    'business_name',        # 1  raw
    'business_address',     # 2  raw
    'country',              # 3
    'name_clean',           # 4
    'name_normalized',      # 5
    'address_normalized',   # 6
    'address_numbers',      # 7  JSON list
    'house_numbers',        # 8  JSON list
    'house_number_candidates', # 9  JSON list
    'house_number_source',  # 10
    'street_numbers',       # 11 JSON list
    'postal_codes',         # 12 JSON list
    'postal_candidates',    # 13 JSON list
    'postal_code_source',   # 14
    'name_missing',         # 15 int flag
    'address_missing',      # 16 int flag
    'name_null_like',       # 17 int flag
    'address_null_like',    # 18 int flag
    'name_normalized_empty',# 19 int flag
    'address_normalized_empty', # 20 int flag
    'house_number_missing', # 21 int flag
    'house_number_ambiguous',# 22 int flag
    'postal_code_missing',  # 23 int flag
    'postal_code_ambiguous',# 24 int flag
]
COL = {name: i for i, name in enumerate(_H)}

FEATURE_NAMES: list[str] = []  # populated at module load by _register()


def _register(*names):
    FEATURE_NAMES.extend(names)
    return names


# ── Feature groups (names registered in order) ──────────────────────────────
_NAME_FEATURES = _register(
    'name_edit_ratio',          # rapidfuzz WRatio (handles transpositions)
    'name_partial_ratio',       # rapidfuzz partial_ratio (substring matching)
    'name_token_set_ratio',     # rapidfuzz token_set_ratio (word-order invariant)
    'name_jaccard_tokens',      # Jaccard on whitespace tokens
    'name_jaccard_chars3',      # Jaccard on char 3-grams
    'name_len_s1',              # log(1+len(name_normalized)) for S1
    'name_len_cand',            # log(1+len(name_normalized)) for candidate
    'name_len_ratio',           # min/max of the two lengths
    'name_both_empty',          # both normalized names are empty
    'name_s1_empty',
    'name_cand_empty',
)

_ADDR_FEATURES = _register(
    'addr_edit_ratio',
    'addr_partial_ratio',
    'addr_token_set_ratio',
    'addr_jaccard_tokens',
    'addr_jaccard_chars3',
    'addr_len_s1',
    'addr_len_cand',
    'addr_len_ratio',
    'addr_both_empty',
    'addr_s1_empty',
    'addr_cand_empty',
)

_NUMERIC_FEATURES = _register(
    'house_both_present',       # both have at least one house number
    'house_agree',              # at least one house number in common
    'house_conflict',           # both present, no overlap
    'house_s1_missing',
    'house_cand_missing',
    'house_s1_ambiguous',
    'house_cand_ambiguous',
    'postal_both_present',
    'postal_agree',
    'postal_conflict',
    'postal_s1_missing',
    'postal_cand_missing',
    'postal_s1_ambiguous',
    'postal_cand_ambiguous',
)

_RARE_TOKEN_FEATURES = _register(
    'rare_token_overlap_count',         # count of rare tokens shared
    'rare_token_overlap_frac_s1',       # rare shared / total rare in S1 name
    'rare_token_overlap_frac_cand',     # rare shared / total rare in cand name
    'has_rare_token_overlap',           # binary
)

# Stage 3 retrieval routes, matching the labels stored in candidates.metadata.
# Country-agnostic: the same routes run for every country including unseen ones.
ROUTE_LABELS = ('name', 'address', 'combined', 'rare', 'components')

_RETRIEVAL_FEATURES = _register(
    # Per-route cosine score (0.0 if this route did not retrieve the pair).
    'score_name', 'score_address', 'score_combined', 'score_rare', 'score_components',
    # Per-route shortlist position as 1/(1+rank) so higher = better, 0 if absent.
    'invrank_name', 'invrank_address', 'invrank_combined', 'invrank_rare', 'invrank_components',
    'n_routes_found',           # number of routes that retrieved this pair
    'best_retrieval_score',     # max cosine score across all routes
    'source_s2',                # candidate comes from S2
    'source_s3',                # candidate comes from S3
)

_INTERACTION_FEATURES = _register(
    'name_high_addr_low',       # name_edit_ratio > 0.85 and addr_edit_ratio < 0.5
    'name_low_addr_high',       # addr strong, name weak
    'both_high',                # both > 0.8
    'house_agree_name_high',    # house numbers agree AND name edit > 0.8
    'postal_agree_name_high',
    'house_conflict_name_high', # conflict but name looks like a match
)

# Total feature count
N_FEATURES = len(FEATURE_NAMES)


# ── Helper functions ─────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return set(text.split()) if text else set()


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _parse_list(field: str) -> list[str]:
    """Parse a JSON list column (e.g. '["12","14"]') to a Python list."""
    if not field:
        return []
    try:
        v = json.loads(field)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _safe_log1p_len(text: str) -> float:
    import math
    return math.log1p(len(text))


# ── Record parser ─────────────────────────────────────────────────────────────

def parse_row(row: list[str]) -> dict[str, Any]:
    """Parse one row of the 25-column normalized TSV into a typed dict."""
    r: dict[str, Any] = {}
    r['entity_id'] = row[COL['entity_id']]
    r['business_name'] = row[COL['business_name']]
    r['business_address'] = row[COL['business_address']]
    r['country'] = row[COL['country']]
    r['name_clean'] = row[COL['name_clean']]
    r['name_norm'] = row[COL['name_normalized']]
    r['addr_norm'] = row[COL['address_normalized']]
    r['house_numbers'] = set(_parse_list(row[COL['house_numbers']]))
    r['house_candidates'] = set(_parse_list(row[COL['house_number_candidates']]))
    r['postal_codes'] = set(_parse_list(row[COL['postal_codes']]))
    r['postal_candidates'] = set(_parse_list(row[COL['postal_candidates']]))
    # Integer flags
    for col in ('name_missing', 'address_missing', 'name_null_like', 'address_null_like',
                 'name_normalized_empty', 'address_normalized_empty',
                 'house_number_missing', 'house_number_ambiguous',
                 'postal_code_missing', 'postal_code_ambiguous'):
        r[col] = int(row[COL[col]]) if row[COL[col]] else 0
    return r


# ── Core feature computation ─────────────────────────────────────────────────

def compute_features(
    s1: dict[str, Any],
    cand: dict[str, Any],
    routes: str,         # comma-separated route labels, e.g. "name,combined"
    best_score: float,
    source: int | None = None,           # 2 or 3; falls back to entity_id prefix
    metadata: str | None = None,         # JSON: {route: {"rank": int, "score": float}}
    rare_vocab: set[str] | None = None,  # optional: token set considered "rare"
) -> list[float]:
    """Compute the full feature vector for one (S1, candidate) pair.

    Args:
        s1: parsed S1 record dict from parse_row()
        cand: parsed candidate record dict from parse_row()
        routes: comma-separated route labels that found this pair
        best_score: highest retrieval cosine score for this pair
        source: 2 or 3 (S2/S3); if None, inferred from the candidate id prefix
        metadata: JSON string of per-route {"rank", "score"} from Stage 3
        rare_vocab: optional set of rare tokens (built from pool once); if None,
                    rare-token features default to 0.

    Returns:
        list of floats, length == N_FEATURES, in FEATURE_NAMES order.
    """
    feat: list[float] = []

    n1, n2 = s1['name_norm'], cand['name_norm']
    a1, a2 = s1['addr_norm'], cand['addr_norm']

    # ── Name similarity ──────────────────────────────────────────────────────
    if n1 and n2:
        feat += [
            fuzz.WRatio(n1, n2) / 100.0,
            fuzz.partial_ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            _jaccard(_tokens(n1), _tokens(n2)),
            _jaccard(_char_ngrams(n1, 3), _char_ngrams(n2, 3)),
        ]
    else:
        feat += [0.0, 0.0, 0.0, 0.0, 0.0]

    feat += [
        _safe_log1p_len(n1),
        _safe_log1p_len(n2),
        min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
        float(not n1 and not n2),
        float(not n1),
        float(not n2),
    ]

    # ── Address similarity ───────────────────────────────────────────────────
    if a1 and a2:
        feat += [
            fuzz.WRatio(a1, a2) / 100.0,
            fuzz.partial_ratio(a1, a2) / 100.0,
            fuzz.token_set_ratio(a1, a2) / 100.0,
            _jaccard(_tokens(a1), _tokens(a2)),
            _jaccard(_char_ngrams(a1, 3), _char_ngrams(a2, 3)),
        ]
    else:
        feat += [0.0, 0.0, 0.0, 0.0, 0.0]

    feat += [
        _safe_log1p_len(a1),
        _safe_log1p_len(a2),
        min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
        float(not a1 and not a2),
        float(not a1),
        float(not a2),
    ]

    # ── Numeric evidence ─────────────────────────────────────────────────────
    h1, h2 = s1['house_numbers'], cand['house_numbers']
    p1, p2 = s1['postal_codes'], cand['postal_codes']

    h_both = bool(h1) and bool(h2)
    h_agree = bool(h1 & h2) if h_both else False
    h_conflict = h_both and not h_agree

    feat += [
        float(h_both),
        float(h_agree),
        float(h_conflict),
        float(s1['house_number_missing']),
        float(cand['house_number_missing']),
        float(s1['house_number_ambiguous']),
        float(cand['house_number_ambiguous']),
    ]

    p_both = bool(p1) and bool(p2)
    p_agree = bool(p1 & p2) if p_both else False
    p_conflict = p_both and not p_agree

    feat += [
        float(p_both),
        float(p_agree),
        float(p_conflict),
        float(s1['postal_code_missing']),
        float(cand['postal_code_missing']),
        float(s1['postal_code_ambiguous']),
        float(cand['postal_code_ambiguous']),
    ]

    # ── Rare token overlap ───────────────────────────────────────────────────
    if rare_vocab is not None:
        t1 = _tokens(n1) & rare_vocab
        t2 = _tokens(n2) & rare_vocab
        overlap = t1 & t2
        count = len(overlap)
        frac_s1 = len(overlap) / len(t1) if t1 else 0.0
        frac_cand = len(overlap) / len(t2) if t2 else 0.0
        feat += [float(count), frac_s1, frac_cand, float(count > 0)]
    else:
        feat += [0.0, 0.0, 0.0, 0.0]

    # ── Retrieval metadata ───────────────────────────────────────────────────
    route_set = set(routes.split(',')) if routes else set()
    md = {}
    if metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                md = parsed
        except (json.JSONDecodeError, ValueError):
            md = {}
    # Per-route score, then per-route inverse rank, in ROUTE_LABELS order.
    feat += [float(md.get(r, {}).get('score', 0.0) or 0.0) for r in ROUTE_LABELS]
    feat += [
        1.0 / (1.0 + md[r]['rank']) if r in md and md[r].get('rank') is not None else 0.0
        for r in ROUTE_LABELS
    ]
    if source == 2:
        is_s2, is_s3 = 1.0, 0.0
    elif source == 3:
        is_s2, is_s3 = 0.0, 1.0
    else:
        is_s2 = float(cand['entity_id'].startswith('S2-'))
        is_s3 = float(cand['entity_id'].startswith('S3-'))
    feat += [
        float(len(route_set)),
        float(best_score),
        is_s2,
        is_s3,
    ]

    # ── Interaction features ─────────────────────────────────────────────────
    name_sim = feat[FEATURE_NAMES.index('name_edit_ratio')]
    addr_sim = feat[FEATURE_NAMES.index('addr_edit_ratio')]

    feat += [
        float(name_sim > 0.85 and addr_sim < 0.50),
        float(name_sim < 0.50 and addr_sim > 0.85),
        float(name_sim > 0.80 and addr_sim > 0.80),
        float(h_agree and name_sim > 0.80),
        float(p_agree and name_sim > 0.80),
        float(h_conflict and name_sim > 0.80),
    ]

    assert len(feat) == N_FEATURES, f'Feature count mismatch: {len(feat)} != {N_FEATURES}'
    return feat
