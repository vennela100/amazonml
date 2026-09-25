"""Population-aware evaluation and submission writing (standard library only).

This module makes ``scoring.py`` the single source of truth for the competition
metric. Stages 5 and 6 must not compute F0.5 on the candidate-pair table
directly: doing so hides two things the metric cares about most.

  * True matches lost during candidate generation never appear as pair rows, so
    a pair-level metric overstates recall. They must be counted as false
    negatives against the *complete* ground truth.
  * Singletons (entities with no true match) have no positive pair rows. Skipping
    them removes both the reward for a correct empty prediction (score 1.0) and
    the penalty for a false merge (score 0.0) -- the precision error F0.5 weights
    four times as heavily as a miss.

The evaluator therefore works on predicted *sets* per Source 1 entity, scored
against complete truth over an explicit *population* of every S1 entity in scope
(including singletons and entities that retrieved zero candidates).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import scoring


def predicted_sets_from_pairs(s1_ids, cand_ids, proba, threshold):
    """Build ``{s1_id: set(cand_id)}`` for pairs scoring at or above threshold.

    Only entities with at least one surviving pair appear. The population, not
    this mapping, defines which entities are scored; a missing key means an
    empty prediction (which is correct for a true singleton).
    """
    predicted = defaultdict(set)
    for s1_id, cand_id, score in zip(s1_ids, cand_ids, proba):
        if score >= threshold:
            predicted[s1_id].add(cand_id)
    return dict(predicted)


def score_population(population_ids, truth_map, predicted_map):
    """Exact macro F0.5 over an explicit population using ``scoring.MacroMetrics``.

    ``population_ids``   every S1 entity in scope, scored exactly once.
    ``truth_map``        complete ground truth ``{s1_id: set(match_id)}``; a
                         missing key or empty set marks a true singleton.
    ``predicted_map``    model output ``{s1_id: set(match_id)}``; a missing key
                         is an empty prediction.
    """
    seen = set()
    metrics = scoring.MacroMetrics()
    for s1_id in population_ids:
        if s1_id in seen:
            raise ValueError(f'Duplicate S1 id in population: {s1_id!r}')
        seen.add(s1_id)
        truth = truth_map.get(s1_id, set())
        prediction = predicted_map.get(s1_id, set())
        metrics.add(truth, prediction)
    return metrics.result()


def sweep_thresholds(population_ids, truth_map, s1_ids, cand_ids, proba,
                     thresholds):
    """Score every threshold over the full population; return sweep and best.

    ``best`` maximizes macro F0.5. Ties break toward the *higher* threshold,
    because the metric is precision-weighted and a higher threshold merges less.
    """
    sweep = []
    for threshold in thresholds:
        predicted_map = predicted_sets_from_pairs(
            s1_ids, cand_ids, proba, threshold)
        result = score_population(population_ids, truth_map, predicted_map)
        sweep.append({
            'threshold': round(float(threshold), 4),
            'macro_f05': result['macro_f05'],
            'false_singleton_rate': result['false_singleton_rate'],
            'missed_non_singleton_rate': result['missed_non_singleton_rate'],
            'nonempty_predictions': result['nonempty_predictions'],
        })
    best = max(sweep, key=lambda row: (row['macro_f05'], row['threshold']))
    return {'sweep': sweep, 'best': best}


def write_matching_results(population_ids, predicted_map, out_path):
    """Write ``matching_results.tsv``: one row per S1, comma-joined match ids.

    Every entity in ``population_ids`` appears exactly once; an empty prediction
    yields an empty second field (a correct singleton). IDs are sorted for
    determinism. Header and column names match ``scoring.MATCH_HEADER``.
    """
    with Path(out_path).open('w', encoding='utf-8', newline='') as handle:
        handle.write('\t'.join(scoring.MATCH_HEADER) + '\n')
        for s1_id in population_ids:
            ids = predicted_map.get(s1_id, set())
            handle.write(f'{s1_id}\t{",".join(sorted(ids))}\n')


def write_candidate_pairs(population_ids, candidate_map, out_path):
    """Write ``candidate_pairs.tsv`` in the same one-row-per-S1 schema.

    ``candidate_map`` is the exact candidate set scored by the final model.
    ``matching_results.tsv`` must be a per-entity subset of this file.
    """
    with Path(out_path).open('w', encoding='utf-8', newline='') as handle:
        handle.write('\t'.join(scoring.MATCH_HEADER) + '\n')
        for s1_id in population_ids:
            ids = candidate_map.get(s1_id, set())
            handle.write(f'{s1_id}\t{",".join(sorted(ids))}\n')


def check_subset(population_ids, predicted_map, candidate_map):
    """Every predicted match must be within that entity's candidate set."""
    for s1_id in population_ids:
        extra = predicted_map.get(s1_id, set()) - candidate_map.get(s1_id, set())
        if extra:
            raise ValueError(
                f'{s1_id}: predicted ids outside candidate set: {sorted(extra)}')
