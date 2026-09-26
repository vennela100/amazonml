"""Tests for population-aware evaluation and submission writing.

These lock in the exact behaviors the pair-level metric in the Stage 5/6 drafts
got wrong: singleton scoring and matches lost during retrieval.
"""
from __future__ import annotations

import evaluation
import scoring


def test_correct_singleton_scores_one():
    # True singleton, empty prediction -> 1.0. It must be scored, not skipped.
    result = evaluation.score_population(
        population_ids=['S1-1'], truth_map={}, predicted_map={})
    assert result['macro_f05'] == 1.0
    assert result['true_singletons'] == 1
    assert result['false_singleton_count'] == 0


def test_false_merge_on_singleton_scores_zero():
    # A false merge onto a true singleton is the worst precision error; 0.0.
    result = evaluation.score_population(
        population_ids=['S1-1'], truth_map={},
        predicted_map={'S1-1': {'S2-9'}})
    assert result['macro_f05'] == 0.0
    assert result['false_singleton_count'] == 1


def test_lost_true_match_counts_as_false_negative():
    # Truth has two matches; only one survived retrieval. Recall must suffer.
    # F0.5 for tp=1, fp=0, fn=1 is 5/(5+0+1) = 0.8333...
    result = evaluation.score_population(
        population_ids=['S1-1'],
        truth_map={'S1-1': {'S2-1', 'S3-2'}},
        predicted_map={'S1-1': {'S2-1'}})
    assert abs(result['macro_f05'] - 5 / 6) < 1e-12


def test_population_average_mixes_singletons_and_matches():
    # One perfect singleton (1.0) and one exact non-singleton (1.0) -> 1.0;
    # a zero-candidate non-singleton drags the average down.
    result = evaluation.score_population(
        population_ids=['S1-1', 'S1-2', 'S1-3'],
        truth_map={'S1-2': {'S2-1'}, 'S1-3': {'S2-2'}},
        predicted_map={'S1-2': {'S2-1'}})  # S1-3 retrieved nothing
    # S1-1 singleton empty = 1.0; S1-2 exact = 1.0; S1-3 missed all = 0.0.
    assert abs(result['macro_f05'] - 2 / 3) < 1e-12


def test_threshold_sweep_prefers_precision_on_ties():
    # Two thresholds yield identical F0.5; the higher one must win.
    population = ['S1-1']
    truth = {'S1-1': {'S2-1'}}
    s1 = ['S1-1', 'S1-1']
    cand = ['S2-1', 'S2-9']
    proba = [0.9, 0.1]  # S2-9 is a false candidate with a low score
    out = evaluation.sweep_thresholds(
        population, truth, s1, cand, proba, thresholds=[0.5, 0.8])
    # Both thresholds keep only S2-1 -> identical F0.5; higher threshold wins.
    assert out['best']['threshold'] == 0.8


def test_submission_round_trips_through_scorer(tmp_path):
    population = ['S1-1', 'S1-2', 'S1-3']
    predicted = {'S1-1': {'S2-1', 'S3-2'}, 'S1-2': set()}  # S1-3 absent = empty
    candidates = {'S1-1': {'S2-1', 'S3-2', 'S2-5'}, 'S1-3': {'S2-8'}}
    match_path = tmp_path / 'matching_results.tsv'
    cand_path = tmp_path / 'candidate_pairs.tsv'
    evaluation.write_matching_results(population, predicted, match_path)
    evaluation.write_candidate_pairs(population, candidates, cand_path)

    rows = list(scoring.read_tsv(match_path, scoring.MATCH_HEADER))
    assert [r[0] for r in rows] == population  # every S1 exactly once, in order
    parsed = {r[0]: scoring.parse_ids(r[1]) for r in rows}
    assert parsed['S1-1'] == {'S2-1', 'S3-2'}
    assert parsed['S1-2'] == set()
    assert parsed['S1-3'] == set()
    # candidate_pairs.tsv uses its own header (candidate_entity_ids).
    crows = list(scoring.read_tsv(cand_path, evaluation.CANDIDATE_HEADER))
    assert [r[0] for r in crows] == population


def test_check_subset_rejects_prediction_outside_candidates():
    import pytest
    with pytest.raises(ValueError):
        evaluation.check_subset(
            ['S1-1'], predicted_map={'S1-1': {'S2-1'}},
            candidate_map={'S1-1': {'S2-2'}})
