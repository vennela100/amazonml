"""Stage 6: Generate predictions and the two required submission files.

Reads:
  - reports/stage5/model.cbm       (trained CatBoost)
  - reports/stage5/decision.json   (decision threshold + scale)
  - a Stage 4 features.npz          (feature matrix, pair IDs, population, truth)

Writes (into the output dir):
  - matching_results.tsv   one row per S1: source1_entity_id, matched_entity_ids
  - candidate_pairs.tsv    one row per S1: the exact candidate set the model scored
  - metrics.json / report.md

Output format (from the problem statement): both files are one row per Source 1
entity, tab-separated, header `source1_entity_id<TAB>matched_entity_ids`, the id
list comma-separated, empty when the entity has no matches. Every S1 in the
queried population appears exactly once, and matches are a per-entity subset of
candidates.

Scale note: the threshold is applied to the RAW model probability, the same
scale Stage 5 selected it on. There is no calibrator between selection and
application (an isotonic calibrator is monotonic, so it cannot change the
decision anyway, and mixing scales was a latent bug).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import evaluation
import scoring

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise SystemExit('catboost not installed: pip install catboost')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_npz(npz_path):
    """Load a Stage 4 .npz (locally generated, trusted) into a plain dict."""
    data = np.load(npz_path, allow_pickle=True)
    if 'population' not in data:
        raise ValueError(f'{npz_path}: missing population; rebuild with Stage 4.')
    truth_map = {}
    for row in data['population_truth']:
        s1_id, ids = str(row).split('\t')
        truth_map[s1_id] = set(ids.split(',')) if ids else set()
    return {
        'X': data['X'].astype(np.float32),
        's1_ids': [str(x) for x in data['s1_ids']],
        'cand_ids': [str(x) for x in data['cand_ids']],
        'population': [str(x) for x in data['population']],
        'truth_map': truth_map,
    }


def predict_proba(model, X, batch_size=500_000):
    """Raw model probability of the positive class, scored in batches."""
    out = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        out.append(model.predict_proba(X[start:end])[:, 1])
        log.info(f'Scored {end:,}/{len(X):,} pairs')
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def build_candidate_map(population, s1_ids, cand_ids):
    """Every S1 in the population -> set of all its candidates (empty allowed)."""
    cand_map = {s: set() for s in population}
    for s, c in zip(s1_ids, cand_ids):
        cand_map.setdefault(s, set()).add(c)
    return cand_map


def run(args):
    started = time.monotonic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold
    if threshold is None:
        threshold = json.loads(Path(args.decision).read_text())['threshold']
    log.info(f'Decision threshold: {threshold} (raw model probability)')

    model = CatBoostClassifier()
    model.load_model(args.model)

    data = load_npz(args.features_npz)
    population, s1_ids, cand_ids = data['population'], data['s1_ids'], data['cand_ids']
    log.info(f'{len(s1_ids):,} pairs over {len(population):,} S1 entities')

    proba = predict_proba(model, data['X'])
    predicted_map = evaluation.predicted_sets_from_pairs(
        s1_ids, cand_ids, proba, threshold)
    candidate_map = build_candidate_map(population, s1_ids, cand_ids)

    # Safety invariant required by the rules: matches subset of candidates.
    evaluation.check_subset(population, predicted_map, candidate_map)

    match_path = out_dir / 'matching_results.tsv'
    cand_path = out_dir / 'candidate_pairs.tsv'
    evaluation.write_matching_results(population, predicted_map, match_path)
    evaluation.write_candidate_pairs(population, candidate_map, cand_path)

    # Re-read both through the strict parser to guarantee they are well formed.
    _validate_written(match_path, population)
    _validate_written(cand_path, population)

    # Score only when ground truth is present (training/validation, not test).
    metrics = {'threshold': threshold,
               'n_entities': len(population),
               'n_predicted_pairs': sum(len(v) for v in predicted_map.values()),
               'n_candidate_pairs': sum(len(v) for v in candidate_map.values())}
    has_truth = any(data['truth_map'].get(s) for s in population)
    if has_truth:
        result = evaluation.score_population(
            population, data['truth_map'], predicted_map)
        metrics['scored'] = result
        log.info(f'Macro F0.5: {result["macro_f05"]:.4f} '
                 f'(false-singleton rate {result["false_singleton_rate"]}, '
                 f'missed non-singleton rate {result["missed_non_singleton_rate"]})')
    else:
        log.info('No ground truth in population (test set) — outputs only.')

    metrics['elapsed_seconds'] = time.monotonic() - started
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2),
                                          encoding='utf-8')
    _render_report(metrics, match_path, cand_path, out_dir / 'report.md')

    log.info('\nSTAGE 6 PASS')
    print('\nSTAGE 6 PASS')
    print(f'  Entities: {metrics["n_entities"]:,}  '
          f'predicted pairs: {metrics["n_predicted_pairs"]:,}')
    if has_truth:
        print(f'  Macro F0.5: {metrics["scored"]["macro_f05"]:.4f} @ {threshold}')
    print(f'  {match_path}')
    print(f'  {cand_path}')


def _validate_written(path, population):
    """Strict re-read: header, one row per population S1, valid id lists."""
    rows = list(scoring.read_tsv(path, scoring.MATCH_HEADER))
    ids = [r[0] for r in rows]
    if ids != list(population):
        raise ValueError(f'{path}: rows must equal the population exactly, once each')
    for r in rows:
        scoring.validate_s1(r[0])
        scoring.parse_ids(r[1])


def _render_report(metrics, match_path, cand_path, path):
    lines = [
        '# Stage 6: Predictions & Submission', '', '**Status**: PASS', '',
        '## Outputs', '',
        f'- `{match_path.name}` — one row per S1 (matching_results)',
        f'- `{cand_path.name}` — one row per S1 (candidate_pairs)', '',
        '## Summary', '',
        f'| Metric | Value |', '|---|---|',
        f'| Threshold (raw proba) | {metrics["threshold"]} |',
        f'| S1 entities | {metrics["n_entities"]:,} |',
        f'| Predicted match pairs | {metrics["n_predicted_pairs"]:,} |',
        f'| Candidate pairs | {metrics["n_candidate_pairs"]:,} |',
    ]
    if 'scored' in metrics:
        s = metrics['scored']
        lines += [
            f'| Macro F0.5 | **{s["macro_f05"]:.4f}** |',
            f'| False-singleton rate | {s["false_singleton_rate"]} |',
            f'| Missed non-singleton rate | {s["missed_non_singleton_rate"]} |',
        ]
    lines += ['', '## Next Steps', '',
              '- Run `utils/validate_submission.py --check-ids` before upload.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-npz', default='reports/stage4/features.npz')
    parser.add_argument('--model', default='reports/stage5/model.cbm')
    parser.add_argument('--decision', default='reports/stage5/decision.json')
    parser.add_argument('--output-dir', default='reports/stage6')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override decision threshold (default: decision.json)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
