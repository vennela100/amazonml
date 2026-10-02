"""Memory-safe streaming prediction for the full test set.

Stage 4 materializes the whole (n_pairs x 60) feature matrix, which is tens of GB
at full-test scale and will OOM. This streams instead: it reads candidate pairs
grouped by Source 1 id, computes features for that entity's candidates, scores
them with the model, thresholds, and writes the two output rows — holding only
one entity's pairs at a time.

Population = every test Source 1 id in the candidates DB `queries` table, so the
output covers all test entities (empty where retrieval found nothing).

  python predict_streaming.py --cand-db reports/stage3/test/candidates.sqlite \
      --model reports/stage5/model.cbm --decision reports/stage5/decision.json \
      --data-dir artifacts/stage2 --split test --output-dir output
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent))
import evaluation
import scoring
from features import compute_features, parse_row
from stage4 import NORM_HEADER, COL_ENTITY_ID, load_records, build_rare_vocab, needed_ids


def load_population(cand_db):
    with closing(sqlite3.connect(str(cand_db))) as db:
        return [r[0] for r in db.execute('SELECT id FROM queries ORDER BY id')]


def run(args):
    started = time.monotonic()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cand_db = Path(args.cand_db)
    data = Path(args.data_dir) / args.split
    threshold = json.loads(Path(args.decision).read_text())['threshold']

    model = CatBoostClassifier(); model.load_model(args.model)

    # Load only the records referenced by candidates (S1 with candidates + pool).
    pool_keep = needed_ids(cand_db)
    with closing(sqlite3.connect(str(cand_db))) as db:
        s1_keep = {r[0] for r in db.execute('SELECT DISTINCT s1_id FROM candidates')}
    print(f'[stream] loading {len(s1_keep):,} S1 + {len(pool_keep):,} pool records', flush=True)
    s1_records = load_records(data / f'{args.split}_source1.tsv', keep=s1_keep)
    pool = load_records(data / f'{args.split}_source2.tsv', keep=pool_keep)
    pool.update(load_records(data / f'{args.split}_source3.tsv', keep=pool_keep))
    rare_vocab = build_rare_vocab(pool)

    population = load_population(cand_db)
    match_path = out / 'matching_results.tsv'
    cand_path = out / 'candidate_pairs.tsv'
    mf = match_path.open('w', encoding='utf-8', newline='')
    cf = cand_path.open('w', encoding='utf-8', newline='')
    mf.write('\t'.join(scoring.MATCH_HEADER) + '\n')
    cf.write('\t'.join(evaluation.CANDIDATE_HEADER) + '\n')

    pop_iter = iter(population)
    current = next(pop_iter, None)
    n_pred = n_cand = done = 0

    def flush_until(target, matched, cands):
        """Write rows for population ids up to (and including) target."""
        nonlocal current, done
        while current is not None and (target is None or current <= target):
            if current == target:
                mf.write(f'{current}\t{",".join(sorted(matched))}\n')
                cf.write(f'{current}\t{",".join(sorted(cands))}\n')
            else:
                mf.write(f'{current}\t\n')   # queried but no candidates -> empty
                cf.write(f'{current}\t\n')
            done += 1
            current = next(pop_iter, None)

    with closing(sqlite3.connect(str(cand_db))) as db:
        cur = db.execute('SELECT s1_id,candidate_id,source,routes,best_score,metadata '
                         'FROM candidates ORDER BY s1_id')
        cur_s1, feats, cids = None, [], []

        def score_group():
            nonlocal n_pred, n_cand
            if not cids:
                return set(), set()
            proba = model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
            matched = {c for c, p in zip(cids, proba) if p >= threshold}
            n_cand += len(cids); n_pred += len(matched)
            return matched, set(cids)

        for s1_id, cid, source, routes, best, meta in cur:
            if s1_id != cur_s1:
                if cur_s1 is not None:
                    matched, cset = score_group()
                    flush_until(cur_s1, matched, cset)
                cur_s1, feats, cids = s1_id, [], []
            s1r, cr = s1_records.get(s1_id), pool.get(cid)
            if s1r is None or cr is None:
                continue
            feats.append(compute_features(s1r, cr, routes or '', best, source, meta, rare_vocab))
            cids.append(cid)
        if cur_s1 is not None:
            matched, cset = score_group()
            flush_until(cur_s1, matched, cset)
    flush_until(None, set(), set())   # remaining empties
    mf.close(); cf.close()

    print(f'[stream] DONE: {done:,} entities, {n_cand:,} candidates, {n_pred:,} predicted '
          f'in {time.monotonic()-started:.0f}s', flush=True)
    print(f'  {match_path}\n  {cand_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cand-db', required=True)
    p.add_argument('--model', default='reports/stage5/model.cbm')
    p.add_argument('--decision', default='reports/stage5/decision.json')
    p.add_argument('--data-dir', default='artifacts/stage2')
    p.add_argument('--split', default='test')
    p.add_argument('--output-dir', default='output')
    run(p.parse_args())


if __name__ == '__main__':
    main()
