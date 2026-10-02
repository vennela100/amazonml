"""Memory-light fallback: predict from retrieval score already in candidates.sqlite.

If the model-based streaming predictor is too slow under memory pressure, this
guarantees a valid submission fast. It never loads name/address records — it only
scans the candidates table (best_score per pair) and marks a match when the
retrieval cosine is at or above a precision-leaning threshold. Lower quality than
the model, but complete, valid, and finishes in minutes.

Population = every test S1 in the queries table (un-retrieved -> empty).

  python fallback_predict.py --cand-db reports/stage3/test/candidates.sqlite \
      --threshold 0.55 --output-dir output
"""
from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import evaluation
import scoring


def run(args):
    cand_db = Path(args.cand_db)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    thr = args.threshold

    with closing(sqlite3.connect(str(cand_db))) as db:
        population = [r[0] for r in db.execute('SELECT id FROM queries ORDER BY id')]
        mf = (out / 'matching_results.tsv').open('w', encoding='utf-8', newline='')
        cf = (out / 'candidate_pairs.tsv').open('w', encoding='utf-8', newline='')
        mf.write('\t'.join(scoring.MATCH_HEADER) + '\n')
        cf.write('\t'.join(evaluation.CANDIDATE_HEADER) + '\n')

        cur = db.execute('SELECT s1_id,candidate_id,best_score FROM candidates ORDER BY s1_id')
        pop_iter = iter(population)
        current = next(pop_iter, None)
        cur_s1, matched, cands = None, set(), set()
        n_pred = 0

        def flush_to(target):
            nonlocal current
            while current is not None and (target is None or current <= target):
                if current == target:
                    mf.write(f'{current}\t{",".join(sorted(matched))}\n')
                    cf.write(f'{current}\t{",".join(sorted(cands))}\n')
                else:
                    mf.write(f'{current}\t\n')
                    cf.write(f'{current}\t\n')
                current = next(pop_iter, None)

        for s1_id, cid, score in cur:
            if s1_id != cur_s1:
                if cur_s1 is not None:
                    flush_to(cur_s1)
                cur_s1, matched, cands = s1_id, set(), set()
            cands.add(cid)
            if score is not None and score >= thr:
                matched.add(cid); n_pred += 1
        if cur_s1 is not None:
            flush_to(cur_s1)
        flush_to(None)
        mf.close(); cf.close()
    print(f'FALLBACK done: {len(population):,} entities, {n_pred:,} predicted @ thr={thr}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cand-db', default='reports/stage3/test/candidates.sqlite')
    p.add_argument('--threshold', type=float, default=0.55)
    p.add_argument('--output-dir', default='output')
    run(p.parse_args())


if __name__ == '__main__':
    main()
