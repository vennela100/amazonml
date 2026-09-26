"""Stage 3 (test inference): retrieve candidates for the unlabeled test set.

The training Stage 3 (`stage3.py`) is bound to the training splits DB: it selects
queries by grouped role and scores against ground truth. The test set has no
labels and no splits, so this driver provides the missing path:

  * build blocking indexes over test_source1/2/3 (splits_path=None, no roles),
  * reuse the TRAIN-fitted IDF so retrieval scores stay on the same scale the
    Stage 5 model was trained on,
  * populate the query table from every test_source1 record (optionally limited
    or filtered by country for a bounded demo, e.g. to include France),
  * retrieve S2/S3 candidates with the same Retriever, and
  * write candidates.sqlite in the identical schema Stage 4 consumes.

There is no evaluation step here — the test answers are hidden. Feed the output
to Stage 4 (`--restrict`) and Stage 6 to produce matching_results.tsv.
"""
from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from blocking_index import (build_index, connect, get_state, load_idf, log,
                            prepare_statistics, set_state)
from stage3 import ROUTES, run_source


def expected_rows(stage2_report_dir, source):
    """Exact row count for a normalized test file (build_index requires a match)."""
    report = Path(stage2_report_dir) / f'test_source{source}.tsv.json'
    return json.loads(report.read_text(encoding='utf-8'))['rows']


def prepare_test_queries(db, source1_index, limit=0, countries=None):
    """Populate the query table from the test source1 index (no splits/roles).

    ``limit`` caps the number of queries (0 = all). ``countries`` optionally
    restricts to a set of country labels (e.g. {'France'}); country stays an
    open string label — nothing is hard-coded to a fixed set.
    """
    db.executescript('''
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS queries(id TEXT PRIMARY KEY,country TEXT NOT NULL,
            name TEXT,address TEXT,houses TEXT,postals TEXT,done2 INTEGER DEFAULT 0,done3 INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS candidates(s1_id TEXT,candidate_id TEXT,source INTEGER,
            routes TEXT,best_score REAL,metadata TEXT,PRIMARY KEY(s1_id,candidate_id)) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS pending2 ON queries(id) WHERE done2=0;
        CREATE INDEX IF NOT EXISTS pending3 ON queries(id) WHERE done3=0;
    ''')
    if get_state(db, 'queries_ready', False):
        return
    where, params = '', []
    if countries:
        placeholders = ','.join('?' * len(countries))
        where = f' WHERE country IN ({placeholders})'
        params = list(countries)
    sql = ('SELECT entity_id,country,name,address,houses,postals FROM records'
           + where + ' ORDER BY rid' + (' LIMIT ?' if limit else ''))
    if limit:
        params = params + [limit]
    with closing(connect(source1_index, True)) as src:
        with db:
            db.executemany(
                'INSERT INTO queries(id,country,name,address,houses,postals) VALUES (?,?,?,?,?,?)',
                src.execute(sql, params))
            total = db.execute('SELECT count(*) FROM queries').fetchone()[0]
            set_state(db, 'population_entities', total)
            set_state(db, 'queries_ready', True)
    log(f'Prepared {total:,} test queries'
        + (f' (countries={sorted(countries)})' if countries else ''))


def run(args):
    norm = Path(args.normalized_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.output_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    indexes = {i: index_dir / f'test_source{i}.sqlite' for i in (1, 2, 3)}
    for source in (1, 2, 3):
        path = norm / f'test_source{source}.tsv'
        build_index(path, indexes[source], expected_rows(args.stage2_report_dir, source))
        if source in (2, 3):
            prepare_statistics(indexes[source])
    if args.index_only:
        log('Index-only: test indexes built; stopping before retrieval.')
        print('STAGE 3 TEST INDEXES READY')
        return

    idf = load_idf(args.idf)
    config = {'anchors': args.anchors, 'postings_per_anchor': args.postings_per_anchor,
              'shortlist': args.shortlist, 'rare_max_df': args.rare_max_df,
              'component_max_df': args.component_max_df,
              'budgets': {route: getattr(args, 'topk_' + route) for route in ROUTES}}

    with closing(connect(out_dir / 'candidates.sqlite')) as db:
        prepare_test_queries(db, indexes[1], args.limit,
                             set(args.countries) if args.countries else None)
        (out_dir / 'config.json').write_text(
            json.dumps({'retrieval': config, 'mode': 'test', 'idf': str(args.idf),
                        'limit': args.limit, 'countries': args.countries}, indent=2),
            encoding='utf-8')
        for source in (2, 3):
            run_source(db, indexes[source], idf, config, source)
        done = db.execute('SELECT count(*) FROM candidates').fetchone()[0]
        n_q = db.execute('SELECT count(*) FROM queries').fetchone()[0]
    log(f'Test retrieval complete: {n_q:,} queries, {done:,} candidate pairs -> {out_dir}')
    print(f'STAGE 3 TEST PASS: {n_q:,} queries, {done:,} candidates -> {out_dir}/candidates.sqlite')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normalized-dir', default='artifacts/stage2/test')
    p.add_argument('--stage2-report-dir', default='reports/stage2')
    p.add_argument('--index-dir', default='artifacts/stage3/test_indexes')
    p.add_argument('--output-dir', default='reports/stage3/test')
    p.add_argument('--idf', default='artifacts/stage3/indexes/idf_grouped.sqlite',
                   help='Reuse the TRAIN-fitted IDF for feature-scale consistency')
    p.add_argument('--limit', type=int, default=0, help='Cap queries (0 = all)')
    p.add_argument('--index-only', action='store_true',
                   help='Build test indexes then stop (prerequisite step)')
    p.add_argument('--countries', nargs='*', default=None,
                   help='Restrict to country labels, e.g. --countries France')
    for name, default in [('anchors', 8), ('postings-per-anchor', 96),
                          ('shortlist', 96), ('rare-max-df', 100), ('component-max-df', 5000)]:
        p.add_argument('--' + name, type=int, default=default)
    for route in ROUTES:
        p.add_argument('--topk-' + route, type=int, default=50)
    run(p.parse_args())


if __name__ == '__main__':
    main()
