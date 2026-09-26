"""Run full test retrieval in parallel rid-modulo shards, then merge.

Single-threaded test retrieval over 1.7M entities is ~days. This launches N
processes of stage3_inference, each covering a disjoint slice (rid % N == k) and
writing its OWN candidates.sqlite (no shared writer -> no lock/crash), then merges
them into one candidates.sqlite that Stage 4 consumes.

Prerequisite: build the test indexes once with `stage3_inference.py --index-only`.

  python run_test_sharded.py --shards 8 --output-dir reports/stage3/test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

from blocking_index import connect, log

HERE = Path(__file__).resolve().parent


def launch_shards(n, base, index_dir, extra):
    procs = []
    for k in range(n):
        out = base / f'shard{k}'
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', '-X', 'utf8', str(HERE / 'stage3_inference.py'),
               '--index-dir', str(index_dir), '--output-dir', str(out),
               '--shard-count', str(n), '--shard-index', str(k)] + extra
        logf = open(base / f'shard{k}.log', 'w', encoding='utf-8')
        procs.append((k, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf))
        log(f'launched shard {k}/{n} -> {out}')
    return procs


def wait_all(procs):
    failed = []
    for k, p, logf in procs:
        rc = p.wait()
        logf.close()
        log(f'shard {k} exited rc={rc}')
        if rc != 0:
            failed.append(k)
    if failed:
        raise SystemExit(f'shards failed: {failed} (see per-shard logs)')


def merge(base, n, final_path):
    """Combine every shard's queries + candidates into one candidates.sqlite."""
    if final_path.exists():
        final_path.unlink()
    with closing(connect(final_path)) as db:
        db.executescript('''
            CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE queries(id TEXT PRIMARY KEY,country TEXT NOT NULL,
                name TEXT,address TEXT,houses TEXT,postals TEXT,done2 INTEGER DEFAULT 0,done3 INTEGER DEFAULT 0);
            CREATE TABLE candidates(s1_id TEXT,candidate_id TEXT,source INTEGER,
                routes TEXT,best_score REAL,metadata TEXT,PRIMARY KEY(s1_id,candidate_id)) WITHOUT ROWID;
        ''')
        for k in range(n):
            shard_db = base / f'shard{k}' / 'candidates.sqlite'
            db.execute('ATTACH DATABASE ? AS s', (str(shard_db),))
            with db:
                db.execute('INSERT INTO queries SELECT * FROM s.queries')
                db.execute('INSERT INTO candidates SELECT * FROM s.candidates')
            db.execute('DETACH DATABASE s')
            log(f'merged shard {k}')
        nq = db.execute('SELECT count(*) FROM queries').fetchone()[0]
        nc = db.execute('SELECT count(*) FROM candidates').fetchone()[0]
    log(f'merged total: {nq:,} queries, {nc:,} candidate pairs -> {final_path}')
    print(f'MERGED: {nq:,} queries, {nc:,} candidates -> {final_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--index-dir', default='artifacts/stage3/test_indexes')
    p.add_argument('--output-dir', default='reports/stage3/test')
    p.add_argument('--idf', default='artifacts/stage3/indexes/idf_grouped.sqlite')
    p.add_argument('--limit', type=int, default=0)
    a, extra = p.parse_known_args()
    base = Path(a.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    passthrough = ['--idf', a.idf, '--limit', str(a.limit)] + extra
    started = time.monotonic()
    procs = launch_shards(a.shards, base, Path(a.index_dir), passthrough)
    wait_all(procs)
    merge(base, a.shards, base / 'candidates.sqlite')
    log(f'sharded test retrieval done in {time.monotonic()-started:.0f}s')


if __name__ == '__main__':
    main()
