"""Deadline orchestrator: guarantee a complete, valid submission by a cutoff time.

Full test retrieval may not finish before the deadline. This waits until the
retrieval either completes OR a hard cutoff is reached, then:
  1. stops any running shard processes,
  2. merges whatever candidates each shard has committed (partial is fine —
     each shard pre-loaded its full entity slice, so the merged population
     covers every test entity; un-retrieved ones simply have no candidates),
  3. builds features (Stage 4), runs the model (Stage 6),
  4. validates the two output files.

The result is always a valid submission for the full test population; quality
scales with how much retrieval finished. Run in the background well ahead of time.

  python orchestrate_deadline.py --cutoff 20:45
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent


def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)


def parse_cutoff(s):
    now = dt.datetime.now()
    h, m = map(int, s.split(':'))
    cut = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if cut <= now:
        cut += dt.timedelta(days=1)
    return cut


def retrieval_done(final_db):
    return final_db.exists()


def stop_shards():
    # Kill any stage3_inference / run_test_sharded python processes.
    ps = ('Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | '
          "Where-Object { $_.CommandLine -like '*stage3_inference*' -or "
          "$_.CommandLine -like '*run_test_sharded*' } | "
          'ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }')
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', ps], capture_output=True)


def run(cmd):
    log('RUN ' + ' '.join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f'command failed ({r.returncode}): {cmd}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoff', required=True, help='Local HH:MM hard cutoff')
    p.add_argument('--shards', type=int, default=6)
    p.add_argument('--test-dir', default='reports/stage3/test')
    p.add_argument('--output-dir', default='output')
    p.add_argument('--model', default='reports/stage5/model.cbm')
    p.add_argument('--decision', default='reports/stage5/decision.json')
    p.add_argument('--poll', type=int, default=120)
    a = p.parse_args()

    cutoff = parse_cutoff(a.cutoff)
    test_dir = ROOT / a.test_dir
    final_db = test_dir / 'candidates.sqlite'
    log(f'orchestrator armed; cutoff {cutoff:%Y-%m-%d %H:%M}')

    # 1. Wait for retrieval completion or cutoff.
    while dt.datetime.now() < cutoff and not retrieval_done(final_db):
        time.sleep(a.poll)

    if retrieval_done(final_db):
        log('retrieval completed on its own; using merged candidates.sqlite')
    else:
        log('CUTOFF reached — stopping shards and merging partial results')
        stop_shards()
        time.sleep(5)
        sys.path.insert(0, str(HERE))
        from run_test_sharded import merge
        merge(test_dir, a.shards, final_db)

    # 2+3. Memory-safe streaming prediction over the (possibly partial) candidates.
    # Population = all test S1 in the queries table; un-retrieved ones -> empty.
    out = ROOT / a.output_dir
    run([sys.executable, '-u', '-X', 'utf8', str(HERE / 'predict_streaming.py'),
         '--cand-db', str(final_db), '--model', a.model, '--decision', a.decision,
         '--data-dir', 'artifacts/stage2', '--split', 'test',
         '--output-dir', str(out)])

    # 4. Validate.
    run([sys.executable, str(ROOT / 'student_resource/utils/validate_submission.py'),
         '--matching', str(out / 'matching_results.tsv'),
         '--candidate', str(out / 'candidate_pairs.tsv'),
         '--test-dir', 'student_resource/dataset/test'])

    log(f'DONE — submission files in {out}')
    print('ORCHESTRATOR COMPLETE — output/matching_results.tsv ready to submit')


if __name__ == '__main__':
    main()
