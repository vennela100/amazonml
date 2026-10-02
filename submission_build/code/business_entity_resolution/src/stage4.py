"""Stage 4: Pairwise feature engineering on Stage 3 candidates.

Reads:
  - Stage 2 normalized TSVs (for S1, S2, S3 records)
  - Stage 3 candidates.sqlite (for candidate pairs and retrieval metadata)
  - Stage 1 splits.sqlite (for ground-truth labels and split roles)

Writes:
  - reports/stage4/features.npz  (X matrix, y labels, pair IDs)
  - reports/stage4/rare_vocab.json
  - reports/stage4/report.json + report.md

Memory strategy:
  - Load all S1 records into a dict (2.2M * ~200 bytes ~ 400 MB)
  - Load S2 and S3 records into separate dicts (5M * ~200 bytes ~ 1 GB each)
    — only one pool at a time to cap peak RAM
  - Stream candidate pairs from SQLite, batch-compute features, write to NPZ
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import N_FEATURES, FEATURE_NAMES, parse_row, compute_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

NORM_HEADER = [
    'entity_id', 'business_name', 'business_address', 'country',
    'name_clean', 'name_normalized', 'address_normalized',
    'address_numbers', 'house_numbers', 'house_number_candidates',
    'house_number_source', 'street_numbers', 'postal_codes',
    'postal_candidates', 'postal_code_source',
    'name_missing', 'address_missing', 'name_null_like', 'address_null_like',
    'name_normalized_empty', 'address_normalized_empty',
    'house_number_missing', 'house_number_ambiguous',
    'postal_code_missing', 'postal_code_ambiguous'
]

COL_ENTITY_ID = NORM_HEADER.index('entity_id')


def load_records(tsv_path: str | Path, keep: set[str] | None = None) -> dict[str, dict]:
    """Load a normalized TSV into a dict keyed by entity_id.

    When ``keep`` is given, only rows whose entity_id is in that set are parsed
    and retained. This lets a targeted run (e.g. a pilot) hold a small slice of a
    multi-million-row source in memory instead of the whole file.
    """
    path = Path(tsv_path)
    records = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if header != NORM_HEADER:
            raise ValueError(f'{path.name}: unexpected header (first 5): {header[:5]}')
        eid = COL_ENTITY_ID
        for row in reader:
            if keep is not None and row[eid] not in keep:
                continue
            r = parse_row(row)
            records[r['entity_id']] = r
    log.info(f'Loaded {len(records):,} records from {path.name}'
             + ('' if keep is None else f' (filtered to {len(keep):,} ids)'))
    return records


def needed_ids(cand_db: str | Path) -> set[str]:
    """Return every candidate_id referenced in the candidates table."""
    with closing(sqlite3.connect(str(cand_db))) as db:
        return {row[0] for row in db.execute('SELECT candidate_id FROM candidates')}


def build_rare_vocab(
    records: dict[str, dict],
    max_df_frac: float = 0.001,
    min_df: int = 2
) -> set[str]:
    """Build a set of rare tokens from all pool name_norm fields.

    A token is rare if it appears in at least min_df records but no more than
    max_df_frac * total records. Avoids country-specific thresholds.
    """
    token_df: Counter = Counter()
    for r in records.values():
        for tok in r['name_norm'].split():
            if len(tok) >= 3:
                token_df[tok] += 1
    total = len(records)
    max_df = max(min_df, int(max_df_frac * total))
    rare = {t for t, df in token_df.items() if min_df <= df <= max_df}
    log.info(f'Rare vocab: {len(rare):,} tokens (df {min_df}–{max_df} out of {total:,})')
    return rare


def load_ground_truth(splits_db: str | Path) -> dict[str, set[str]]:
    """Load matched_ids per S1 entity from the splits database."""
    truth = {}
    with closing(sqlite3.connect(str(splits_db))) as db:
        for s1_id, matched_ids_str in db.execute(
                'SELECT id, matched_ids FROM entities ORDER BY id'):
            if matched_ids_str:
                truth[s1_id] = set(matched_ids_str.split(','))
            else:
                truth[s1_id] = set()
    log.info(f'Loaded ground truth for {len(truth):,} S1 entities')
    return truth


def load_split_roles(splits_db: str | Path) -> dict[str, str]:
    """Return dict[s1_id -> grouped_role ('train' or 'validation')]."""
    roles = {}
    with closing(sqlite3.connect(str(splits_db))) as db:
        for s1_id, role in db.execute('SELECT id, grouped_role FROM entities'):
            roles[s1_id] = role
    return roles


def load_queried_population(cand_db: str | Path) -> list[str]:
    """Return the exact S1 entities Stage 3 queried, sorted and deduplicated.

    This is the population the model scores against. Entities that were queried
    but retrieved zero candidates still belong here (empty prediction), so the
    population is taken from the ``queries`` table, not from the candidate rows.
    """
    with closing(sqlite3.connect(str(cand_db))) as db:
        rows = db.execute('SELECT id FROM queries ORDER BY id').fetchall()
    return [r[0] for r in rows]


def process_pairs(
    cand_db: str | Path,
    s1_records: dict[str, dict],
    pool_records: dict[str, dict],  # S2 + S3 merged
    truth: dict[str, set[str]],
    rare_vocab: set[str],
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Stream candidate pairs, compute features, and return arrays.

    Returns:
        X: (n_pairs, N_FEATURES) float32
        y: (n_pairs,) int8  — 1 if pair is a true match, 0 otherwise
        s1_ids: list of S1 entity IDs (length n_pairs)
        cand_ids: list of candidate IDs (length n_pairs)
    """
    X_batches, y_batches, s1_id_list, cand_id_list = [], [], [], []
    X_buf, y_buf = [], []

    missing_s1, missing_cand = 0, 0
    total_pairs = 0

    with closing(sqlite3.connect(str(cand_db))) as db:
        cur = db.execute(
            'SELECT s1_id, candidate_id, source, routes, best_score, metadata '
            'FROM candidates ORDER BY s1_id'
        )

        for s1_id, cand_id, source, routes, best_score, metadata in cur:
            s1_rec = s1_records.get(s1_id)
            cand_rec = pool_records.get(cand_id)

            if s1_rec is None:
                missing_s1 += 1
                continue
            if cand_rec is None:
                missing_cand += 1
                continue

            label = 1 if (cand_id in truth.get(s1_id, set())) else 0
            feat = compute_features(s1_rec, cand_rec, routes or '', best_score,
                                    source, metadata, rare_vocab)

            X_buf.append(feat)
            y_buf.append(label)
            s1_id_list.append(s1_id)
            cand_id_list.append(cand_id)
            total_pairs += 1

            if len(X_buf) >= batch_size:
                X_batches.append(np.array(X_buf, dtype=np.float32))
                y_batches.append(np.array(y_buf, dtype=np.int8))
                X_buf.clear()
                y_buf.clear()
                if total_pairs % 1_000_000 == 0:
                    log.info(f'  Processed {total_pairs:,} pairs so far '
                             f'(missing s1={missing_s1}, missing cand={missing_cand})')

        if X_buf:
            X_batches.append(np.array(X_buf, dtype=np.float32))
            y_batches.append(np.array(y_buf, dtype=np.int8))

    log.info(f'Total pairs: {total_pairs:,}, missing_s1={missing_s1}, missing_cand={missing_cand}')
    if not X_batches:
        return (np.zeros((0, N_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.int8), [], [])

    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    return X, y, s1_id_list, cand_id_list


def report_statistics(
    X: np.ndarray,
    y: np.ndarray,
    s1_ids: list[str],
    cand_ids: list[str],
    split_roles: dict[str, str],
    truth: dict[str, set[str]],
) -> dict:
    """Compute and return a statistics dict for the report."""
    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos
    pos_rate = n_pos / n_total if n_total else 0

    # Per-split stats
    train_mask = np.array([split_roles.get(s, '') == 'train' for s in s1_ids])
    val_mask = ~train_mask

    def split_stats(mask):
        yt = y[mask]
        return {
            'pairs': int(mask.sum()),
            'positive': int(yt.sum()),
            'negative': int((yt == 0).sum()),
            'positive_rate': float(yt.mean()) if len(yt) else 0,
        }

    # Feature stats (mean, std for each feature)
    feat_means = X.mean(axis=0).tolist()
    feat_stds = X.std(axis=0).tolist()

    # Hard negatives: negatives with high name similarity
    high_name = X[:, FEATURE_NAMES.index('name_edit_ratio')] > 0.80
    hard_neg_count = int(((y == 0) & high_name).sum())

    return {
        'n_total_pairs': n_total,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'positive_rate': pos_rate,
        'hard_negatives_high_name_sim': hard_neg_count,
        'hard_neg_frac': hard_neg_count / n_neg if n_neg else 0,
        'n_features': N_FEATURES,
        'train': split_stats(train_mask),
        'validation': split_stats(val_mask),
        'feature_means': {k: round(v, 4) for k, v in zip(FEATURE_NAMES, feat_means)},
        'feature_stds': {k: round(v, 4) for k, v in zip(FEATURE_NAMES, feat_stds)},
    }


def render_report(stats: dict, path: Path) -> None:
    lines = [
        '# Stage 4: Pairwise Feature Engineering', '',
        f'**Status**: PASS', '',
        '## Dataset Summary', '',
        f'- Total pairs: {stats["n_total_pairs"]:,}',
        f'- Positive (true matches): {stats["n_positive"]:,} ({stats["positive_rate"]*100:.2f}%)',
        f'- Negative: {stats["n_negative"]:,}',
        f'- Hard negatives (name sim > 0.80): {stats["hard_negatives_high_name_sim"]:,} '
        f'({stats["hard_neg_frac"]*100:.1f}% of negatives)',
        f'- Features: {stats["n_features"]}', '',
        '## Split Breakdown', '',
        '| Split | Pairs | Positives | Negatives | Pos rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for split in ('train', 'validation'):
        s = stats[split]
        lines.append(f'| {split} | {s["pairs"]:,} | {s["positive"]:,} | '
                     f'{s["negative"]:,} | {s["positive_rate"]*100:.2f}% |')
    lines += [
        '', '## Top Feature Means (positives vs negatives analysis available in report.json)', '',
        '| Feature | Mean | Std |',
        '|---|---:|---:|',
    ]
    for fname in FEATURE_NAMES[:20]:  # top 20 in order
        lines.append(f'| {fname} | {stats["feature_means"][fname]:.4f} | '
                     f'{stats["feature_stds"][fname]:.4f} |')
    lines += ['', '## Next Steps', '',
              '- Stage 5: Train CatBoost classifier on these features with entity-grouped CV',
              '- Hard negative fraction should be > 5% for good calibration',
              '- If positive rate is too low (< 0.5%), consider stratified sampling', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    started = time.monotonic()
    data_dir = Path(args.data_dir)
    split = args.split
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cand_db = Path(args.cand_db)
    if not cand_db.exists():
        raise FileNotFoundError(f'Candidates DB not found: {cand_db}')

    splits_db = Path(args.splits_db)
    if not splits_db.exists():
        raise FileNotFoundError(f'Splits DB not found: {splits_db}')

    # When restricting, only load the records the candidate DB actually references.
    s1_keep = pool_keep = None
    if args.restrict:
        pool_keep = needed_ids(cand_db)
        s1_keep = set(load_queried_population(cand_db))
        log.info(f'Restricting load to {len(s1_keep):,} S1 and '
                 f'{len(pool_keep):,} pool ids referenced by candidates')

    # Load S1 records
    log.info('Loading S1 records...')
    s1_path = data_dir / split / f'{split}_source1.tsv'
    s1_records = load_records(s1_path, keep=s1_keep)

    # Load S2 and S3 pool records (merge into one dict)
    log.info('Loading S2 records...')
    s2_path = data_dir / split / f'{split}_source2.tsv'
    pool_records = load_records(s2_path, keep=pool_keep)
    gc.collect()

    log.info('Loading S3 records...')
    s3_path = data_dir / split / f'{split}_source3.tsv'
    s3_records = load_records(s3_path, keep=pool_keep)
    pool_records.update(s3_records)
    del s3_records
    gc.collect()
    log.info(f'Combined pool: {len(pool_records):,} records (S2+S3)')

    # Build rare vocab from the combined pool name_norm fields
    log.info('Building rare vocabulary...')
    rare_vocab = build_rare_vocab(pool_records, max_df_frac=args.rare_max_df_frac)
    rare_vocab_path = report_dir / 'rare_vocab.json'
    rare_vocab_path.write_text(
        json.dumps(sorted(rare_vocab)[:50000], ensure_ascii=False),  # cap list size
        encoding='utf-8'
    )
    log.info(f'Saved rare vocab ({len(rare_vocab):,} tokens) to {rare_vocab_path}')

    # Load ground truth and split roles
    truth = load_ground_truth(splits_db)
    split_roles = load_split_roles(splits_db)

    # Exact population Stage 3 queried — the scope the model is scored over.
    population = load_queried_population(cand_db)
    log.info(f'Queried population: {len(population):,} S1 entities')

    # Compute features
    log.info('Computing pairwise features...')
    X, y, s1_id_list, cand_id_list = process_pairs(
        cand_db, s1_records, pool_records, truth, rare_vocab,
        batch_size=args.batch_size
    )
    log.info(f'Feature matrix: {X.shape}, dtype={X.dtype}')

    # Persist the population, per-entity roles, and complete truth alongside the
    # pair features. Downstream scoring (evaluation.py) needs all three to charge
    # missed matches and singletons correctly — the pair table alone cannot.
    pop_arr = np.array(population)
    pop_roles = np.array([split_roles.get(s, '') for s in population])
    truth_flat = np.array(
        [f'{s}\t{",".join(sorted(truth.get(s, set())))}' for s in population])

    out_path = report_dir / 'features.npz'
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        s1_ids=np.array(s1_id_list),
        cand_ids=np.array(cand_id_list),
        feature_names=np.array(FEATURE_NAMES),
        population=pop_arr,
        population_roles=pop_roles,
        population_truth=truth_flat,
    )
    log.info(f'Saved features to {out_path} ({out_path.stat().st_size/1024**2:.0f} MB)')

    # Statistics and report
    stats = report_statistics(X, y, s1_id_list, cand_id_list, split_roles, truth)
    stats['elapsed_seconds'] = time.monotonic() - started
    stats['feature_names'] = FEATURE_NAMES

    report_json = report_dir / 'report.json'
    report_json.write_text(json.dumps(stats, indent=2), encoding='utf-8')

    report_md = report_dir / 'report.md'
    render_report(stats, report_md)

    log.info(f'\nSTAGE 4 PASS: {report_md}')
    log.info(f'Elapsed: {stats["elapsed_seconds"]:.0f}s')
    print(f'\nSTAGE 4 PASS')
    print(f'  Pairs: {stats["n_total_pairs"]:,}')
    print(f'  Positives: {stats["n_positive"]:,} ({stats["positive_rate"]*100:.2f}%)')
    print(f'  Hard negatives: {stats["hard_negatives_high_name_sim"]:,} '
          f'({stats["hard_neg_frac"]*100:.1f}%)')
    print(f'  Features: {N_FEATURES}')
    print(f'  Report: {report_md}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='artifacts/stage2')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--cand-db', default='reports/stage3/candidates.sqlite')
    parser.add_argument('--splits-db', default='reports/stage1/splits.sqlite')
    parser.add_argument('--report-dir', default='reports/stage4')
    parser.add_argument('--batch-size', type=int, default=100_000)
    parser.add_argument('--rare-max-df-frac', type=float, default=0.001,
                        help='Max pool fraction for a token to be considered rare')
    parser.add_argument('--restrict', action='store_true',
                        help='Load only records referenced by the candidate DB '
                             '(fast pilot runs; rare vocab is then pool-restricted)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
