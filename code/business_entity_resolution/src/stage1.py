"""Build and validate the scoring harness and entity splits; no modeling."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib

from scoring import MATCH_HEADER, MacroMetrics, parse_ids, read_tsv, validate_s1
from splits import assign_grouped, entity_hash, make_modes, role_for, selection


def log(message):
    print(message, flush=True)


def fingerprint(path):
    digest, crc, size = hashlib.sha256(), 0, 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {'sha256': digest.hexdigest(), 'crc32': f'{crc:08x}', 'bytes': size}


def insert_batches(db, sql, rows, batch_size=25000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            db.executemany(sql, batch)
            db.commit()
            batch.clear()
    if batch:
        db.executemany(sql, batch)
        db.commit()


def build_database(db, train_dir, seed):
    db.executescript('''
        CREATE TABLE source1 (id TEXT PRIMARY KEY, country TEXT NOT NULL,
                              split_hash TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE labels (id TEXT PRIMARY KEY, matched_ids TEXT NOT NULL,
                             match_count INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE entities (id TEXT PRIMARY KEY, country TEXT NOT NULL,
            matched_ids TEXT NOT NULL, match_count INTEGER NOT NULL,
            split_hash TEXT NOT NULL, grouped_role TEXT NOT NULL
            CHECK(grouped_role IN ('train','validation'))) WITHOUT ROWID;
    ''')

    def source_rows():
        for entity_id, _, _, country in read_tsv(train_dir / 'train_source1.tsv',
                ['entity_id', 'business_name', 'business_address', 'country']):
            validate_s1(entity_id)
            if not country or country != country.strip():
                raise ValueError(f'Missing or padded country for {entity_id}')
            yield entity_id, country, entity_hash(entity_id, seed)

    def label_rows():
        for entity_id, text in read_tsv(train_dir / 'train_ground_truth.tsv', MATCH_HEADER):
            validate_s1(entity_id)
            yield entity_id, text, len(parse_ids(text))

    log('Loading every training S1 and its complete ground-truth list...')
    insert_batches(db, 'INSERT INTO source1 VALUES (?,?,?)', source_rows())
    insert_batches(db, 'INSERT INTO labels VALUES (?,?,?)', label_rows())
    missing = db.execute('SELECT COUNT(*) FROM source1 s LEFT JOIN labels l ON s.id=l.id '
                         'WHERE l.id IS NULL').fetchone()[0]
    extra = db.execute('SELECT COUNT(*) FROM labels l LEFT JOIN source1 s ON s.id=l.id '
                       'WHERE s.id IS NULL').fetchone()[0]
    if missing or extra:
        raise ValueError(f'Ground-truth coverage error: {missing} missing, {extra} extra')
    db.execute("INSERT INTO entities SELECT s.id,s.country,l.matched_ids,l.match_count,"
               "s.split_hash,'train' FROM source1 s JOIN labels l ON s.id=l.id")
    db.executescript('DROP TABLE source1; DROP TABLE labels; '
                     'CREATE INDEX strata ON entities(country,match_count,split_hash,id);')
    db.commit()


def baseline_checks(db, modes):
    metrics = defaultdict(MacroMetrics)
    country_metrics = defaultdict(MacroMetrics)
    for _, country, text, count, grouped_role in db.execute(
            'SELECT id,country,matched_ids,match_count,grouped_role FROM entities ORDER BY id'):
        truth = parse_ids(text)
        if len(truth) != count:
            raise AssertionError('Stored match count does not equal complete label list')
        metrics['all'].add(truth, set())
        metrics['perfect_oracle'].add(truth, truth)
        for name, mode in modes.items():
            role = role_for(mode, country, grouped_role)
            if role:
                key = f'{name}/{role}'
                metrics[key].add(truth, set())
                country_metrics[(key, country)].add(truth, set())
    results = {key: value.result() for key, value in metrics.items()}
    for key, result in results.items():
        expected = 1.0 if key == 'perfect_oracle' else result['singleton_fraction']
        if result['macro_f05'] != expected:
            raise AssertionError(f'{key}: scorer failed analytic baseline: {result}')
    slices = defaultdict(dict)
    for (key, country), metric in country_metrics.items():
        slices[key][country] = metric.result()
    for name, mode in modes.items():
        for role in ('train', 'validation'):
            key = f'{name}/{role}'
            if key not in results:
                raise ValueError(f'Empty fold: {key}')
            predicate, params = selection(mode, role)
            counts = db.execute('SELECT COUNT(*), SUM(match_count=0) FROM entities e WHERE '
                                + predicate, params).fetchone()
            if counts != (results[key]['entities'], results[key]['true_singletons']):
                raise AssertionError(f'SQL selection and scoring selection disagree: {key}')
        train_pred, train_params = selection(mode, 'train')
        val_pred, val_params = selection(mode, 'validation')
        overlap = db.execute('SELECT COUNT(*) FROM entities e WHERE (' + train_pred +
                             ') AND (' + val_pred + ')', train_params + val_params).fetchone()[0]
        if overlap:
            raise AssertionError(f'Entity leakage: {name}')
    return results, dict(slices)


def render_report(report, destination):
    lines = ['# Stage 1: scorer and validation splits', '',
             '**PASS**. No normalization, candidate generation, or training performed.', '',
             '## Scoring', '',
             'Each S1 contributes exactly one F_0.5 value to the arithmetic mean. '
             'For nonempty truth the formula is `5 TP / (5 TP + 4 FP + FN)`. '
             'Empty truth scores 1 for empty prediction, otherwise 0. '
             'Empty predictions on non-singletons score 0.', '',
             'Full-data empty predictions reproduce Stage 0; a perfect-label oracle scores '
             '1.0. The oracle is a scorer sanity check, not a trained model or validation result.', '',
             '## Split construction', '',
             f'Seed: {report["seed"]}; requested validation fraction: {report["validation_fraction"]}. '
             'Strata are (country, exact true-match count). Within each stratum, sort by '
             'SHA256(JSON([seed, S1 ID])), breaking ties by ID. Round validation size half up; '
             'keep both roles nonempty when a stratum has at least two entities. '
             'One-entity strata stay in training. Input ordering does not affect assignments.', '',
             'An S1 and its full positive set always travel together. Current labels have no '
             'candidate owned by multiple S1 entities, verified in Stage 0 and protected here '
             'by matching source-file CRC and size before reuse of that check. '
             'No test data or labels are consulted.', '',
             '| Mode / role | S1 entities | Singletons | All-empty macro F_0.5 |',
             '|---|---:|---:|---:|']
    for key, result in sorted(report['baseline_metrics'].items()):
        if key != 'perfect_oracle':
            lines.append(f'| {key} | {result["entities"]:,} | {result["true_singletons"]:,} | '
                         f'{result["macro_f05"]:.10f} |')
    lines += ['', '## Country transfer', '']
    for key, mode in report['modes'].items():
        if mode['type'] == 'country_transfer':
            lines.append(f'- `{key}`: train on **{mode["train_country"]}**, validate on '
                         f'**{mode["validation_country"]}**. Other countries, if present, are excluded.')
    lines += ['', 'Country modes are generated from observed strings, with no US/India whitelist. '
              'Each transfer mode uses the entire training-country S1 population for training '
              'and the entire validation-country S1 population for validation, independently '
              'of the grouped holdout. These stress tests approximate geographic shift; '
              'they cannot estimate France performance directly.', '',
              '## How later stages must use these splits', '',
              '- Retrieve against the supplied training S2/S3 pool to retain realistic distractors; '
              'candidate appearance alone does not make its label available for fitting.',
              '- Fit pair classifiers only on training S1 groups. Never add validation positives '
              'as training examples or mine negatives from validation S1 labels.',
              '- Fit learned text transformations/vocabularies on the training partition. In strict '
              'country transfer, fit those on training-country records only; transform other '
              'countries without refitting. Fixed rules may process any country.',
              '- Threshold selection on validation is tuning, not an unbiased final score. '
              'Use grouped out-of-fold predictions/nested evaluation for later decision models.',
              '- Deduplicated S1 IDs are the supplied grouping unit. Semantically duplicated '
              'businesses or chains across distinct IDs are not ruled out by structural checks.', '',
              '## Files and next step', '',
              '`splits.tsv` contains one row per S1, country, true-match count, and a role column '
              'for every mode. `splits.sqlite` stores complete labels and supports bounded-memory '
              'loading/scoring. `manifest.json` records definitions, source hashes, and split-file '
              'hash. `report.json` includes exact per-country metrics and per-stratum sizes.', '',
              'Stage 2 is pending review. If later validation scores are weak, first check '
              'candidate misses and false merges separately, then compare country, missing-address, '
              'and multi-match slices. Do not change the scorer or weight high-match entities more.', '']
    destination.write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    started = time.monotonic()
    if not 0 < args.validation_fraction < 1:
        raise ValueError('--validation-fraction must be strictly between 0 and 1')
    audit = json.loads((args.audit_dir / 'audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or audit['integrity']['targets_assigned_to_multiple_s1']:
        raise ValueError('A passing Stage 0 with no shared positive targets is required')
    inventory = {item['path']: item for item in audit['archive_inventory']}
    fingerprints = {}
    for filename in ('train_source1.tsv', 'train_ground_truth.tsv'):
        info = fingerprint(args.train_dir / filename)
        original = inventory['student_resource/dataset/train/' + filename]
        if (info['crc32'], info['bytes']) != (original['crc32'], original['bytes']):
            raise ValueError(f'{filename} changed since Stage 0; rerun the audit first')
        fingerprints[filename] = info
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir / 'splits.sqlite'
    if database.exists():
        raise FileExistsError(f'{database} exists; choose a fresh --output-dir')
    with closing(sqlite3.connect(database)) as db:
        db.execute('PRAGMA cache_size=-65536')
        db.execute('PRAGMA temp_store=FILE')
        build_database(db, args.train_dir, args.seed)
        log('Assigning deterministic S1-grouped folds...')
        strata = assign_grouped(db, args.validation_fraction)
        countries = [row[0] for row in db.execute('SELECT DISTINCT country FROM entities ORDER BY country')]
        modes = make_modes(countries)
        split_path = args.output_dir / 'splits.tsv'
        with split_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['source1_entity_id', 'country', 'match_count', *modes])
            for entity_id, country, count, grouped_role in db.execute(
                    'SELECT id,country,match_count,grouped_role FROM entities ORDER BY id'):
                writer.writerow([entity_id, country, count,
                                 *(role_for(mode, country, grouped_role) or 'excluded'
                                   for mode in modes.values())])
        log('Scoring all-empty and perfect-oracle predictions on every training entity...')
        baselines, country_baselines = baseline_checks(db, modes)
    if baselines['all']['macro_f05'] != audit['all_empty_baseline']['macro_f05']:
        raise AssertionError('Full-data baseline differs from Stage 0')
    if baselines['all']['entities'] != audit['all_empty_baseline']['entities']:
        raise AssertionError('Entity count differs from Stage 0')
    report = {'stage': 1, 'status': 'PASS', 'seed': args.seed,
              'validation_fraction': args.validation_fraction, 'modes': modes,
              'source_fingerprints': fingerprints, 'split_fingerprint': fingerprint(split_path),
              'baseline_metrics': baselines, 'per_country_baselines': country_baselines,
              'strata': strata, 's1_train_validation_overlap': 0,
              'shared_positive_targets': 0, 'elapsed_seconds': time.monotonic() - started}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest = {key: report[key] for key in ('seed', 'validation_fraction', 'modes',
                                            'source_fingerprints', 'split_fingerprint')}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    render_report(report, args.output_dir / 'report.md')
    for key, metric in baselines.items():
        log(f'{key}: entities={metric["entities"]:,}, singletons={metric["true_singletons"]:,}, '
            f'macro F_0.5={metric["macro_f05"]:.10f}')
    log(f'STAGE 1 PASS: {args.output_dir / "report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=Path('student_resource/dataset/train'))
    parser.add_argument('--audit-dir', type=Path, default=Path('reports/stage0'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/stage1'))
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
