"""Full-data, offline Stage 0 audit. No normalization, split, or model training."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import zipfile

import pandas as pd

SOURCE_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
TRUTH_COLUMNS = ['source1_entity_id', 'matched_entity_ids']
NULL_TOKENS = {'null', 'none', 'nan', 'na', 'n/a', '<na>'}


def log(message):
    print(message, flush=True)


def distribution(counts):
    """Exact integer histogram and linear-interpolated quantiles; no sampling."""
    counts = dict(sorted(counts.items()))
    n = sum(counts.values())
    if not n:
        return {'count': 0, 'histogram': {}}

    def at(index):
        cumulative = 0
        for value, frequency in counts.items():
            cumulative += frequency
            if cumulative > index:
                return value
        raise AssertionError(index)

    result = {'count': n, 'min': min(counts), 'max': max(counts),
              'mean': sum(k * v for k, v in counts.items()) / n,
              'histogram': counts}
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        position = (n - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        result[f'p{int(q * 100):02}'] = at(lo) + (at(hi) - at(lo)) * (position - lo)
    return result


def prepare(args, report):
    if args.pdf_tools_directory:
        sys.path.insert(0, str(args.pdf_tools_directory.resolve()))
    from pypdf import PdfReader
    pdf = PdfReader(args.pdf)
    pages = [page.extract_text(extraction_mode='layout') for page in pdf.pages]
    (args.report_dir / 'problem_statement.txt').write_text(
        '\n\n'.join(f'=== PAGE {i + 1} ===\n{text}' for i, text in enumerate(pages)),
        encoding='utf-8')
    report['pdf'] = {'path': str(args.pdf.resolve()), 'pages': len(pages),
                     'text_characters_per_page': [len(p.strip()) for p in pages],
                     'sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest()}
    with zipfile.ZipFile(args.archive) as archive:
        report['archive_inventory'] = [
            {'path': i.filename, 'bytes': i.file_size, 'crc32': f'{i.CRC:08x}'}
            for i in archive.infolist()]
        (args.report_dir / 'archive_inventory.json').write_text(
            json.dumps(report['archive_inventory'], indent=2), encoding='utf-8')
        selected = [i for i in archive.infolist()
                    if i.filename.startswith('student_resource/') and not i.is_dir()
                    and Path(i.filename).name != '.DS_Store']
        for info in selected:
            relative = Path(info.filename).relative_to('student_resource')
            destination = (args.resource_dir / relative).resolve()
            if not destination.is_relative_to(args.resource_dir.resolve()):
                raise ValueError(f'Unsafe archive member: {info.filename}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Always read the ZIP member to EOF: zipfile checks its CRC.
            with archive.open(info) as source, destination.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            log(f'Extracted and CRC-checked {relative} ({info.file_size:,} bytes)')


def check_shape(path, expected):
    counts = Counter()
    bad = []
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t', strict=True)
        header = next(reader)
        if header != expected:
            raise ValueError(f'{path}: expected {expected}, got {header}')
        for row in reader:
            counts[len(row)] += 1
            if len(row) != len(expected) and len(bad) < 5:
                bad.append(reader.line_num)
    if bad:
        raise ValueError(f'{path}: malformed column counts {counts}; lines {bad}')
    return {'header': header, 'column_count_histogram': dict(counts),
            'rows': sum(counts.values()), 'malformed_rows': 0}


def chunks(path, chunk_size):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                       na_filter=False, skip_blank_lines=False,
                       encoding='utf-8', on_bad_lines='error', chunksize=chunk_size)


def audit_source(path, split, source, db, chunk_size):
    result = check_shape(path, SOURCE_COLUMNS)
    countries, lengths = Counter(), Counter()
    blank = Counter({c: 0 for c in SOURCE_COLUMNS})
    null_like = Counter(blank)
    whitespace = Counter(blank)
    parsed_nulls = Counter(blank)
    missing_address_country = Counter()
    bad_prefix = 0
    processed = 0
    for frame in chunks(path, chunk_size):
        processed += len(frame)
        countries.update(frame.country.value_counts().to_dict())
        lengths.update(frame.business_name.str.len().value_counts().to_dict())
        for column in SOURCE_COLUMNS:
            raw = frame[column]
            stripped = raw.str.strip()
            blank[column] += int(stripped.eq('').sum())
            null_like[column] += int(stripped.str.lower().isin(NULL_TOKENS).sum())
            whitespace[column] += int(raw.ne(stripped).sum())
            parsed_nulls[column] += int(raw.isna().sum())
        missing = frame.business_address.str.strip().eq('')
        missing_address_country.update(frame.loc[missing, 'country'].value_counts().to_dict())
        bad_prefix += int((~frame.entity_id.str.startswith(f'S{source}-')).sum())
        db.executemany(f'INSERT INTO {split}_records VALUES (?, ?, ?)',
                       frame[['entity_id', 'country']].assign(source=source)
                       .itertuples(index=False, name=None))
        db.commit()
    if processed != result['rows']:
        raise ValueError(f'{path}: CSV and pandas row counts disagree')
    result.update(country_counts=dict(countries), blank_or_whitespace_fields=dict(blank),
                  literal_null_like_fields=dict(null_like), parser_null_fields=dict(parsed_nulls),
                  leading_or_trailing_whitespace=dict(whitespace), invalid_id_prefix=bad_prefix,
                  blank_address_by_country=dict(missing_address_country),
                  name_length_characters=distribution(lengths))
    log(f'{split}/source{source}: rows={processed:,}, countries={dict(countries)}, '
        f'blank addresses={blank["business_address"]:,}, '
        f'name length median={result["name_length_characters"]["p50"]}')
    return result


def audit_truth(path, db, chunk_size):
    result = check_shape(path, TRUTH_COLUMNS)
    histogram, types = Counter(), Counter()
    duplicate_lists = bad_tokens = 0
    for frame in chunks(path, chunk_size):
        heads, edges = [], []
        for s1, text in frame.itertuples(index=False, name=None):
            ids = text.split(',') if text else []
            duplicate_lists += int(len(ids) != len(set(ids)))
            bad_tokens += sum(not x.startswith(('S2-', 'S3-')) or x != x.strip() for x in ids)
            has2 = any(x.startswith('S2-') for x in ids)
            has3 = any(x.startswith('S3-') for x in ids)
            kind = 'both' if has2 and has3 else 'S2_only' if has2 else 'S3_only' if has3 else 'singleton'
            types[kind] += 1
            histogram[len(ids)] += 1
            heads.append((s1, len(ids), kind))
            edges.extend((s1, target) for target in ids)
        db.executemany('INSERT INTO truth VALUES (?, ?, ?)', heads)
        db.executemany('INSERT INTO edges VALUES (?, ?)', edges)
        db.commit()
    if sum(histogram.values()) != result['rows']:
        raise ValueError(f'{path}: CSV and pandas ground-truth row counts disagree')
    result.update(match_count=distribution(histogram), match_source_types=dict(types),
                  duplicate_ids_within_list_rows=duplicate_lists, invalid_match_tokens=bad_tokens)
    return result


def run(args):
    args.report_dir.mkdir(parents=True, exist_ok=True)
    database = args.report_dir / 'audit.sqlite'
    if database.exists():
        raise FileExistsError(f'{database} already exists; use a fresh --report-dir for a rerun')
    report = {'stage': 0, 'status': 'running', 'files': {},
              'missing_value_policy': 'Preserve all raw strings. Count blank/whitespace separately '
              'from whole-field literal null/none/nan/na/n/a/<na>; do not impute or reinterpret.',
              'environment': {'python': sys.version, 'pandas': pd.__version__}}
    started = time.monotonic()
    prepare(args, report)
    db = sqlite3.connect(database)
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA temp_store=FILE')
    for split in ('train', 'test'):
        db.execute(f'CREATE TABLE {split}_records (id TEXT, country TEXT, source INTEGER)')
    db.execute('CREATE TABLE truth (id TEXT, match_count INTEGER, kind TEXT)')
    db.execute('CREATE TABLE edges (s1 TEXT, target TEXT)')
    for split in ('train', 'test'):
        for source in (1, 2, 3):
            filename = f'{split}_source{source}.tsv'
            report['files'][filename] = audit_source(
                args.resource_dir / 'dataset' / split / filename, split, source, db, args.chunk_size)
        log(f'Indexing {split} IDs for exact uniqueness and referential checks...')
        db.execute(f'CREATE INDEX {split}_ids ON {split}_records(id)')
        db.commit()
    log('Auditing every ground-truth row and match ID...')
    truth = audit_truth(args.resource_dir / 'dataset/train/train_ground_truth.tsv', db, args.chunk_size)
    report['ground_truth'] = truth
    db.execute('CREATE INDEX truth_ids ON truth(id)')
    db.execute('CREATE INDEX edge_targets ON edges(target)')
    db.commit()
    queries = {
        'duplicate_train_id_groups': 'SELECT COUNT(*) FROM (SELECT id FROM train_records GROUP BY id HAVING COUNT(*)>1)',
        'duplicate_test_id_groups': 'SELECT COUNT(*) FROM (SELECT id FROM test_records GROUP BY id HAVING COUNT(*)>1)',
        'duplicate_truth_s1_groups': 'SELECT COUNT(*) FROM (SELECT id FROM truth GROUP BY id HAVING COUNT(*)>1)',
        'missing_truth_s1': 'SELECT COUNT(*) FROM train_records r LEFT JOIN truth t ON r.id=t.id WHERE r.source=1 AND t.id IS NULL',
        'unknown_truth_s1': 'SELECT COUNT(*) FROM truth t LEFT JOIN train_records r ON t.id=r.id AND r.source=1 WHERE r.id IS NULL',
        'unknown_truth_targets': 'SELECT COUNT(*) FROM edges e LEFT JOIN train_records r ON e.target=r.id AND r.source IN (2,3) WHERE r.id IS NULL',
        'targets_assigned_to_multiple_s1': 'SELECT COUNT(*) FROM (SELECT target FROM edges GROUP BY target HAVING COUNT(DISTINCT s1)>1)',
        'cross_country_true_pairs': 'SELECT COUNT(*) FROM edges e JOIN train_records a ON e.s1=a.id JOIN train_records b ON e.target=b.id WHERE a.country<>b.country',
        'train_test_id_overlap': 'SELECT COUNT(*) FROM train_records a JOIN test_records b ON a.id=b.id',
    }
    integrity = {}
    for name, query in queries.items():
        integrity[name] = db.execute(query).fetchone()[0]
        log(f'Integrity {name}: {integrity[name]:,}')
    report['integrity'] = integrity
    rows = db.execute('SELECT r.country, COUNT(*), SUM(t.match_count=0), SUM(t.match_count) '
                      'FROM truth t JOIN train_records r ON t.id=r.id GROUP BY r.country').fetchall()
    report['ground_truth_by_country'] = {
        country: {'entities': n, 'singletons': singletons, 'true_pairs': pairs,
                  'all_empty_macro_f05': singletons / n}
        for country, n, singletons, pairs in rows}
    n = truth['rows']
    singletons = truth['match_source_types'].get('singleton', 0)
    report['all_empty_baseline'] = {
        'entities': n, 'singletons': singletons, 'macro_f05': singletons / n,
        'derivation': 'Each true singleton scores 1; each non-singleton scores 0. '
                      'This is the analytic baseline, not the Stage 1 general scorer.'}
    errors = [f'{k}={v}' for k, v in integrity.items()
              if v and k not in ('train_test_id_overlap', 'cross_country_true_pairs')]
    for filename, stats in report['files'].items():
        if stats['invalid_id_prefix'] or stats['blank_or_whitespace_fields']['entity_id']:
            errors.append(f'{filename}: invalid or blank IDs')
    if truth['duplicate_ids_within_list_rows'] or truth['invalid_match_tokens']:
        errors.append('Invalid ground-truth ID lists')
    report['validation_errors'] = errors
    report['status'] = 'FAIL' if errors else 'PASS'
    report['elapsed_seconds'] = time.monotonic() - started
    report['resource_root'] = str(args.resource_dir.resolve())
    report['resource_files'] = [str(p.relative_to(args.resource_dir))
                                for p in sorted(args.resource_dir.rglob('*')) if p.is_file()]
    db.close()
    (args.report_dir / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'Ground truth match-count distribution: {truth["match_count"]["histogram"]}')
    log(f'Ground truth source types: {truth["match_source_types"]}')
    log(f'ALL-EMPTY BASELINE macro F_0.5 = {singletons / n:.10f} ({singletons:,}/{n:,})')
    log(f'Per-country baselines: {report["ground_truth_by_country"]}')
    log(f'STAGE 0 {report["status"]}: {args.report_dir / "audit.json"}')
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--resource-dir', type=Path, default=Path('student_resource'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports/stage0'))
    parser.add_argument('--pdf-tools-directory', type=Path)
    parser.add_argument('--chunk-size', type=int, default=100000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error('--chunk-size must be positive')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
