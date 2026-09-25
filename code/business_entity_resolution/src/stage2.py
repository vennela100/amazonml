"""Normalize all source rows, then verify every written raw record by digest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time

from normalization import (EVIDENCE_COLUMNS, FLAG_COLUMNS, OUTPUT_COLUMNS, RAW_COLUMNS,
                           TEXT_COLUMNS, VERSION, normalize_record)
from scoring import read_tsv
from stage1 import fingerprint

LIST_COLUMNS = {'address_numbers', 'house_numbers', 'house_number_candidates',
                'street_numbers', 'postal_codes', 'postal_candidates'}
TRACKED = ['name_missing', 'address_missing', 'name_null_like', 'address_null_like',
           'name_normalized_empty', 'address_normalized_empty', 'house_number_missing',
           'house_number_ambiguous', 'postal_code_missing', 'postal_code_ambiguous']


def log(message):
    print(message, flush=True)


def raw_bytes(row):
    return (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def verify_output(path, expected_rows, expected_raw_digest):
    digest = hashlib.sha256()
    rows = 0
    flag_offset = len(RAW_COLUMNS) + len(TEXT_COLUMNS) + len(EVIDENCE_COLUMNS)
    for row in read_tsv(path, OUTPUT_COLUMNS):
        rows += 1
        digest.update(raw_bytes(row[:4]))
        if any(flag not in ('0', '1') for flag in row[flag_offset:]):
            raise ValueError(f'{path}: invalid missingness/ambiguity flag at row {rows}')
    if rows != expected_rows or digest.hexdigest() != expected_raw_digest:
        raise AssertionError(f'{path}: serialized raw fields, order, or row counts changed')
    return {'rows': rows, 'raw_sha256': digest.hexdigest(),
            'exact_raw_roundtrip': True, 'all_flag_values_valid': True}


def process_file(task):
    source = Path(task['source'])
    destination = Path(task['destination'])
    started = time.monotonic()
    info = fingerprint(source)
    if (info['bytes'], info['crc32']) != (task['inventory']['bytes'], task['inventory']['crc32']):
        raise ValueError(f'{source} differs from the audited archive')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.tsv')
    if destination.exists() or partial.exists():
        raise FileExistsError(f'Output already exists for {source.name}; choose a fresh artifacts directory')
    raw_digest = hashlib.sha256()
    countries, flags, changes = Counter(), Counter(), Counter()
    house_sources, postal_sources = Counter(), Counter()
    slices = defaultdict(Counter)
    examples = defaultdict(list)
    rows = 0
    with partial.open('w', encoding='utf-8', newline='', buffering=1024 * 1024) as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(OUTPUT_COLUMNS)
        for row in read_tsv(source, RAW_COLUMNS):
            result = normalize_record(row)
            raw_digest.update(raw_bytes(row))
            country = row[3]
            rows += 1
            countries[country] += 1
            slices[country]['rows'] += 1
            for flag in TRACKED:
                flags[flag] += result[flag]
                slices[country][flag] += result[flag]
            house_sources[result['house_number_source'] or 'none'] += 1
            postal_sources[result['postal_code_source'] or 'none'] += 1
            changes['name_changed'] += int(result['name_normalized'] != row[1])
            changes['address_changed'] += int(result['address_normalized'] != row[2])
            changes['legal_suffix_changed_clean_view'] += int(result['name_normalized'] != result['name_clean'])
            for key in ('name_null_like', 'house_number_ambiguous', 'postal_code_ambiguous'):
                if result[key] and len(examples[key]) < 3:
                    examples[key].append({k: result[k] for k in RAW_COLUMNS +
                                         ['name_normalized', 'house_numbers', 'house_number_candidates',
                                          'house_number_source', 'postal_codes', 'postal_candidates',
                                          'postal_code_source']})
            values = [json.dumps(result[column], ensure_ascii=False, separators=(',', ':'))
                      if column in LIST_COLUMNS else result[column] for column in OUTPUT_COLUMNS]
            writer.writerow(values)
            if rows % 500000 == 0:
                log(f'{source.name}: normalized {rows:,} rows')
    audit = task['audit']
    if rows != audit['rows'] or dict(countries) != audit['country_counts']:
        raise AssertionError(f'{source}: row or country counts differ from Stage 0')
    for column, flag in [('business_name', 'name_missing'), ('business_address', 'address_missing')]:
        if flags[flag] != audit['blank_or_whitespace_fields'][column]:
            raise AssertionError(f'{source}: missingness changed for {column}')
    log(f'{source.name}: re-reading all {rows:,} output rows to verify raw preservation...')
    verified = verify_output(partial, rows, raw_digest.hexdigest())
    partial.rename(destination)
    result = {'rows': rows, 'country_counts': dict(countries), 'flags': dict(flags),
              'changes': dict(changes), 'house_number_sources': dict(house_sources),
              'postal_code_sources': dict(postal_sources), 'per_country': dict(slices),
              'examples': dict(examples), 'input_fingerprint': info,
              'output_path': str(destination.resolve()), 'output_bytes': destination.stat().st_size,
              'validation': verified, 'digit_preservation_checked_rows': rows,
              'elapsed_seconds': time.monotonic() - started}
    log(f'{source.name}: PASS; house evidence={rows - flags["house_number_missing"]:,}, '
        f'postal evidence={rows - flags["postal_code_missing"]:,}')
    return source.name, result


def render_report(report, path):
    lines = ['# Stage 2: normalization and structured evidence', '',
             f'**PASS**. Processed and re-read all {report["total_rows"]:,} source records. '
             'No model, candidate generator, or submission was created.', '',
             '## Text policy', '',
             '- Original entity ID, business name, address, and country remain byte-for-byte '
             'equivalent as UTF-8 field values, in the same row order. TSV quoting may change.',
             '- Unicode NFKC, case folding, whitespace cleaning, punctuation/symbol spacing, '
             'and `&` → `and`. Accents and non-Latin combining marks are retained.',
             '- A separate basic name view retains lexical suffixes. The legal-normalized view '
             'expands recognized trailing suffix tokens such as Ltd/Limited, Pvt/Private, '
             'Corp/Corporation, and Inc/Incorporated. It never removes suffixes or interior words.',
             '- Decimal digits, including zero padding, survive in both primary normalized views. '
             'Unicode decimal digits become ASCII; raw originals remain intact.', '',
             '## Numeric evidence and missingness', '',
             '- House/premise numbers retain fractions, ranges, slash numbering, and letter suffixes. '
             'Explicit house/door/plot/building labels and weaker leading-number heuristics have '
             'separate provenance values. These are extracted hypotheses, not verified addresses.',
             '- Street ordinals and explicitly numbered streets are stored separately.',
             '- Numeric postcode shapes supported: 5 digits, 6 digits, and 5+4 digits. '
             'Explicit postcode labels and weaker trailing-number heuristics have separate provenance. '
             'No country whitelist or country-specific lookup is used. Unsupported alphanumeric '
             'postcodes remain in raw/normalized text and are not claimed as extracted codes.',
             '- Ambiguous numbers remain in candidate fields. Confirmed positional house-number '
             'occurrences are not also used as postcode occurrences.',
             '- Missing extraction means no accepted value was found, not that the real business '
             'has no number/code. Empty lists, explicit missingness flags, ambiguity flags, and '
             'provenance must be used together; absence is never a numeric conflict.',
             '- Whole-field null-like strings are diagnostic flags. A name such as `NA` is preserved '
             'and is not silently replaced with a missing value.', '',
             '## Full-data results', '',
             '| File | Rows | Blank addresses | House evidence | Postal evidence | Ambiguous postal |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in sorted(report['files'].items()):
        flags, n = result['flags'], result['rows']
        lines.append(f'| {name} | {n:,} | {flags["address_missing"]:,} | '
                     f'{n - flags["house_number_missing"]:,} | '
                     f'{n - flags["postal_code_missing"]:,} | {flags["postal_code_ambiguous"]:,} |')
    lines += ['', 'Evidence counts measure extraction coverage, not extraction accuracy. '
              'No gold structured-address annotations are supplied.', '',
              '| File | Country | Rows | House evidence | Postal evidence |',
              '|---|---|---:|---:|---:|---:|']
    for name, result in sorted(report['files'].items()):
        for country, stats in sorted(result['per_country'].items()):
            n = stats['rows']
            lines.append(f'| {name} | {country} | {n:,} | {n - stats["house_number_missing"]:,} | '
                         f'{n - stats["postal_code_missing"]:,} |')
    lines += ['', '## Validation and reproducibility', '',
              'All input sizes and CRCs match Stage 0. Every output was re-read to verify its '
              f'{len(OUTPUT_COLUMNS)}-column schema, flags, row count, and SHA256 digest of the original four fields '
              'in order. Row counts, country counts, and raw missingness match Stage 0. '
              'Name/address digit preservation was asserted on every record before writing.', '',
              '`report.json` contains provenance counts, per-country coverage, deterministic '
              'examples of ambiguous evidence, source fingerprints, and round-trip digests. '
              'The transformed TSV paths are recorded there. No labeled pairs were used to '
              'select normalization rules, and no model score is reported in this stage.', '',
              '## Review gate', '',
              'Stage 3 is pending approval. Next, build the retrieval routes and measure per-S1 '
              'candidate recall. If recall is weak, inspect misses by script, country, missing '
              'address, and numeric ambiguity. Preserve raw/basic views as alternative retrieval '
              'inputs; do not compensate by accepting weak matches.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError('Choose an empty/fresh --report-dir')
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        raise FileExistsError('Choose an empty/fresh --artifact-dir')
    audit = json.loads((args.audit_dir / 'audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS':
        raise ValueError('Passing Stage 0 required')
    inventory = {item['path']: item for item in audit['archive_inventory']}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for split in ('train', 'test'):
        for source in (1, 2, 3):
            name = f'{split}_source{source}.tsv'
            tasks.append({'source': str(args.dataset_dir / split / name),
                          'destination': str(args.artifact_dir / split / name),
                          'audit': audit['files'][name],
                          'inventory': inventory[f'student_resource/dataset/{split}/{name}']})
    tasks.sort(key=lambda task: task['audit']['rows'], reverse=True)
    started = time.monotonic()
    files = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = [pool.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            name, result = future.result()
            files[name] = result
            (args.report_dir / f'{name}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                       encoding='utf-8')
    report = {'stage': 2, 'status': 'PASS', 'normalization_version': VERSION,
              'total_rows': sum(result['rows'] for result in files.values()),
              'schema': OUTPUT_COLUMNS, 'workers': args.workers,
              'files': files, 'elapsed_seconds': time.monotonic() - started}
    (args.report_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    render_report(report, args.report_dir / 'report.md')
    log(f'STAGE 2 PASS: {report["total_rows"]:,} rows; {args.report_dir / "report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path('student_resource/dataset'))
    parser.add_argument('--audit-dir', type=Path, default=Path('reports/stage0'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports/stage2'))
    parser.add_argument('--artifact-dir', type=Path, default=Path('artifacts/stage2'))
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
