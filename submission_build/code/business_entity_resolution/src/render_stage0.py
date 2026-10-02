"""Render a readable report from the completed audit, without rereading data."""
import argparse
import json
from pathlib import Path


def render(report_dir):
    report = json.loads((report_dir / 'audit.json').read_text(encoding='utf-8'))
    lines = ['# Stage 0: full-data audit', '',
             f'Status: **{report["status"]}**. All seven TSVs were read in full; no sampling.', '',
             '## Inputs and task confirmation', '',
             f'- Resource root: `{report["resource_root"]}`.',
             f'- PDF pages read: {report["pdf"]["pages"]}; the last page is blank.',
             '- Each S1 reference entity can match zero, one, or multiple S2/S3 records.',
             '- Only names, addresses, country labels, and provided labels are used; no external data.',
             '- Country is an open-label string; France must be included at inference.',
             '- Optimize exact per-S1 macro F_0.5, including singleton scores of 1 for empty '
             'predictions and 0 for nonempty predictions.',
             '- Final model must have MIT/Apache-2.0 licensing and at most 8 billion parameters.',
             '- Both output TSVs require every test S1 exactly once, unique valid S2/S3 IDs only, '
             'and final matches must be a subset of the exact candidate set scored by the model.',
             '- Only `matching_results.tsv` is leaderboard-scored; final rankings use the private split.', '',
             '### Corrections and additions from supplied materials', '',
             '1. The PDF also requires the filled `Documentation_template.md` in the final ZIP.',
             '2. The PDF says precision is weighted "2x", but the supplied formula is '
             '`1.25 TP / (1.25 TP + FP + 0.25 FN)`: FP has four times the FN denominator '
             'coefficient. Use the exact formula.',
             '3. The validator skips target-ID existence by default; use `--check-ids`. '
             'Missing candidate files and match/candidate subset violations produce only warnings. '
             'Our final checks must enforce these requirements strictly.',
             '4. The validator documentation says nonexistent IDs only lower scores, whereas the PDF '
             'says they cause rejection. Follow the stricter PDF requirement.', '',
             '## Actual extracted files', '', '```text', *report['resource_files'], '```', '',
             'The original archive additionally contains macOS metadata; its complete inventory '
             'is in `archive_inventory.json`. Extracted members passed ZIP CRC checks.', '',
             '## Source audit', '',
             '| File | Rows | Blank names | Blank addresses | Name length min / p50 / p95 / max |',
             '|---|---:|---:|---:|---|']
    for name, stats in report['files'].items():
        length = stats['name_length_characters']
        blanks = stats['blank_or_whitespace_fields']
        lines.append(f'| {name} | {stats["rows"]:,} | {blanks["business_name"]:,} | '
                     f'{blanks["business_address"]:,} | {length["min"]} / {length["p50"]:g} / '
                     f'{length["p95"]:g} / {length["max"]} |')
    lines += ['', 'Name lengths count original Unicode code points. Exact histograms, means, '
              'and additional quantiles are in `audit.json`.', '',
              '| File | Country | Records | Blank addresses |', '|---|---|---:|---:|']
    for name, stats in report['files'].items():
        for country, count in sorted(stats['country_counts'].items()):
            lines.append(f'| {name} | {country} | {count:,} | '
                         f'{stats["blank_address_by_country"].get(country, 0):,} |')
    lines += ['', '### Missing-value and schema checks', '',
              'All source files have exactly four columns; ground truth has exactly two. '
              'Strict CSV and pandas row counts agree. Blank/whitespace fields are counted '
              'separately from whole-field null-like strings. No automatic NA coercion is used.', '',
              '| File | Blank fields by column | Literal null-like fields by column |',
              '|---|---|---|']
    for name, stats in report['files'].items():
        blank = {k: v for k, v in stats['blank_or_whitespace_fields'].items() if v}
        literal = {k: v for k, v in stats['literal_null_like_fields'].items() if v}
        lines.append(f'| {name} | {blank or "none"} | {literal or "none"} |')
    baseline = report['all_empty_baseline']
    truth = report['ground_truth']
    lines += ['', '## Ground truth and sanity baseline', '',
              f'**All-empty macro F_0.5 = {baseline["macro_f05"]:.10f}** '
              f'({baseline["singletons"]:,} singletons / {baseline["entities"]:,} S1 entities).', '',
              'This is an analytic baseline; the general scorer belongs to Stage 1. '
              'An independent standard-library CSV pass confirmed the same singleton count, '
              'source-type counts, and full match-count histogram.', '',
              '| Match source type | S1 entities | Share |', '|---|---:|---:|']
    for kind, count in sorted(truth['match_source_types'].items()):
        lines.append(f'| {kind} | {count:,} | {count / baseline["entities"]:.4%} |')
    lines += ['', '| True matches per S1 | S1 entities |', '|---:|---:|']
    for count, entities in truth['match_count']['histogram'].items():
        lines.append(f'| {count} | {entities:,} |')
    lines += ['', f'Mean matches per S1: {truth["match_count"]["mean"]:.6f}; '
              f'median: {truth["match_count"]["p50"]:g}.', '',
              '| Country | S1 entities | Singletons | All-empty macro F_0.5 |', '|---|---:|---:|---:|']
    for country, stats in sorted(report['ground_truth_by_country'].items()):
        lines.append(f'| {country} | {stats["entities"]:,} | {stats["singletons"]:,} | '
                     f'{stats["all_empty_macro_f05"]:.10f} |')
    lines += ['', '## Integrity checks', '', '| Check | Count |', '|---|---:|']
    for name, value in report['integrity'].items():
        lines.append(f'| {name} | {value:,} |')
    for name in ('duplicate_ids_within_list_rows', 'invalid_match_tokens'):
        lines.append(f'| {name} | {truth[name]:,} |')
    lines += ['', 'Structural checks cannot establish semantic correctness of supplied labels.', '',
              '## Review gate and next step', '',
              'Stage 0 is complete. Stage 1 has not started. Next, implement and test the exact '
              'macro scorer and entity-grouped/country-transfer validation. If later scores are weak, '
              'first separate candidate misses from false merges and compare missing-address and '
              'country slices. Multiple true matches must remain possible; missing addresses must '
              'be distinct from address conflict.', '',
              'Four focused audit tests passed. Reproduction commands and pinned dependencies are '
              'in `code/business_entity_resolution/README.md`.', '']
    destination = report_dir / 'report.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/stage0'))
    render(parser.parse_args().report_dir)
