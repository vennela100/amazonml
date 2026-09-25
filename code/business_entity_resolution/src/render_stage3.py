"""Render an existing, completed Stage 3 run for review; no new measurements."""
import argparse
import json
from pathlib import Path


def render(directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    diagnostics=json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    g=report['groups']['overall']
    def pct(value):
        return 'undefined' if value is None else f'{value:.6%}'
    lines=['# Stage 3 candidate-retrieval review','',
           f"Evaluation scope: **{report['scope']}**, {g['entities']:,} selected S1 entities out of "
           f"{report['population_entities']:,} in the requested split. Both candidate pools are complete.",'',
           '| Measurement | Result |','|---|---:|',
           f"| Macro candidate recall (non-singletons) | {pct(g['macro_candidate_recall_non_singletons'])} |",
           f"| Micro candidate recall | {pct(g['micro_candidate_recall'])} |",
           f"| Perfect-classifier macro F0.5 ceiling | {g['oracle_macro_f05_ceiling']:.9f} |",
           f"| All-empty macro F0.5 baseline | {g['all_empty_macro_f05']:.9f} |",
           f"| Non-singletons / singletons | {g['non_singletons']:,} / {g['singletons']:,} |",
           f"| Found / total true pairs | {g['found_matches']:,} / {g['true_matches']:,} |",
           f"| Fully recalled non-singletons | {g['fully_recalled']:,} |",
           f"| Non-singletons with zero true candidates | {g['zero_recalled']:,} |",
           f"| Unique candidate pairs | {g['candidates']:,} |",
           f"| Mean candidates per S1 | {g['mean_candidates']:.3f} |",
           f"| Reduction against complete brute-force pools | {pct(g['reduction_ratio'])} |",'',
           'The F0.5 ceiling assumes a perfect classifier rejects every false candidate. It is not a '
           'trained-model score. True singletons are excluded from recall, whose denominator is zero '
           'for them, and included in the oracle score with value one.','',
           '## Country slices','',
           '| Country | S1 entities | Macro recall | Oracle macro F0.5 |',
           '|---|---:|---:|---:|']
    for key,value in report['groups'].items():
        if key.startswith('country:'):
            lines.append(f"| {key[8:]} | {value['entities']:,} | {pct(value['macro_candidate_recall_non_singletons'])} | "
                         f"{value['oracle_macro_f05_ceiling']:.9f} |")
    lines += ['', '## Retrieval routes','', '| Route | Macro recall |','|---|---:|']
    for route,value in report['route_macro_recall'].items():
        lines.append(f'| {route} | {pct(value)} |')
    lines += ['', 'Routes overlap; their recalls must not be added. Each route retrieves separately '
              'from S2 and S3 before union.','', '## Budget comparison','',
              '| Top-k per route per source | Macro recall | Oracle F0.5 | Mean candidates |',
              '|---|---:|---:|---:|']
    for budget,value in diagnostics['equal_budget_per_route_per_source'].items():
        lines.append(f"| {budget} | {pct(value['macro_recall'])} | {value['oracle_macro_f05']:.9f} | "
                     f"{value['mean_candidates']:.3f} |")
    seconds=sum(report['retrieval_seconds'].values())
    lines += ['', f'Retrieval time recorded for this run: {seconds:.1f} seconds, excluding index/IDF construction.',
              '', '## Method and limits','',
              '- All three training sources were indexed in full. IDF uses only the selected training S1 partition.',
              '- Approximate trigram/token/component shortlists are reranked with actual character-trigram TF-IDF cosine.',
              '- Posting and shortlist caps may lose neighbors before route top-k ranking. This is not exact global nearest-neighbor retrieval.',
              '- Country is never used to exclude candidate records. France remains supported as an open label.',
              '- No classifier, acceptance threshold, or test submission was produced in Stage 3.',
              '- Full-split and country-transfer results must be measured separately; this report does not imply those runs completed.',
              '', '## Artifacts','',
              '- `candidate_pairs.tsv`: exact unique candidate union for every evaluated S1, including empty sets.',
              '- `candidates.sqlite`: query manifest, completed-source flags, and per-route ranks/scores.',
              '- `per_entity_recall.tsv`: recall and oracle ceiling for every selected S1.',
              '- `missed_pairs.tsv`: missing true pairs with normalized names and addresses for inspection.',
              '- `config.json`: frozen configuration, implementation hashes, and input fingerprints.',
              '- `report.json` and `diagnostics.json`: machine-readable measurements.',
              '', '## Next review decision','',
              'Inspect the missed pairs before increasing route budgets. If a true match never entered a shortlist, '
              'expand anchor coverage or posting/shortlist limits; increasing final top-k alone cannot recover it. '
              'Retain precision-first decisions for the later classifier and decision layer. Stage 4 requires review approval.',
              '', '## Reproduce this run','', '```powershell']
    route_args=' '.join(f'--topk-{k} {v}' for k,v in config['retrieval']['budgets'].items())
    lines += [f"python -u code/business_entity_resolution/src/stage3.py --mode {next_name(config['mode'])} "
              f"--role {config['role']} --limit {config['limit']} {route_args} --output-dir {directory.as_posix()}",
              f'python code/business_entity_resolution/src/analyze_stage3.py --run-dir {directory.as_posix()}',
              '```','']
    # config.json remains authoritative for custom paths and non-default shortlist settings.
    lines += ['Use `config.json` for any non-default index paths or shortlist settings.','']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def next_name(mode):
    if mode['type']=='grouped':
        return 'grouped'
    manifest=json.loads(Path('reports/stage1/manifest.json').read_text(encoding='utf-8'))
    return next(key for key,value in manifest['modes'].items() if value==mode)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    render(p.parse_args().run_dir)
