"""Score complete validation TSVs in arbitrary row order, with bounded memory."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile

from scoring import MATCH_HEADER, MacroMetrics, parse_ids, read_tsv, validate_s1
from splits import selection
from stage1 import insert_batches


def evaluate(split_dir, predictions=None, baseline=None, mode='grouped', role='validation'):
    if (predictions is None) == (baseline is None):
        raise ValueError('Provide exactly one of predictions or baseline')
    if baseline not in (None, 'empty', 'perfect'):
        raise ValueError(f'Unknown baseline: {baseline}')
    manifest = json.loads((split_dir / 'manifest.json').read_text(encoding='utf-8'))
    if mode == 'all':
        predicate, parameters = '1', ()
    else:
        if mode not in manifest['modes']:
            raise ValueError(f'Unknown mode {mode!r}; use all or {list(manifest["modes"])}')
        predicate, parameters = selection(manifest['modes'][mode], role)
    metrics, by_country = MacroMetrics(), defaultdict(MacroMetrics)
    with tempfile.TemporaryDirectory(prefix='entity-score-') as directory:
        with closing(sqlite3.connect(str(Path(directory) / 'predictions.sqlite'), uri=True)) as db:
            db.execute('PRAGMA cache_size=-32768')
            db.execute('ATTACH DATABASE ? AS truth',
                       ((split_dir / 'splits.sqlite').resolve().as_uri() + '?mode=ro',))
            if predictions is not None:
                db.execute('CREATE TABLE predictions (id TEXT PRIMARY KEY, matched_ids TEXT NOT NULL) WITHOUT ROWID')

                def rows():
                    for entity_id, text in read_tsv(predictions, MATCH_HEADER):
                        validate_s1(entity_id)
                        parse_ids(text)
                        yield entity_id, text

                try:
                    insert_batches(db, 'INSERT INTO predictions VALUES (?,?)', rows())
                except sqlite3.IntegrityError as exc:
                    raise ValueError('Duplicate S1 prediction rows') from exc
                extra = db.execute('SELECT COUNT(*) FROM predictions p LEFT JOIN truth.entities e '
                                   'ON p.id=e.id AND (' + predicate + ') WHERE e.id IS NULL',
                                   parameters).fetchone()[0]
                missing = db.execute('SELECT COUNT(*) FROM truth.entities e LEFT JOIN predictions p '
                                     'ON p.id=e.id WHERE (' + predicate + ') AND p.id IS NULL',
                                     parameters).fetchone()[0]
                if missing or extra:
                    raise ValueError(f'Prediction coverage error: {missing} missing, {extra} extra S1 rows')
                query = ('SELECT e.country,e.matched_ids,p.matched_ids FROM truth.entities e '
                         'JOIN predictions p ON e.id=p.id WHERE ' + predicate)
            else:
                query = ('SELECT e.country,e.matched_ids,NULL FROM truth.entities e WHERE ' + predicate)
            for country, text, predicted_text in db.execute(query, parameters):
                truth = parse_ids(text)
                prediction = (truth if baseline == 'perfect' else set() if baseline == 'empty'
                              else parse_ids(predicted_text))
                metrics.add(truth, prediction)
                by_country[country].add(truth, prediction)
    return {'mode': mode, 'role': role if mode != 'all' else 'all',
            'baseline': baseline, 'metrics': metrics.result(),
            'per_country': {country: metric.result() for country, metric in sorted(by_country.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, default=Path('reports/stage1'))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions', type=Path)
    inputs.add_argument('--baseline', choices=['empty', 'perfect'])
    parser.add_argument('--mode', default='grouped')
    parser.add_argument('--role', choices=['train', 'validation'], default='validation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.split_dir, args.predictions, args.baseline, args.mode, args.role)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
