"""Metric boundaries, strict TSV coverage, reproducible groups, country isolation."""
from fractions import Fraction
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scoring import MacroMetrics, entity_f05, f05_counts, parse_ids
from score_predictions import evaluate
from splits import assign_grouped, entity_hash, iter_entities, make_modes, selection, validation_size


class ScoreTests(unittest.TestCase):
    def test_singletons_and_empty_predictions(self):
        self.assertEqual(entity_f05(set(), set()), 1)
        self.assertEqual(entity_f05(set(), {'S2-a'}), 0)
        self.assertEqual(entity_f05({'S2-a'}, set()), 0)
        self.assertEqual(entity_f05({'S2-a'}, {'S3-b'}), 0)

    def test_perfect_multi_match_and_pdf_example(self):
        truth = {'S2-a', 'S3-b'}
        self.assertEqual(entity_f05(truth, truth), 1)
        self.assertAlmostEqual(entity_f05(truth, truth | {'S2-c'}), 5 / 7)

    def test_formula_against_rational_precision_recall(self):
        for tp in range(5):
            for fp in range(5):
                for fn in range(5):
                    if tp + fn == 0:
                        expected = int(fp == 0)
                    elif tp == 0:
                        expected = 0
                    else:
                        p, r = Fraction(tp, tp + fp), Fraction(tp, tp + fn)
                        expected = Fraction(5, 4) * p * r / (Fraction(1, 4) * p + r)
                    self.assertAlmostEqual(f05_counts(tp, fp, fn), float(expected), places=15)

    def test_false_merge_costs_more_than_one_missed_match(self):
        self.assertLess(f05_counts(2, 1, 0), f05_counts(2, 0, 1))

    def test_macro_does_not_weight_large_match_sets(self):
        metric = MacroMetrics()
        metric.add({'S2-small'}, {'S2-small'})
        large = {f'S3-{i}' for i in range(100)}
        metric.add(large, {'S3-0'})
        self.assertAlmostEqual(metric.result()['macro_f05'], (1 + 5 / 104) / 2)
        self.assertNotAlmostEqual(metric.result()['macro_f05'], 10 / 109)

    def test_baseline_and_false_singleton_denominators(self):
        metric = MacroMetrics()
        for truth in (set(), {'S2-a'}, {'S2-b', 'S3-c'}):
            metric.add(truth, set())
        self.assertEqual(metric.result()['macro_f05'], 1 / 3)
        self.assertEqual(metric.result()['false_singleton_rate'], 0)
        metric = MacroMetrics()
        metric.add(set(), {'S2-a'})
        metric.add({'S2-b'}, {'S2-b'})
        self.assertEqual(metric.result()['false_singleton_rate'], 1)

    def test_no_population_or_invalid_counts_fail(self):
        with self.assertRaises(ValueError):
            MacroMetrics().result()
        for counts in ((-1, 0, 0), (1.5, 0, 0), (True, 0, 0)):
            with self.assertRaises(ValueError):
                f05_counts(*counts)

    def test_invalid_id_lists_fail(self):
        for text in ('S2-a,S2-a', 'S1-a', 'S4-a', 'S2-a,', ' S2-a', 'null', ' ', 'S2-'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ids(text)


def fixture_db(path=':memory:', seed=42, reverse=False):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE entities (id TEXT PRIMARY KEY, country TEXT, matched_ids TEXT, '
               'match_count INTEGER, split_hash TEXT, grouped_role TEXT)')
    rows = [(f'S1-{i:03}', country, '' if i % 2 == 0 else f'S2-{i},S3-{i}',
             0 if i % 2 == 0 else 2, entity_hash(f'S1-{i:03}', seed), 'train')
            for i, country in enumerate(['Country A'] * 40 + ['France'] * 40)]
    db.executemany('INSERT INTO entities VALUES (?,?,?,?,?,?)', reversed(rows) if reverse else rows)
    db.commit()
    return db


class SplitTests(unittest.TestCase):
    def test_order_independence_and_seed_sensitivity(self):
        assignments = []
        for reverse, seed in ((False, 42), (True, 42), (False, 99)):
            with closing(fixture_db(seed=seed, reverse=reverse)) as db:
                assign_grouped(db, .2)
                assignments.append(db.execute('SELECT id,grouped_role FROM entities ORDER BY id').fetchall())
        self.assertEqual(assignments[0], assignments[1])
        self.assertNotEqual(assignments[0], assignments[2])

    def test_group_disjointness_full_positive_lists_and_strata(self):
        with closing(fixture_db()) as db:
            strata = assign_grouped(db, .2)
            modes = make_modes(['Country A', 'France'])
            train = list(iter_entities(db, modes))
            val = list(iter_entities(db, modes, role='validation'))
            self.assertEqual((len(train), len(val)), (64, 16))
            self.assertFalse({r[0] for r in train} & {r[0] for r in val})
            self.assertEqual(len({r[0] for r in train + val}), 80)
            train_targets = set().union(*(parse_ids(r[2]) for r in train))
            val_targets = set().union(*(parse_ids(r[2]) for r in val))
            self.assertFalse(train_targets & val_targets)
            self.assertTrue(all(len(parse_ids(r[2])) in (0, 2) for r in train + val))
            self.assertTrue(all(s['validation_entities'] == 4 for s in strata))

    def test_transfer_uses_open_country_labels(self):
        with closing(fixture_db()) as db:
            modes = make_modes(['France', 'Country A', 'Unseen'])
            self.assertEqual(len(modes), 7)
            for mode in modes.values():
                if mode['type'] != 'country_transfer':
                    continue
                for role in ('train', 'validation'):
                    predicate, args = selection(mode, role)
                    countries = {r[0] for r in db.execute('SELECT country FROM entities e WHERE ' + predicate, args)}
                    self.assertLessEqual(countries, {mode[f'{role}_country']})
                self.assertNotEqual(mode['train_country'], mode['validation_country'])

    def test_tiny_strata_and_invalid_fractions(self):
        self.assertEqual(validation_size(1, .2), 0)
        self.assertEqual(validation_size(2, .2), 1)
        for fraction in (0, 1, -1, float('nan')):
            with self.assertRaises(ValueError):
                validation_size(10, fraction)

    def test_reassignment_does_not_retain_previous_validation_rows(self):
        with closing(fixture_db()) as db:
            assign_grouped(db, .5)
            assign_grouped(db, .2)
            count = db.execute("SELECT COUNT(*) FROM entities WHERE grouped_role='validation'").fetchone()[0]
            self.assertEqual(count, 16)


class FileScorerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with closing(fixture_db(self.root / 'splits.sqlite')) as db:
            assign_grouped(db, .2)
            self.modes = make_modes(['Country A', 'France'])
            self.validation = list(iter_entities(db, self.modes, role='validation'))
        (self.root / 'manifest.json').write_text(json.dumps({'modes': self.modes}), encoding='utf-8')

    def write_predictions(self, rows):
        path = self.root / 'predictions.tsv'
        path.write_text('source1_entity_id\tmatched_entity_ids\n' +
                        ''.join(f'{entity_id}\t{text}\n' for entity_id, text in rows), encoding='utf-8')
        return path

    def test_unordered_perfect_predictions_and_baseline(self):
        path = self.write_predictions([(row[0], row[2]) for row in reversed(self.validation)])
        self.assertEqual(evaluate(self.root, predictions=path)['metrics']['macro_f05'], 1)
        self.assertEqual(evaluate(self.root, baseline='empty')['metrics']['macro_f05'], .5)

    def test_missing_row_is_not_implicitly_empty(self):
        path = self.write_predictions([(row[0], '') for row in self.validation[1:]])
        with self.assertRaisesRegex(ValueError, '1 missing'):
            evaluate(self.root, predictions=path)

    def test_extra_and_duplicate_rows_rejected(self):
        rows = [(row[0], '') for row in self.validation]
        path = self.write_predictions(rows + [('S1-unknown', '')])
        with self.assertRaisesRegex(ValueError, '1 extra'):
            evaluate(self.root, predictions=path)
        path = self.write_predictions(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate S1'):
            evaluate(self.root, predictions=path)

    def test_malformed_columns_rejected(self):
        path = self.root / 'bad.tsv'
        path.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'expected 2 columns'):
            evaluate(self.root, predictions=path)

    def test_country_transfer_file_score(self):
        with closing(sqlite3.connect(self.root / 'splits.sqlite')) as db:
            rows = list(iter_entities(db, self.modes, mode='transfer_01', role='validation'))
        path = self.write_predictions([(row[0], row[2]) for row in rows])
        result = evaluate(self.root, predictions=path, mode='transfer_01')
        self.assertEqual(result['metrics']['entities'], 40)
        self.assertEqual(result['metrics']['macro_f05'], 1)
        self.assertEqual(set(result['per_country']), {'France'})


if __name__ == '__main__':
    unittest.main()
