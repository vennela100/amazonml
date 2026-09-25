"""Focused checks for audit failure detection and lossless TSV parsing."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

from stage0 import audit_source, audit_truth, check_shape, distribution, SOURCE_COLUMNS


class AuditTests(unittest.TestCase):
    def test_distribution_uses_entities_not_distinct_lengths(self):
        result = distribution({0: 1, 10: 3})
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 7.5)
        self.assertEqual(result['p50'], 10)
        self.assertEqual(result['p25'], 7.5)

    def test_rejects_short_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.tsv'
            path.write_text('\t'.join(SOURCE_COLUMNS) + '\nS1-a\tName\tUS\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'malformed column counts'):
                check_shape(path, SOURCE_COLUMNS)

    def test_preserves_null_literals_unicode_and_open_country_labels(self):
        with tempfile.TemporaryDirectory() as directory, sqlite3.connect(':memory:') as db:
            path = Path(directory) / 'source.tsv'
            path.write_text('\t'.join(SOURCE_COLUMNS) +
                            '\nS1-a\tÉcole\t\tFrance\nS1-b\tNA\tnull\tUnseen\n', encoding='utf-8')
            db.execute('CREATE TABLE train_records (id TEXT, country TEXT, source INTEGER)')
            result = audit_source(path, 'train', 1, db, 1)
            self.assertEqual(result['rows'], 2)
            self.assertEqual(result['blank_or_whitespace_fields']['business_address'], 1)
            self.assertEqual(result['literal_null_like_fields']['business_address'], 1)
            self.assertEqual(result['parser_null_fields']['business_name'], 0)
            self.assertEqual(result['name_length_characters']['histogram'], {2: 1, 5: 1})
            self.assertEqual(result['country_counts'], {'France': 1, 'Unseen': 1})

    def test_truth_flags_duplicates_and_invalid_targets(self):
        with tempfile.TemporaryDirectory() as directory, sqlite3.connect(':memory:') as db:
            path = Path(directory) / 'truth.tsv'
            path.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\t\n'
                            'S1-b\tS2-x,S2-x,S1-b\nS1-c\tS2-y,S3-z\n', encoding='utf-8')
            db.execute('CREATE TABLE truth (id TEXT, match_count INTEGER, kind TEXT)')
            db.execute('CREATE TABLE edges (s1 TEXT, target TEXT)')
            result = audit_truth(path, db, 1)
            self.assertEqual(result['duplicate_ids_within_list_rows'], 1)
            self.assertEqual(result['invalid_match_tokens'], 1)
            self.assertEqual(result['match_source_types']['singleton'], 1)
            self.assertEqual(result['match_source_types']['both'], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM edges').fetchone()[0], 5)


if __name__ == '__main__':
    unittest.main()
