"""Constructed fixtures test normalization rules; they never augment the dataset."""
import unittest
import csv
import json
from pathlib import Path
import tempfile

from normalization import normalize_name, normalize_record, numeric_evidence, prepare
from normalization import RAW_COLUMNS, OUTPUT_COLUMNS
from scoring import read_tsv
from stage1 import fingerprint
from stage2 import process_file, verify_output


def record(name='Acme', address='', country='Unseen'):
    return normalize_record(['S1-test', name, address, country])


class NormalizationTests(unittest.TestCase):
    def test_legal_suffixes_and_ampersand(self):
        self.assertEqual(normalize_name(' ACME & Sons PVT. LTD. ')[1], 'acme and sons private limited')
        self.assertEqual(normalize_name('Acme and Sons Private Limited')[1], 'acme and sons private limited')
        self.assertEqual(normalize_name('Acme Corp.')[1], normalize_name('Acme Corporation')[1])
        self.assertEqual(normalize_name('Acme L.L.C.')[1], 'acme llc')

    def test_distinctive_interior_words_remain(self):
        self.assertEqual(normalize_name('Private Eye Ltd')[1], 'private eye limited')
        self.assertEqual(normalize_name('Corp Music Studio')[1], 'corp music studio')

    def test_unicode_scripts_marks_and_accents_survive(self):
        self.assertEqual(normalize_name('ÉCOLE')[1], normalize_name('E\u0301cole')[1])
        name = 'राम मार्केटिंग प्राइवेट लिमिटेड'
        self.assertEqual(record(name)['name_normalized'], name)
        self.assertIn('é', record('École')['name_normalized'])
        self.assertEqual(record('ＡＣＭＥ １２')['name_normalized'], 'acme 12')
        self.assertEqual(record('Shop १२')['name_normalized'], 'shop 12')

    def test_raw_fields_and_country_are_unchanged(self):
        row = ['S1-test', '  ACME\tLtd ', '12\n Main St.', 'Anything at all']
        result = normalize_record(row)
        self.assertEqual([result[k] for k in ('entity_id','business_name','business_address','country')], row)

    def test_branches_digits_and_zero_padding_are_not_removed(self):
        self.assertNotEqual(record('Shop 01')['name_normalized'], record('Shop 02')['name_normalized'])
        result = record(address='House No. 0012/3, 42nd Street, ZIP 01234')
        self.assertEqual(result['house_numbers'], ['0012/3'])
        self.assertEqual(result['street_numbers'], ['42'])
        self.assertEqual(result['postal_codes'], ['01234'])

    def test_house_ranges_fractions_and_letters(self):
        for address, expected in [('12–14 Main Road', '12-14'), ('19 1/2 Main St', '19 1/2'),
                                  ('12A Main Road', '12a'), ('Plot A-68, Delhi', 'a-68')]:
            with self.subTest(address=address):
                self.assertIn(expected, record(address=address)['house_numbers'])

    def test_country_independent_numeric_postcodes(self):
        for address, code in [('12 Main St, 90210-1234', '90210-1234'),
                              ('12 Rue des Fleurs, 75001', '75001'),
                              ('12 Main Road, PIN 560001', '560001')]:
            a = record(address=address, country='France')
            b = record(address=address, country='NeverSeen')
            self.assertEqual(a['postal_codes'], [code])
            self.assertEqual(a['postal_codes'], b['postal_codes'])

    def test_missing_is_separate_from_present_and_ambiguous(self):
        empty, present = record(address=''), record(address='12 Main St')
        self.assertEqual(empty['address_missing'], 1)
        self.assertEqual(empty['house_number_missing'], 1)
        self.assertEqual(empty['house_numbers'], [])
        self.assertEqual(present['address_missing'], 0)
        self.assertEqual(present['house_number_missing'], 0)
        ambiguous = record(address='75001 Paris')
        self.assertEqual(ambiguous['house_numbers'], [])
        self.assertEqual(ambiguous['postal_codes'], [])
        self.assertEqual(ambiguous['postal_candidates'], ['75001'])
        self.assertEqual(ambiguous['postal_code_ambiguous'], 1)

    def test_long_house_number_not_misclassified_as_postcode(self):
        result = record(address='12345 Main Street')
        self.assertEqual(result['house_numbers'], ['12345'])
        self.assertEqual(result['postal_codes'], [])
        self.assertEqual(result['postal_candidates'], [])
        self.assertEqual(record(address='Main Street, House No 12345')['postal_codes'], [])
        self.assertEqual(record(address='90210-1234 Los Angeles')['house_numbers'], [])
        self.assertEqual(record(address='12345 Main Street, 12345')['postal_codes'], ['12345'])

    def test_null_like_is_diagnostic_not_silent_missing(self):
        result = record('NA', 'null')
        self.assertEqual(result['name_null_like'], 1)
        self.assertEqual(result['name_missing'], 0)
        self.assertEqual(result['business_name'], 'NA')
        self.assertEqual(result['address_null_like'], 1)
        self.assertEqual(result['address_missing'], 0)

    def test_punctuation_only_empty_view_is_flagged(self):
        result = record('---', '...')
        self.assertEqual(result['name_missing'], 0)
        self.assertEqual(result['name_normalized_empty'], 1)
        self.assertEqual(result['address_normalized_empty'], 1)

    def test_no_default_numeric_values_or_bare_number_claims(self):
        for text in ('', 'Near the central market', '75001'):
            result = record(address=text)
            self.assertEqual(result['house_numbers'], [])
            self.assertEqual(result['postal_codes'], [])

    def test_multiple_postcodes_are_flagged_ambiguous(self):
        result = record(address='ZIP 01234, postal code 56789')
        self.assertEqual(result['postal_codes'], ['01234','56789'])
        self.assertEqual(result['postal_code_ambiguous'], 1)

    def test_normalization_idempotent(self):
        for name in ('ACME & Sons PVT. LTD.', 'École', 'राम मार्केटिंग', 'Shop 01'):
            once = normalize_name(name)[1]
            self.assertEqual(normalize_name(once)[1], once)


class PipelineTests(unittest.TestCase):
    def test_roundtrip_preserves_embedded_tabs_quotes_and_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'train_source1.tsv', root / 'out.tsv'
            rows = [['S1-a', ' A\t"Shop" Ltd ', '12 Main\nStreet, ZIP 01234', 'Unseen'],
                    ['S1-b', 'राम मार्केटिंग', '', 'France']]
            with source.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
                writer.writerow(RAW_COLUMNS)
                writer.writerows(rows)
            task = {'source': str(source), 'destination': str(target), 'inventory': fingerprint(source),
                    'audit': {'rows': 2, 'country_counts': {'Unseen': 1, 'France': 1},
                              'blank_or_whitespace_fields': {'business_name': 0, 'business_address': 1}}}
            _, result = process_file(task)
            written = list(read_tsv(target, OUTPUT_COLUMNS))
            self.assertEqual([row[:4] for row in written], rows)
            self.assertEqual(json.loads(written[0][OUTPUT_COLUMNS.index('postal_codes')]), ['01234'])
            self.assertTrue(result['validation']['exact_raw_roundtrip'])
            with target.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
                writer.writerow(OUTPUT_COLUMNS)
                written[0][1] = 'Changed raw name'
                writer.writerows(written)
            with self.assertRaisesRegex(AssertionError, 'raw fields'):
                verify_output(target, 2, result['validation']['raw_sha256'])
            task['inventory']['crc32'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'differs from the audited archive'):
                process_file(task)


if __name__ == '__main__':
    unittest.main()
