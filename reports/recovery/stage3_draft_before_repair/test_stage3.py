"""Unit tests for Stage 3 candidate generation logic."""
import csv
import json
import os
import sqlite3
import tempfile
import unittest
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from stage3 import (
    merge_route_candidates, evaluate_recall, rare_token_blocking,
    tfidf_blocking, NORM_HEADER
)


def make_normalized_tsv(path, rows):
    """Create a minimal normalized TSV for testing."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(NORM_HEADER)
        for row in rows:
            # Pad to 25 columns
            padded = list(row) + [''] * (25 - len(row))
            writer.writerow(padded)


def make_splits_db(path, entities):
    """Create a minimal splits database for testing.
    entities: list of (id, country, matched_ids, match_count, role)
    """
    with closing(sqlite3.connect(str(path))) as db:
        db.execute('''CREATE TABLE entities (
            id TEXT PRIMARY KEY, country TEXT, matched_ids TEXT,
            match_count INTEGER, split_hash TEXT, grouped_role TEXT
        )''')
        for eid, country, matched, count, role in entities:
            db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?)',
                       (eid, country, matched, count, 'hash', role))
        db.commit()


class TestMergeCandidates(unittest.TestCase):
    def test_union_of_routes(self):
        all_cands = {}
        route1 = {0: [(10, 0.8), (20, 0.3)]}
        route2 = {0: [(10, 0.9), (30, 0.5)]}
        merge_route_candidates(all_cands, route1, 'name')
        merge_route_candidates(all_cands, route2, 'address')

        self.assertIn(0, all_cands)
        self.assertEqual(len(all_cands[0]), 3)  # 10, 20, 30
        self.assertAlmostEqual(all_cands[0][10]['score'], 0.9)  # max score
        self.assertEqual(all_cands[0][10]['routes'], ['name', 'address'])
        self.assertEqual(all_cands[0][20]['routes'], ['name'])
        self.assertEqual(all_cands[0][30]['routes'], ['address'])

    def test_empty_merge(self):
        all_cands = {}
        merge_route_candidates(all_cands, {}, 'name')
        self.assertEqual(len(all_cands), 0)

    def test_multiple_s1_entities(self):
        all_cands = {}
        route1 = {0: [(10, 0.8)], 1: [(20, 0.7)]}
        merge_route_candidates(all_cands, route1, 'name')
        self.assertEqual(len(all_cands), 2)
        self.assertIn(0, all_cands)
        self.assertIn(1, all_cands)


class TestTfidfBlocking(unittest.TestCase):
    def test_self_similarity(self):
        """Identical texts should have high cosine similarity."""
        # Use enough texts with shared substrings so min_df=2 retains features
        texts = [
            'abc corporation limited services',
            'abc enterprises private group',
            'qrs consulting group services',
            'mno logistics solutions group',
            'xyz corporation consulting limited',
            'abc logistics private solutions',
        ]
        # Query with a subset
        queries = texts[:3]
        cands = tfidf_blocking(queries, texts, top_k=4, chunk_size=2,
                               route_name='test', max_features=10000, threshold=0.05)
        # Each query should find its own index in the pool
        for i in range(len(queries)):
            self.assertIn(i, cands, f'Query {i} ({queries[i]}) not in candidates')
            pool_indices = [idx for idx, _ in cands[i]]
            self.assertIn(i, pool_indices,
                          f'Self-match not found for query {i}')

    def test_chunk_consistency(self):
        """Results should be the same regardless of chunk size."""
        pool = ['alpha beta', 'gamma delta', 'epsilon zeta',
                'eta theta', 'iota kappa']
        queries = ['alpha beta gamma', 'delta epsilon']

        cands_1 = tfidf_blocking(queries, pool, top_k=3, chunk_size=1,
                                 route_name='t1', max_features=10000, threshold=0.01)
        cands_2 = tfidf_blocking(queries, pool, top_k=3, chunk_size=100,
                                 route_name='t2', max_features=10000, threshold=0.01)

        for i in range(len(queries)):
            indices_1 = sorted([idx for idx, _ in cands_1.get(i, [])])
            indices_2 = sorted([idx for idx, _ in cands_2.get(i, [])])
            self.assertEqual(indices_1, indices_2,
                             f'Chunk size mismatch for query {i}')


class TestRareTokenBlocking(unittest.TestCase):
    def test_basic_matching(self):
        s1_names = ['xylophonics corporation', 'generic company']
        pool_names = ['xylophonics limited', 'another company',
                      'xylophonics enterprises', 'random business']
        cands = rare_token_blocking(s1_names, pool_names, max_df=5)
        # 'xylophonics' is rare (df=2 in pool), should match pool indices 0 and 2
        self.assertIn(0, cands)  # s1_idx=0 has 'xylophonics'
        pool_indices = {idx for idx, _ in cands[0]}
        self.assertIn(0, pool_indices)
        self.assertIn(2, pool_indices)

    def test_common_tokens_excluded(self):
        pool_names = ['company alpha', 'company beta', 'company gamma'] * 50
        s1_names = ['company delta']
        cands = rare_token_blocking(s1_names, pool_names, max_df=5)
        # 'company' appears 150 times, should not be rare
        # 'delta' does not appear in pool
        self.assertEqual(len(cands), 0)


class TestEvaluateRecall(unittest.TestCase):
    def test_perfect_recall(self):
        """All true matches found in candidates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'candidates.sqlite'
            splits_path = Path(tmpdir) / 'splits.sqlite'

            make_splits_db(splits_path, [
                ('S1-1', 'US', 'S2-10,S2-20', 2, 'validation'),
                ('S1-2', 'US', '', 0, 'validation'),  # singleton
            ])

            with closing(sqlite3.connect(str(db_path))) as db:
                db.execute('''CREATE TABLE candidates (
                    s1_id TEXT, candidate_id TEXT, routes TEXT,
                    best_score REAL, source TEXT)''')
                db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?)', [
                    ('S1-1', 'S2-10', 'name', 0.9, 's2'),
                    ('S1-1', 'S2-20', 'name', 0.8, 's2'),
                    ('S1-1', 'S2-30', 'name', 0.3, 's2'),  # extra candidate
                ])
                db.commit()

            result = evaluate_recall(db_path, splits_path, 'validation')
            self.assertEqual(result['total_entities'], 2)
            self.assertAlmostEqual(result['overall_recall'], 1.0)
            self.assertEqual(result['perfect_recall_entities'], 2)
            self.assertEqual(result['zero_recall_entities'], 0)

    def test_partial_recall(self):
        """Only some true matches found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'candidates.sqlite'
            splits_path = Path(tmpdir) / 'splits.sqlite'

            make_splits_db(splits_path, [
                ('S1-1', 'US', 'S2-10,S2-20,S2-30', 3, 'validation'),
            ])

            with closing(sqlite3.connect(str(db_path))) as db:
                db.execute('''CREATE TABLE candidates (
                    s1_id TEXT, candidate_id TEXT, routes TEXT,
                    best_score REAL, source TEXT)''')
                db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?)', [
                    ('S1-1', 'S2-10', 'name', 0.9, 's2'),
                    ('S1-1', 'S2-40', 'name', 0.3, 's2'),
                ])
                db.commit()

            result = evaluate_recall(db_path, splits_path, 'validation')
            self.assertAlmostEqual(result['overall_recall'], 1/3)
            self.assertEqual(result['zero_recall_entities'], 0)
            self.assertEqual(result['perfect_recall_entities'], 0)

    def test_zero_recall_counted(self):
        """Entity with matches but none found should count as zero recall."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'candidates.sqlite'
            splits_path = Path(tmpdir) / 'splits.sqlite'

            make_splits_db(splits_path, [
                ('S1-1', 'US', 'S2-10', 1, 'validation'),
            ])

            with closing(sqlite3.connect(str(db_path))) as db:
                db.execute('''CREATE TABLE candidates (
                    s1_id TEXT, candidate_id TEXT, routes TEXT,
                    best_score REAL, source TEXT)''')
                db.commit()

            result = evaluate_recall(db_path, splits_path, 'validation')
            self.assertEqual(result['zero_recall_entities'], 1)
            self.assertAlmostEqual(result['overall_recall'], 0.0)


if __name__ == '__main__':
    unittest.main()
