"""Unit tests for the pairwise feature engineering module."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    N_FEATURES, FEATURE_NAMES, compute_features, parse_row,
    _jaccard, _char_ngrams, _tokens, _parse_list
)


# ── Helpers ──────────────────────────────────────────────────────────────────

_MISSING = object()


def make_record(
    entity_id='S1-1',
    name_norm='acme corporation',
    addr_norm='123 main street new york',
    house_numbers=_MISSING,
    postal_codes=_MISSING,
    house_number_missing=0,
    house_number_ambiguous=0,
    postal_code_missing=0,
    postal_code_ambiguous=0,
    address_missing=0,
    name_missing=0,
):
    hn = ['123'] if house_numbers is _MISSING else (house_numbers or [])
    pc = [] if postal_codes is _MISSING else (postal_codes or [])
    return {
        'entity_id': entity_id,
        'business_name': '',
        'business_address': '',
        'country': 'US',
        'name_clean': name_norm,
        'name_norm': name_norm,
        'addr_norm': addr_norm,
        'house_numbers': set(hn),
        'house_candidates': set(hn),
        'postal_codes': set(pc),
        'postal_candidates': set(pc),
        'name_missing': name_missing,
        'address_missing': address_missing,
        'name_null_like': 0,
        'address_null_like': 0,
        'name_normalized_empty': int(not name_norm),
        'address_normalized_empty': int(not addr_norm),
        'house_number_missing': house_number_missing,
        'house_number_ambiguous': house_number_ambiguous,
        'postal_code_missing': postal_code_missing,
        'postal_code_ambiguous': postal_code_ambiguous,
    }


def feats(s1, cand, routes='name', score=0.8, source=None, metadata=None, rare=None):
    return compute_features(s1, cand, routes, score, source, metadata, rare)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureCount(unittest.TestCase):
    def test_n_features_constant(self):
        """Every call must return exactly N_FEATURES values."""
        s1 = make_record()
        cand = make_record(entity_id='S2-1')
        result = feats(s1, cand)
        self.assertEqual(len(result), N_FEATURES)

    def test_n_features_with_empty_names(self):
        s1 = make_record(name_norm='', addr_norm='')
        cand = make_record(entity_id='S2-1', name_norm='', addr_norm='')
        result = feats(s1, cand)
        self.assertEqual(len(result), N_FEATURES)

    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), N_FEATURES)

    def test_all_finite(self):
        s1 = make_record()
        cand = make_record(entity_id='S2-1')
        for v in feats(s1, cand):
            self.assertTrue(math.isfinite(v), f'Non-finite feature value: {v}')


class TestNameSimilarity(unittest.TestCase):
    def test_identical_names_score_one(self):
        s1 = make_record(name_norm='acme corporation')
        cand = make_record(entity_id='S2-1', name_norm='acme corporation')
        f = feats(s1, cand)
        idx = FEATURE_NAMES.index
        self.assertAlmostEqual(f[idx('name_edit_ratio')], 1.0, places=2)
        self.assertAlmostEqual(f[idx('name_jaccard_tokens')], 1.0, places=2)

    def test_very_different_names_score_low(self):
        s1 = make_record(name_norm='quantum technologies private limited')
        cand = make_record(entity_id='S2-1', name_norm='sunshine bakery')
        f = feats(s1, cand)
        self.assertLess(f[FEATURE_NAMES.index('name_jaccard_tokens')], 0.3)

    def test_empty_name_features(self):
        s1 = make_record(name_norm='')
        cand = make_record(entity_id='S2-1', name_norm='')
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('name_both_empty')], 1.0)
        self.assertEqual(f[FEATURE_NAMES.index('name_edit_ratio')], 0.0)

    def test_name_len_ratio_bounded(self):
        s1 = make_record(name_norm='a b c d e f g h i j k l m n o p')
        cand = make_record(entity_id='S2-1', name_norm='x')
        f = feats(s1, cand)
        ratio = f[FEATURE_NAMES.index('name_len_ratio')]
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class TestAddressSimilarity(unittest.TestCase):
    def test_identical_addresses(self):
        addr = '456 oak avenue chicago'
        s1 = make_record(addr_norm=addr)
        cand = make_record(entity_id='S2-1', addr_norm=addr)
        f = feats(s1, cand)
        self.assertAlmostEqual(f[FEATURE_NAMES.index('addr_edit_ratio')], 1.0, places=2)

    def test_empty_addresses(self):
        s1 = make_record(addr_norm='')
        cand = make_record(entity_id='S2-1', addr_norm='')
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('addr_both_empty')], 1.0)


class TestNumericEvidence(unittest.TestCase):
    def test_house_agree(self):
        s1 = make_record(house_numbers=['42'], postal_codes=[])
        cand = make_record(entity_id='S2-1', house_numbers=['42'], postal_codes=[])
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('house_agree')], 1.0)
        self.assertEqual(f[FEATURE_NAMES.index('house_conflict')], 0.0)
        self.assertEqual(f[FEATURE_NAMES.index('house_both_present')], 1.0)

    def test_house_conflict(self):
        s1 = make_record(house_numbers=['10'], postal_codes=[])
        cand = make_record(entity_id='S2-1', house_numbers=['20'], postal_codes=[])
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('house_conflict')], 1.0)
        self.assertEqual(f[FEATURE_NAMES.index('house_agree')], 0.0)

    def test_one_missing_no_conflict(self):
        s1 = make_record(house_numbers=[], house_number_missing=1)
        cand = make_record(entity_id='S2-1', house_numbers=['15'])
        f = feats(s1, cand)
        # One missing -> both_present=0 -> no agree, no conflict
        self.assertEqual(f[FEATURE_NAMES.index('house_both_present')], 0.0)
        self.assertEqual(f[FEATURE_NAMES.index('house_conflict')], 0.0)
        self.assertEqual(f[FEATURE_NAMES.index('house_s1_missing')], 1.0)

    def test_postal_agree(self):
        s1 = make_record(postal_codes=['10001'])
        cand = make_record(entity_id='S2-1', postal_codes=['10001'])
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('postal_agree')], 1.0)
        self.assertEqual(f[FEATURE_NAMES.index('postal_conflict')], 0.0)

    def test_postal_conflict(self):
        s1 = make_record(postal_codes=['10001'])
        cand = make_record(entity_id='S2-1', postal_codes=['90210'])
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('postal_conflict')], 1.0)


class TestRareTokenFeatures(unittest.TestCase):
    def test_rare_token_overlap(self):
        rare = {'xylophonics', 'gruntworth'}
        s1 = make_record(name_norm='xylophonics corporation')
        cand = make_record(entity_id='S2-1', name_norm='xylophonics limited')
        f = feats(s1, cand, rare=rare)
        self.assertEqual(f[FEATURE_NAMES.index('rare_token_overlap_count')], 1.0)
        self.assertEqual(f[FEATURE_NAMES.index('has_rare_token_overlap')], 1.0)

    def test_no_rare_vocab_gives_zeros(self):
        s1 = make_record(name_norm='xylophonics corporation')
        cand = make_record(entity_id='S2-1', name_norm='xylophonics limited')
        f = feats(s1, cand, rare=None)
        self.assertEqual(f[FEATURE_NAMES.index('rare_token_overlap_count')], 0.0)
        self.assertEqual(f[FEATURE_NAMES.index('has_rare_token_overlap')], 0.0)


class TestRetrievalMetadata(unittest.TestCase):
    def test_route_scores_and_ranks(self):
        s1 = make_record()
        cand = make_record(entity_id='S2-1')
        md = '{"name": {"rank": 0, "score": 0.9}, "combined": {"rank": 3, "score": 0.4}}'
        f = feats(s1, cand, routes='name,combined', score=0.9, metadata=md)
        idx = FEATURE_NAMES.index
        self.assertAlmostEqual(f[idx('score_name')], 0.9)
        self.assertAlmostEqual(f[idx('score_combined')], 0.4)
        self.assertEqual(f[idx('score_rare')], 0.0)
        # invrank = 1/(1+rank): rank 0 -> 1.0, rank 3 -> 0.25, absent -> 0.
        self.assertAlmostEqual(f[idx('invrank_name')], 1.0)
        self.assertAlmostEqual(f[idx('invrank_combined')], 0.25)
        self.assertEqual(f[idx('invrank_address')], 0.0)
        self.assertAlmostEqual(f[idx('best_retrieval_score')], 0.9)
        self.assertEqual(f[idx('n_routes_found')], 2.0)

    def test_missing_metadata_gives_zero_route_features(self):
        s1 = make_record()
        cand = make_record(entity_id='S2-1')
        f = feats(s1, cand, routes='name', score=0.5, metadata=None)
        for name in ('score_name', 'invrank_name', 'score_rare', 'invrank_components'):
            self.assertEqual(f[FEATURE_NAMES.index(name)], 0.0)

    def test_source_flags(self):
        s1 = make_record()
        cand_s2 = make_record(entity_id='S2-99')
        cand_s3 = make_record(entity_id='S3-99')
        # Explicit source int takes precedence; also verify prefix fallback.
        f2 = feats(s1, cand_s2, source=2)
        f3 = feats(s1, cand_s3, source=3)
        f_prefix = feats(s1, cand_s3, source=None)
        self.assertEqual(f2[FEATURE_NAMES.index('source_s2')], 1.0)
        self.assertEqual(f2[FEATURE_NAMES.index('source_s3')], 0.0)
        self.assertEqual(f3[FEATURE_NAMES.index('source_s3')], 1.0)
        self.assertEqual(f_prefix[FEATURE_NAMES.index('source_s3')], 1.0)


class TestInteractionFeatures(unittest.TestCase):
    def test_name_high_addr_low(self):
        s1 = make_record(name_norm='acme corporation', addr_norm='xyz far away place')
        cand = make_record(entity_id='S2-1', name_norm='acme corporation',
                           addr_norm='different street entirely')
        f = feats(s1, cand)
        name_sim = f[FEATURE_NAMES.index('name_edit_ratio')]
        addr_sim = f[FEATURE_NAMES.index('addr_edit_ratio')]
        expected = float(name_sim > 0.85 and addr_sim < 0.50)
        self.assertEqual(f[FEATURE_NAMES.index('name_high_addr_low')], expected)

    def test_both_high(self):
        name = 'quantum solutions private limited'
        addr = '42 baker street london'
        s1 = make_record(name_norm=name, addr_norm=addr)
        cand = make_record(entity_id='S2-1', name_norm=name, addr_norm=addr)
        f = feats(s1, cand)
        self.assertEqual(f[FEATURE_NAMES.index('both_high')], 1.0)


class TestHelpers(unittest.TestCase):
    def test_jaccard_empty(self):
        self.assertEqual(_jaccard(set(), set()), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(_jaccard({'a'}, {'b'}), 0.0)

    def test_jaccard_full_overlap(self):
        self.assertEqual(_jaccard({'a', 'b'}, {'a', 'b'}), 1.0)

    def test_parse_list_valid(self):
        self.assertEqual(_parse_list('["12","34"]'), ['12', '34'])

    def test_parse_list_empty(self):
        self.assertEqual(_parse_list(''), [])

    def test_parse_list_invalid(self):
        self.assertEqual(_parse_list('not_json'), [])

    def test_char_ngrams_short(self):
        # text shorter than n returns the text itself
        result = _char_ngrams('ab', 3)
        self.assertEqual(result, {'ab'})


if __name__ == '__main__':
    unittest.main()
