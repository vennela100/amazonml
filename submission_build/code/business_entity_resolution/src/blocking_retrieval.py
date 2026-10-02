"""Bounded approximate retrieval, followed by true character-trigram TF-IDF cosine."""
from collections import Counter
from functools import lru_cache
import json
import math

from blocking_index import connect, get_state

ROUTES = ('name', 'address', 'combined', 'rare', 'components')


def grams(text):
    # Same contiguous Unicode code-point trigrams as the native inverted index.
    return Counter(text[i:i + 3] for i in range(max(0, len(text) - 2)))


def vector(counts, weights, unseen):
    values = {gram: count * weights.get(gram, unseen) for gram, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in values.values()))
    return {gram: value / norm for gram, value in values.items()} if norm else {}


def vectors(name, address, idf):
    a, b = grams(name), grams(address)
    return (vector(a, idf['name'], idf['unseen']),
            vector(b, idf['address'], idf['unseen']),
            vector(a + b, idf['combined'], idf['unseen']))


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return min(1.0, max(0.0, sum(value * b.get(term, 0) for term, value in a.items())))


def quote(term):
    return '"' + term.replace('"', '""') + '"'


class Retriever:
    def __init__(self, index_path, idf, config):
        self.db = connect(index_path, True)
        if not get_state(self.db, 'complete', False):
            self.db.close()
            raise ValueError('Candidate pool index is incomplete')
        self.idf, self.config = idf, config
        self.pool_rows = get_state(self.db, 'rows')
        self.stored_statistics = get_state(self.db, 'statistics_ready', False)
        # Bounded caches. No global per-S1 candidate accumulator.
        self.df = lru_cache(maxsize=100000)(self._df)
        self.record = lru_cache(maxsize=10000)(self._record)

    def close(self):
        self.df.cache_clear()
        self.record.cache_clear()
        self.db.close()

    def _df(self, kind, field, term):
        if self.stored_statistics:
            row = self.db.execute('SELECT df FROM statistics WHERE kind=? AND term=? AND field=?',
                                  (kind, term, field)).fetchone()
            return row[0] if row else 0
        table = 'gram_vocab' if kind == 'gram' else 'word_vocab'
        row = self.db.execute(f'SELECT doc FROM {table} WHERE term=? AND col=?', (term, field)).fetchone()
        return row[0] if row else 0

    def anchors(self, field, text):
        ranked = [(self.df('gram', field, term), term) for term in grams(text)]
        return [term for count, term in sorted(ranked) if count > 0][:self.config['anchors']]

    def hits(self, table, expression, limit):
        return [row[0] for row in self.db.execute(
            f'SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rowid LIMIT ?', (expression, limit))]

    def shortlist(self, field, anchors):
        counts = Counter()
        quota = self.config['postings_per_anchor']
        for term in anchors:
            counts.update(self.hits('grams', f'{field} : {quote(term)}', quota))
        # Conjunctions reach useful records beyond individual posting-list caps.
        for offset in range(0, min(len(anchors) - 1, 6), 2):
            expression = f'{field} : ({quote(anchors[offset])} AND {quote(anchors[offset + 1])})'
            counts.update(self.hits('grams', expression, quota))
        return {rid for rid, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                [:self.config['shortlist']]}

    def word_terms(self, field, text, ceiling):
        terms = [(self.df('word', field, term), term) for term in set(text.split()) if len(term) >= 2]
        return [term for count, term in sorted(terms) if 1 <= count <= ceiling]

    def _record(self, rid):
        row = self.db.execute('SELECT entity_id,name,address FROM records WHERE rid=?', (rid,)).fetchone()
        if row is None:
            raise AssertionError(f'Posting references missing record {rid}')
        return row[0], vectors(row[1], row[2], self.idf)

    def retrieve(self, query):
        """Return union keyed by valid pool ID; every route retains rank and score."""
        name, address = query['name'], query['address']
        anchors = {field: self.anchors(field, text) for field, text in (('name', name), ('address', address))}
        short = {field: self.shortlist(field, terms) for field, terms in anchors.items()}
        short['combined'] = set(short['name']) | short['address']
        if anchors['name'] and anchors['address']:
            for offset in range(min(3, len(anchors['name']), len(anchors['address']))):
                expr = f'name : {quote(anchors["name"][offset])} AND address : {quote(anchors["address"][offset])}'
                short['combined'].update(self.hits('grams', expr, self.config['postings_per_anchor']))
        short['rare'], short['components'] = set(), set()
        rare_terms = {}
        for field, text in (('name', name), ('address', address)):
            rare_terms[field] = self.word_terms(field, text, self.config['rare_max_df'])[:self.config['anchors']]
            for term in rare_terms[field]:
                short['rare'].update(self.hits('words', f'{field} : {quote(term)}', self.config['rare_max_df']))
        for postal in json.loads(query['postals']):
            short['components'].update(row[0] for row in self.db.execute(
                "SELECT rid FROM components WHERE kind='postal' AND value=? LIMIT ?",
                (postal, self.config['shortlist'])))
        for house in json.loads(query['houses']):
            for field, text in (('name', name), ('address', address)):
                for term in self.word_terms(field, text, self.config['component_max_df'])[:2]:
                    expression = f'{field} : {quote(term)}'
                    short['components'].update(row[0] for row in self.db.execute(
                        "SELECT words.rowid FROM words JOIN components c ON c.rid=words.rowid "
                        "WHERE words MATCH ? AND c.kind='house' AND c.value=? ORDER BY words.rowid LIMIT ?",
                        (expression, house, self.config['postings_per_anchor'])))
        q_vectors = vectors(name, address, self.idf)
        union = {}
        for route in ROUTES:
            # All routes are bounded before expensive cosine scoring as well as after it.
            ids = sorted(short[route])[:self.config['shortlist'] * (3 if route == 'combined' else 1)]
            scored = []
            for rid in ids:
                entity_id, candidate_vectors = self.record(rid)
                field_index = 0 if route == 'name' else 1 if route == 'address' else 2
                score = cosine(q_vectors[field_index], candidate_vectors[field_index])
                if score > 0 or route in ('rare', 'components'):
                    scored.append((score, entity_id))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            for rank, (score, entity_id) in enumerate(scored[:self.config['budgets'][route]], 1):
                union.setdefault(entity_id, {})[route] = {'rank': rank, 'score': score}
        return union
