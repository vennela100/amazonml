"""Deterministic entity-grouped holdout and open-label country-transfer modes."""
from __future__ import annotations

import hashlib
import json
import math


def entity_hash(entity_id, seed):
    return hashlib.sha256(json.dumps([seed, entity_id], separators=(',', ':')).encode('utf-8')).hexdigest()


def validation_size(n, fraction):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be strictly between 0 and 1')
    if n < 1:
        raise ValueError('Stratum must contain at least one entity')
    # A one-entity stratum cannot appear in both folds; keep it in training.
    return 0 if n == 1 else min(n - 1, max(1, math.floor(n * fraction + 0.5)))


def assign_grouped(db, fraction):
    """Rank each (country, exact match count) stratum by SHA256(seed, S1 ID)."""
    validation_size(1, fraction)
    db.execute("UPDATE entities SET grouped_role='train'")
    strata = db.execute('SELECT country, match_count, COUNT(*) FROM entities '
                        'GROUP BY country, match_count ORDER BY country, match_count').fetchall()
    for country, count, n in strata:
        size = validation_size(n, fraction)
        db.execute("UPDATE entities SET grouped_role='validation' WHERE id IN "
                   '(SELECT id FROM entities WHERE country=? AND match_count=? '
                   'ORDER BY split_hash, id LIMIT ?)', (country, count, size))
    db.commit()
    return [{'country': country, 'match_count': count, 'entities': n,
             'validation_entities': validation_size(n, fraction)} for country, count, n in strata]


def make_modes(countries):
    countries = sorted(set(countries))
    modes = {'grouped': {'type': 'grouped'}}
    for train in countries:
        for val in countries:
            if train != val:
                key = f'transfer_{len(modes):02d}'
                modes[key] = {'type': 'country_transfer', 'train_country': train,
                              'validation_country': val}
    return modes


def selection(mode, role, alias='e'):
    """Parameterized SQL predicate used identically by loaders and the scorer."""
    if role not in ('train', 'validation'):
        raise ValueError(f'Unknown role: {role}')
    if alias not in ('e', ''):
        raise ValueError('Unsupported table alias')
    prefix = alias + '.' if alias else ''
    if mode['type'] == 'grouped':
        return f'{prefix}grouped_role=?', (role,)
    if mode['type'] == 'country_transfer':
        return f'{prefix}country=?', (mode[f'{role}_country'],)
    raise ValueError(f'Unknown mode type: {mode["type"]}')


def iter_entities(db, modes, mode='grouped', role='train'):
    """Yield each selected S1 with its entire positive set; order is deterministic."""
    predicate, parameters = selection(modes[mode], role)
    return db.execute('SELECT e.id, e.country, e.matched_ids FROM entities e WHERE '
                      + predicate + ' ORDER BY e.id', parameters)


def role_for(mode, country, grouped_role):
    if mode['type'] == 'grouped':
        return grouped_role
    if country == mode['train_country']:
        return 'train'
    if country == mode['validation_country']:
        return 'validation'
    return None
