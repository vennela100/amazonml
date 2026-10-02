"""Exact per-reference macro F_0.5 and strict TSV parsing (standard library)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

MATCH_HEADER = ['source1_entity_id', 'matched_entity_ids']


def read_tsv(path, header):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t', strict=True)
        actual = next(reader, None)
        if actual != header:
            raise ValueError(f'{path}: expected header {header}, got {actual}')
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'{path}:{reader.line_num}: expected {len(header)} columns')
            yield row


def validate_s1(entity_id):
    if (not entity_id.startswith('S1-') or len(entity_id) <= 3
            or any(c.isspace() for c in entity_id) or ',' in entity_id):
        raise ValueError(f'Invalid S1 ID: {entity_id!r}')


def parse_ids(text):
    """An empty field is the sole empty-list representation; never coerce NA."""
    if text == '':
        return set()
    ids = text.split(',')
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate ID in list: {text[:100]!r}')
    for entity_id in ids:
        if (not entity_id.startswith(('S2-', 'S3-')) or len(entity_id) <= 3
                or any(c.isspace() for c in entity_id)):
            raise ValueError(f'Invalid candidate ID: {entity_id!r}')
    return set(ids)


def f05_counts(tp, fp, fn):
    """5 TP / (5 TP + 4 FP + FN), with the challenge's singleton convention."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (tp, fp, fn)):
        raise ValueError('TP, FP, and FN must be nonnegative integers')
    if tp + fn == 0:
        return 1.0 if fp == 0 else 0.0
    return 5 * tp / (5 * tp + 4 * fp + fn)


def entity_f05(truth, prediction):
    """Inputs are sets of IDs. TSV callers must use parse_ids for strict validation."""
    tp = len(truth & prediction)
    return f05_counts(tp, len(prediction) - tp, len(truth) - tp)


@dataclass
class MacroMetrics:
    entities: int = 0
    total: float = 0.0
    correction: float = 0.0
    true_singletons: int = 0
    false_singletons: int = 0  # True singleton incorrectly assigned any match.
    missed_non_singletons: int = 0
    nonempty_predictions: int = 0
    exact_sets: int = 0

    def add(self, truth, prediction):
        value = entity_f05(truth, prediction)
        # Neumaier compensated summation: bounded memory, equal entity weight.
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated
        self.entities += 1
        self.true_singletons += int(not truth)
        self.false_singletons += int(not truth and bool(prediction))
        self.missed_non_singletons += int(bool(truth) and not prediction)
        self.nonempty_predictions += int(bool(prediction))
        self.exact_sets += int(truth == prediction)
        return value

    def result(self):
        if self.entities == 0:
            raise ValueError('Cannot score an empty evaluation population')
        score = (self.total + self.correction) / self.entities
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise AssertionError(f'Invalid macro score: {score}')
        non_singletons = self.entities - self.true_singletons
        return {
            'entities': self.entities, 'macro_f05': score,
            'true_singletons': self.true_singletons,
            'singleton_fraction': self.true_singletons / self.entities,
            'false_singleton_count': self.false_singletons,
            'false_singleton_rate': self.false_singletons / self.true_singletons if self.true_singletons else None,
            'missed_non_singleton_rate': self.missed_non_singletons / non_singletons if non_singletons else None,
            'nonempty_predictions': self.nonempty_predictions,
            'exact_set_accuracy': self.exact_sets / self.entities,
        }
