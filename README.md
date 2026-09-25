# Business Entity Resolution — Amazon ML Challenge

Given three noisy business record sources (Source 1 = deduplicated reference,
Sources 2 and 3 = candidate pools), find every Source 2/3 record that matches
each Source 1 entity, or predict "no match". Records have name + address only,
across US and India (training) with France added at test time. Scored by
macro-averaged **F₀.₅** per Source 1 entity (precision weighted 4× over recall).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 0 | `stage0.py` | Full-data audit of the seven input TSVs |
| 1 | `splits.py`, `scoring.py` | Exact F₀.₅ scorer + train/validation split |
| 2 | `normalization.py`, `stage2.py` | Country-agnostic name/address normalization |
| 3 | `stage3.py`, `blocking_index.py`, `blocking_retrieval.py` | Blocking + candidate retrieval (train/validation) |
| 3-test | `stage3_inference.py` | Candidate retrieval for the unlabeled test set |
| 4 | `stage4.py`, `features.py` | Pairwise features (+ population & truth for scoring) |
| 5 | `stage5.py` | CatBoost matcher, grouped CV, threshold selection |
| 6 | `stage6.py` | Predictions → `matching_results.tsv` + `candidate_pairs.tsv` |
| — | `evaluation.py` | Population-aware F₀.₅ scoring (single source of truth) |

## Key design points

- **`scoring.py`/`evaluation.py` are the single source of truth for the metric.**
  Scoring is set-based per entity over the full population, so singletons and
  matches lost during retrieval are counted correctly.
- **Country is an open string label** — nothing is hard-coded to {US, India}, so
  France (test-only) flows through the same pipeline.
- The decision threshold is chosen and applied on the same raw-probability scale.

## Reproduce

```bash
pip install -r code/business_entity_resolution/requirements.txt
# See code/business_entity_resolution/README.md for stage-by-stage run commands.
python -m pytest code/business_entity_resolution/src -q
```

Data (`student_resource/`) and generated artifacts (`artifacts/`, large report
binaries) are intentionally excluded from version control — see `.gitignore`.
