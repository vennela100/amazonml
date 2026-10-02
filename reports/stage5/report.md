# Stage 5: CatBoost Classifier

**Status**: PASS

## Decision threshold

- Selected threshold: **0.81** (from out-of-fold training population)
- Scored at inference on the SAME scale (raw model probability); no calibrator sits between the threshold and inference.

## Held-out validation (full population, exact scorer)

- (no held-out validation npz supplied; used OOF)

## Out-of-fold (training population, exact scorer)

- Macro F0.5: **0.8820** at threshold 0.81

## Per-fold diagnostics (pair-level, not the decision metric)

| Fold | Train pairs | Val pairs | pair-F0.5@0.5 | Iters |
|---|---:|---:|---:|---:|
| 1 | 6,023,886 | 1,505,973 | 0.8961 | 399 |
| 2 | 6,023,890 | 1,505,969 | 0.8939 | 399 |
| 3 | 6,023,886 | 1,505,973 | 0.8988 | 399 |
| 4 | 6,023,887 | 1,505,972 | 0.9002 | 397 |
| 5 | 6,023,887 | 1,505,972 | 0.8924 | 398 |

## Top 20 Feature Importances (by PredValuesChange)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | addr_len_cand | 16.48 |
| 2 | addr_token_set_ratio | 16.09 |
| 3 | name_edit_ratio | 10.30 |
| 4 | name_partial_ratio | 8.26 |
| 5 | name_len_cand | 6.22 |
| 6 | addr_jaccard_tokens | 6.20 |
| 7 | house_conflict | 5.97 |
| 8 | name_jaccard_chars3 | 5.49 |
| 9 | name_jaccard_tokens | 3.64 |
| 10 | name_len_ratio | 3.28 |
| 11 | addr_len_ratio | 3.16 |
| 12 | name_high_addr_low | 2.36 |
| 13 | name_token_set_ratio | 1.55 |
| 14 | best_retrieval_score | 1.38 |
| 15 | rare_token_overlap_frac_cand | 1.16 |
| 16 | name_len_s1 | 1.13 |
| 17 | addr_jaccard_chars3 | 0.95 |
| 18 | invrank_rare | 0.94 |
| 19 | source_s3 | 0.86 |
| 20 | addr_len_s1 | 0.68 |