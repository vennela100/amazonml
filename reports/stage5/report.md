# Stage 5: CatBoost Classifier

**Status**: PASS

## Decision threshold

- Selected threshold: **0.85** (from held-out validation population)
- Scored at inference on the SAME scale (raw model probability); no calibrator sits between the threshold and inference.

## Held-out validation (full population, exact scorer)

- Macro F0.5: **0.8591** at threshold 0.85
- False-singleton rate: 0.22448979591836735
- Missed non-singleton rate: 0.035751840168243953

## Out-of-fold (training population, exact scorer)

- Macro F0.5: **0.8589** at threshold 0.9

## Per-fold diagnostics (pair-level, not the decision metric)

| Fold | Train pairs | Val pairs | pair-F0.5@0.5 | Iters |
|---|---:|---:|---:|---:|
| 1 | 726,773 | 181,693 | 0.8451 | 297 |
| 2 | 726,772 | 181,694 | 0.8579 | 293 |
| 3 | 726,773 | 181,693 | 0.8508 | 299 |
| 4 | 726,773 | 181,693 | 0.8549 | 274 |
| 5 | 726,773 | 181,693 | 0.8623 | 298 |

## Top 20 Feature Importances (by PredValuesChange)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | name_edit_ratio | 14.56 |
| 2 | addr_token_set_ratio | 10.94 |
| 3 | name_partial_ratio | 9.86 |
| 4 | addr_jaccard_chars3 | 8.63 |
| 5 | name_len_cand | 7.16 |
| 6 | addr_len_cand | 5.57 |
| 7 | addr_edit_ratio | 5.36 |
| 8 | house_conflict | 5.29 |
| 9 | addr_len_ratio | 4.04 |
| 10 | name_jaccard_chars3 | 3.85 |
| 11 | addr_jaccard_tokens | 3.57 |
| 12 | name_jaccard_tokens | 3.20 |
| 13 | addr_partial_ratio | 2.65 |
| 14 | name_len_ratio | 2.40 |
| 15 | rare_token_overlap_frac_cand | 2.36 |
| 16 | source_s2 | 1.98 |
| 17 | name_token_set_ratio | 1.48 |
| 18 | name_high_addr_low | 1.03 |
| 19 | best_retrieval_score | 0.75 |
| 20 | invrank_combined | 0.64 |