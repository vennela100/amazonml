# Stage 5: CatBoost Classifier

**Status**: PASS

## Decision threshold

- Selected threshold: **0.72** (from held-out validation population)
- Scored at inference on the SAME scale (raw model probability); no calibrator sits between the threshold and inference.

## Held-out validation (full population, exact scorer)

- Macro F0.5: **0.8855** at threshold 0.72
- False-singleton rate: 0.14285714285714285
- Missed non-singleton rate: 0.025236593059936908

## Out-of-fold (training population, exact scorer)

- Macro F0.5: **0.8782** at threshold 0.84

## Per-fold diagnostics (pair-level, not the decision metric)

| Fold | Train pairs | Val pairs | pair-F0.5@0.5 | Iters |
|---|---:|---:|---:|---:|
| 1 | 726,773 | 181,693 | 0.9104 | 787 |
| 2 | 726,772 | 181,694 | 0.9109 | 751 |
| 3 | 726,773 | 181,693 | 0.9091 | 799 |
| 4 | 726,773 | 181,693 | 0.9159 | 740 |
| 5 | 726,773 | 181,693 | 0.9163 | 799 |

## Top 20 Feature Importances (by PredValuesChange)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | addr_token_set_ratio | 12.27 |
| 2 | name_edit_ratio | 10.21 |
| 3 | name_partial_ratio | 10.00 |
| 4 | name_len_cand | 8.63 |
| 5 | name_jaccard_chars3 | 7.08 |
| 6 | house_conflict | 6.89 |
| 7 | addr_len_cand | 6.44 |
| 8 | name_len_ratio | 4.42 |
| 9 | addr_jaccard_tokens | 4.09 |
| 10 | name_jaccard_tokens | 3.34 |
| 11 | addr_len_ratio | 3.26 |
| 12 | addr_jaccard_chars3 | 2.79 |
| 13 | name_token_set_ratio | 2.16 |
| 14 | rare_token_overlap_frac_cand | 2.12 |
| 15 | name_len_s1 | 1.97 |
| 16 | best_retrieval_score | 1.96 |
| 17 | addr_edit_ratio | 1.85 |
| 18 | name_high_addr_low | 1.58 |
| 19 | addr_len_s1 | 1.39 |
| 20 | addr_partial_ratio | 1.09 |