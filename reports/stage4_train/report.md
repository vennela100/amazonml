# Stage 4: Pairwise Feature Engineering

**Status**: PASS

## Dataset Summary

- Total pairs: 908,466
- Positive (true matches): 9,678 (1.07%)
- Negative: 898,788
- Hard negatives (name sim > 0.80): 177,153 (19.7% of negatives)
- Features: 60

## Split Breakdown

| Split | Pairs | Positives | Negatives | Pos rate |
|---|---:|---:|---:|---:|
| train | 908,466 | 9,678 | 898,788 | 1.07% |
| validation | 0 | 0 | 0 | 0.00% |

## Top Feature Means (positives vs negatives analysis available in report.json)

| Feature | Mean | Std |
|---|---:|---:|
| name_edit_ratio | 0.5356 | 0.2105 |
| name_partial_ratio | 0.5439 | 0.1803 |
| name_token_set_ratio | 0.4777 | 0.2031 |
| name_jaccard_tokens | 0.1217 | 0.1676 |
| name_jaccard_chars3 | 0.1408 | 0.1687 |
| name_len_s1 | 3.1977 | 0.3511 |
| name_len_cand | 3.2349 | 0.3962 |
| name_len_ratio | 0.7014 | 0.1964 |
| name_both_empty | 0.0000 | 0.0000 |
| name_s1_empty | 0.0000 | 0.0000 |
| name_cand_empty | 0.0000 | 0.0000 |
| addr_edit_ratio | 0.4962 | 0.1828 |
| addr_partial_ratio | 0.4791 | 0.1323 |
| addr_token_set_ratio | 0.4486 | 0.1456 |
| addr_jaccard_tokens | 0.0811 | 0.1167 |
| addr_jaccard_chars3 | 0.1015 | 0.1248 |
| addr_len_s1 | 3.7258 | 0.3997 |
| addr_len_cand | 3.6549 | 0.6221 |
| addr_len_ratio | 0.7532 | 0.1979 |
| addr_both_empty | 0.0000 | 0.0000 |

## Next Steps

- Stage 5: Train CatBoost classifier on these features with entity-grouped CV
- Hard negative fraction should be > 5% for good calibration
- If positive rate is too low (< 0.5%), consider stratified sampling
