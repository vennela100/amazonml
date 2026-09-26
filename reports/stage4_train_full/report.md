# Stage 4: Pairwise Feature Engineering

**Status**: PASS

## Dataset Summary

- Total pairs: 7,529,859
- Positive (true matches): 77,622 (1.03%)
- Negative: 7,452,237
- Hard negatives (name sim > 0.80): 1,511,990 (20.3% of negatives)
- Features: 60

## Split Breakdown

| Split | Pairs | Positives | Negatives | Pos rate |
|---|---:|---:|---:|---:|
| train | 7,529,859 | 77,622 | 7,452,237 | 1.03% |
| validation | 0 | 0 | 0 | 0.00% |

## Top Feature Means (positives vs negatives analysis available in report.json)

| Feature | Mean | Std |
|---|---:|---:|
| name_edit_ratio | 0.5317 | 0.2117 |
| name_partial_ratio | 0.5499 | 0.1821 |
| name_token_set_ratio | 0.4798 | 0.2063 |
| name_jaccard_tokens | 0.1269 | 0.1685 |
| name_jaccard_chars3 | 0.1459 | 0.1690 |
| name_len_s1 | 3.2730 | 0.3576 |
| name_len_cand | 3.3079 | 0.4000 |
| name_len_ratio | 0.7083 | 0.1958 |
| name_both_empty | 0.0000 | 0.0000 |
| name_s1_empty | 0.0000 | 0.0000 |
| name_cand_empty | 0.0000 | 0.0000 |
| addr_edit_ratio | 0.4974 | 0.1846 |
| addr_partial_ratio | 0.4836 | 0.1317 |
| addr_token_set_ratio | 0.4494 | 0.1459 |
| addr_jaccard_tokens | 0.0820 | 0.1182 |
| addr_jaccard_chars3 | 0.1020 | 0.1241 |
| addr_len_s1 | 3.8499 | 0.4301 |
| addr_len_cand | 3.7704 | 0.6397 |
| addr_len_ratio | 0.7565 | 0.2015 |
| addr_both_empty | 0.0000 | 0.0000 |

## Next Steps

- Stage 5: Train CatBoost classifier on these features with entity-grouped CV
- Hard negative fraction should be > 5% for good calibration
- If positive rate is too low (< 0.5%), consider stratified sampling
