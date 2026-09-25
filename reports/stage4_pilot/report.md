# Stage 4: Pairwise Feature Engineering

**Status**: PASS

## Dataset Summary

- Total pairs: 301,507
- Positive (true matches): 3,082 (1.02%)
- Negative: 298,425
- Hard negatives (name sim > 0.80): 59,604 (20.0% of negatives)
- Features: 60

## Split Breakdown

| Split | Pairs | Positives | Negatives | Pos rate |
|---|---:|---:|---:|---:|
| train | 0 | 0 | 0 | 0.00% |
| validation | 301,507 | 3,082 | 298,425 | 1.02% |

## Top Feature Means (positives vs negatives analysis available in report.json)

| Feature | Mean | Std |
|---|---:|---:|
| name_edit_ratio | 0.5395 | 0.2128 |
| name_partial_ratio | 0.5469 | 0.1828 |
| name_token_set_ratio | 0.4826 | 0.2072 |
| name_jaccard_tokens | 0.1268 | 0.1687 |
| name_jaccard_chars3 | 0.1470 | 0.1711 |
| name_len_s1 | 3.2187 | 0.3531 |
| name_len_cand | 3.2451 | 0.3949 |
| name_len_ratio | 0.7045 | 0.1974 |
| name_both_empty | 0.0000 | 0.0000 |
| name_s1_empty | 0.0000 | 0.0000 |
| name_cand_empty | 0.0000 | 0.0000 |
| addr_edit_ratio | 0.5021 | 0.1858 |
| addr_partial_ratio | 0.4792 | 0.1315 |
| addr_token_set_ratio | 0.4500 | 0.1471 |
| addr_jaccard_tokens | 0.0821 | 0.1181 |
| addr_jaccard_chars3 | 0.1025 | 0.1255 |
| addr_len_s1 | 3.7950 | 0.4392 |
| addr_len_cand | 3.7014 | 0.6388 |
| addr_len_ratio | 0.7435 | 0.2039 |
| addr_both_empty | 0.0000 | 0.0000 |

## Next Steps

- Stage 5: Train CatBoost classifier on these features with entity-grouped CV
- Hard negative fraction should be > 5% for good calibration
- If positive rate is too low (< 0.5%), consider stratified sampling
