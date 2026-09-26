# Stage 6: Predictions & Submission

**Status**: PASS

## Outputs

- `matching_results.tsv` — one row per S1 (matching_results)
- `candidate_pairs.tsv` — one row per S1 (candidate_pairs)

## Summary

| Metric | Value |
|---|---|
| Threshold (raw proba) | 0.72 |
| S1 entities | 1,000 |
| Predicted match pairs | 2,961 |
| Candidate pairs | 301,507 |
| Macro F0.5 | **0.8855** |
| False-singleton rate | 0.14285714285714285 |
| Missed non-singleton rate | 0.025236593059936908 |

## Next Steps

- Run `utils/validate_submission.py --check-ids` before upload.
