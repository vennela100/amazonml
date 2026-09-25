# Stage 6: Predictions & Submission

**Status**: PASS

## Outputs

- `matching_results.tsv` — one row per S1 (matching_results)
- `candidate_pairs.tsv` — one row per S1 (candidate_pairs)

## Summary

| Metric | Value |
|---|---|
| Threshold (raw proba) | 0.85 |
| S1 entities | 1,000 |
| Predicted match pairs | 2,875 |
| Candidate pairs | 301,507 |
| Macro F0.5 | **0.8591** |
| False-singleton rate | 0.22448979591836735 |
| Missed non-singleton rate | 0.035751840168243953 |

## Next Steps

- Run `utils/validate_submission.py --check-ids` before upload.
