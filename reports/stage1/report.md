# Stage 1: scorer and validation splits

**PASS**. No normalization, candidate generation, or training performed.

## Scoring

Each S1 contributes exactly one F_0.5 value to the arithmetic mean. For nonempty truth the formula is `5 TP / (5 TP + 4 FP + FN)`. Empty truth scores 1 for empty prediction, otherwise 0. Empty predictions on non-singletons score 0.

Full-data empty predictions reproduce Stage 0; a perfect-label oracle scores 1.0. The oracle is a scorer sanity check, not a trained model or validation result.

## Split construction

Seed: 20260925; requested validation fraction: 0.2. Strata are (country, exact true-match count). Within each stratum, sort by SHA256(JSON([seed, S1 ID])), breaking ties by ID. Round validation size half up; keep both roles nonempty when a stratum has at least two entities. One-entity strata stay in training. Input ordering does not affect assignments.

An S1 and its full positive set always travel together. Current labels have no candidate owned by multiple S1 entities, verified in Stage 0 and protected here by matching source-file CRC and size before reuse of that check. No test data or labels are consulted.

| Mode / role | S1 entities | Singletons | All-empty macro F_0.5 |
|---|---:|---:|---:|
| all | 2,206,821 | 123,247 | 0.0558482088 |
| grouped/train | 1,765,458 | 98,598 | 0.0558483974 |
| grouped/validation | 441,363 | 24,649 | 0.0558474544 |
| transfer_01/train | 883,188 | 49,351 | 0.0558782502 |
| transfer_01/validation | 1,323,633 | 73,896 | 0.0558281638 |
| transfer_02/train | 1,323,633 | 73,896 | 0.0558281638 |
| transfer_02/validation | 883,188 | 49,351 | 0.0558782502 |

## Country transfer

- `transfer_01`: train on **India**, validate on **US**. Other countries, if present, are excluded.
- `transfer_02`: train on **US**, validate on **India**. Other countries, if present, are excluded.

Country modes are generated from observed strings, with no US/India whitelist. Each transfer mode uses the entire training-country S1 population for training and the entire validation-country S1 population for validation, independently of the grouped holdout. These stress tests approximate geographic shift; they cannot estimate France performance directly.

## How later stages must use these splits

- Retrieve against the supplied training S2/S3 pool to retain realistic distractors; candidate appearance alone does not make its label available for fitting.
- Fit pair classifiers only on training S1 groups. Never add validation positives as training examples or mine negatives from validation S1 labels.
- Fit learned text transformations/vocabularies on the training partition. In strict country transfer, fit those on training-country records only; transform other countries without refitting. Fixed rules may process any country.
- Threshold selection on validation is tuning, not an unbiased final score. Use grouped out-of-fold predictions/nested evaluation for later decision models.
- Deduplicated S1 IDs are the supplied grouping unit. Semantically duplicated businesses or chains across distinct IDs are not ruled out by structural checks.

## Files and next step

`splits.tsv` contains one row per S1, country, true-match count, and a role column for every mode. `splits.sqlite` stores complete labels and supports bounded-memory loading/scoring. `manifest.json` records definitions, source hashes, and split-file hash. `report.json` includes exact per-country metrics and per-stratum sizes.

Stage 2 is pending review. If later validation scores are weak, first check candidate misses and false merges separately, then compare country, missing-address, and multi-match slices. Do not change the scorer or weight high-match entities more.
