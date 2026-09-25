# Stage 0: full-data audit

Status: **PASS**. All seven TSVs were read in full; no sampling.

## Inputs and task confirmation

- Resource root: `C:\Users\alamu\OneDrive\Documents\amazonml\student_resource`.
- PDF pages read: 8; the last page is blank.
- Each S1 reference entity can match zero, one, or multiple S2/S3 records.
- Only names, addresses, country labels, and provided labels are used; no external data.
- Country is an open-label string; France must be included at inference.
- Optimize exact per-S1 macro F_0.5, including singleton scores of 1 for empty predictions and 0 for nonempty predictions.
- Final model must have MIT/Apache-2.0 licensing and at most 8 billion parameters.
- Both output TSVs require every test S1 exactly once, unique valid S2/S3 IDs only, and final matches must be a subset of the exact candidate set scored by the model.
- Only `matching_results.tsv` is leaderboard-scored; final rankings use the private split.

### Corrections and additions from supplied materials

1. The PDF also requires the filled `Documentation_template.md` in the final ZIP.
2. The PDF says precision is weighted "2x", but the supplied formula is `1.25 TP / (1.25 TP + FP + 0.25 FN)`: FP has four times the FN denominator coefficient. Use the exact formula.
3. The validator skips target-ID existence by default; use `--check-ids`. Missing candidate files and match/candidate subset violations produce only warnings. Our final checks must enforce these requirements strictly.
4. The validator documentation says nonexistent IDs only lower scores, whereas the PDF says they cause rejection. Follow the stricter PDF requirement.

## Actual extracted files

```text
dataset\test\test_source1.tsv
dataset\test\test_source2.tsv
dataset\test\test_source3.tsv
dataset\train\train_ground_truth.tsv
dataset\train\train_source1.tsv
dataset\train\train_source2.tsv
dataset\train\train_source3.tsv
Documentation_template.md
README.md
utils\validate_submission.py
```

The original archive additionally contains macOS metadata; its complete inventory is in `archive_inventory.json`. Extracted members passed ZIP CRC checks.

## Source audit

| File | Rows | Blank names | Blank addresses | Name length min / p50 / p95 / max |
|---|---:|---:|---:|---|
| train_source1.tsv | 2,206,821 | 0 | 0 | 3 / 24 / 37 / 105 |
| train_source2.tsv | 5,034,616 | 0 | 168,967 | 2 / 25 / 40 / 104 |
| train_source3.tsv | 5,285,603 | 0 | 175,916 | 2 / 25 / 42 / 123 |
| test_source1.tsv | 1,732,544 | 0 | 0 | 3 / 24 / 36 / 92 |
| test_source2.tsv | 4,887,273 | 0 | 129,408 | 2 / 25 / 42 / 102 |
| test_source3.tsv | 5,082,316 | 0 | 136,098 | 2 / 25 / 42 / 103 |

Name lengths count original Unicode code points. Exact histograms, means, and additional quantiles are in `audit.json`.

| File | Country | Records | Blank addresses |
|---|---|---:|---:|
| train_source1.tsv | India | 883,188 | 0 |
| train_source1.tsv | US | 1,323,633 | 0 |
| train_source2.tsv | India | 2,017,799 | 57,846 |
| train_source2.tsv | US | 3,016,817 | 111,121 |
| train_source3.tsv | India | 2,115,547 | 64,948 |
| train_source3.tsv | US | 3,170,056 | 110,968 |
| test_source1.tsv | France | 259,452 | 0 |
| test_source1.tsv | India | 809,986 | 0 |
| test_source1.tsv | US | 663,106 | 0 |
| test_source2.tsv | France | 703,378 | 21,537 |
| test_source2.tsv | India | 2,312,565 | 52,764 |
| test_source2.tsv | US | 1,871,330 | 55,107 |
| test_source3.tsv | France | 731,615 | 21,541 |
| test_source3.tsv | India | 2,405,000 | 59,240 |
| test_source3.tsv | US | 1,945,701 | 55,317 |

### Missing-value and schema checks

All source files have exactly four columns; ground truth has exactly two. Strict CSV and pandas row counts agree. Blank/whitespace fields are counted separately from whole-field null-like strings. No automatic NA coercion is used.

| File | Blank fields by column | Literal null-like fields by column |
|---|---|---|
| train_source1.tsv | none | none |
| train_source2.tsv | {'business_address': 168967} | {'business_name': 6} |
| train_source3.tsv | {'business_address': 175916} | {'business_name': 18} |
| test_source1.tsv | none | none |
| test_source2.tsv | {'business_address': 129408} | {'business_name': 49} |
| test_source3.tsv | {'business_address': 136098} | {'business_name': 61} |

## Ground truth and sanity baseline

**All-empty macro F_0.5 = 0.0558482088** (123,247 singletons / 2,206,821 S1 entities).

This is an analytic baseline; the general scorer belongs to Stage 1. An independent standard-library CSV pass confirmed the same singleton count, source-type counts, and full match-count histogram.

| Match source type | S1 entities | Share |
|---|---:|---:|
| S2_only | 143,029 | 6.4812% |
| S3_only | 164,498 | 7.4541% |
| both | 1,776,047 | 80.4799% |
| singleton | 123,247 | 5.5848% |

| True matches per S1 | S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

Mean matches per S1: 3.461253; median: 3.

| Country | S1 entities | Singletons | All-empty macro F_0.5 |
|---|---:|---:|---:|
| India | 883,188 | 49,351 | 0.0558782502 |
| US | 1,323,633 | 73,896 | 0.0558281638 |

## Integrity checks

| Check | Count |
|---|---:|
| duplicate_train_id_groups | 0 |
| duplicate_test_id_groups | 0 |
| duplicate_truth_s1_groups | 0 |
| missing_truth_s1 | 0 |
| unknown_truth_s1 | 0 |
| unknown_truth_targets | 0 |
| targets_assigned_to_multiple_s1 | 0 |
| cross_country_true_pairs | 0 |
| train_test_id_overlap | 0 |
| duplicate_ids_within_list_rows | 0 |
| invalid_match_tokens | 0 |

Structural checks cannot establish semantic correctness of supplied labels.

## Review gate and next step

Stage 0 is complete. Stage 1 has not started. Next, implement and test the exact macro scorer and entity-grouped/country-transfer validation. If later scores are weak, first separate candidate misses from false merges and compare missing-address and country slices. Multiple true matches must remain possible; missing addresses must be distinct from address conflict.

Four focused audit tests passed. Reproduction commands and pinned dependencies are in `code/business_entity_resolution/README.md`.
