# Stage 2: normalization and structured evidence

**PASS**. Processed and re-read all 24,229,173 source records. No model, candidate generator, or submission was created.

## Text policy

- Original entity ID, business name, address, and country remain byte-for-byte equivalent as UTF-8 field values, in the same row order. TSV quoting may change.
- Unicode NFKC, case folding, whitespace cleaning, punctuation/symbol spacing, and `&` → `and`. Accents and non-Latin combining marks are retained.
- A separate basic name view retains lexical suffixes. The legal-normalized view expands recognized trailing suffix tokens such as Ltd/Limited, Pvt/Private, Corp/Corporation, and Inc/Incorporated. It never removes suffixes or interior words.
- Decimal digits, including zero padding, survive in both primary normalized views. Unicode decimal digits become ASCII; raw originals remain intact.

## Numeric evidence and missingness

- House/premise numbers retain fractions, ranges, slash numbering, and letter suffixes. Explicit house/door/plot/building labels and weaker leading-number heuristics have separate provenance values. These are extracted hypotheses, not verified addresses.
- Street ordinals and explicitly numbered streets are stored separately.
- Numeric postcode shapes supported: 5 digits, 6 digits, and 5+4 digits. Explicit postcode labels and weaker trailing-number heuristics have separate provenance. No country whitelist or country-specific lookup is used. Unsupported alphanumeric postcodes remain in raw/normalized text and are not claimed as extracted codes.
- Ambiguous numbers remain in candidate fields. Confirmed positional house-number occurrences are not also used as postcode occurrences.
- Missing extraction means no accepted value was found, not that the real business has no number/code. Empty lists, explicit missingness flags, ambiguity flags, and provenance must be used together; absence is never a numeric conflict.
- Whole-field null-like strings are diagnostic flags. A name such as `NA` is preserved and is not silently replaced with a missing value.

## Full-data results

| File | Rows | Blank addresses | House evidence | Postal evidence | Ambiguous postal |
|---|---:|---:|---:|---:|---:|
| test_source1.tsv | 1,732,544 | 0 | 1,218,726 | 85 | 25,614 |
| test_source2.tsv | 4,887,273 | 129,408 | 3,365,034 | 185 | 81,781 |
| test_source3.tsv | 5,082,316 | 136,098 | 3,439,379 | 403 | 86,230 |
| train_source1.tsv | 2,206,821 | 0 | 1,591,186 | 116 | 48,386 |
| train_source2.tsv | 5,034,616 | 168,967 | 3,446,488 | 178 | 118,399 |
| train_source3.tsv | 5,285,603 | 175,916 | 3,572,193 | 409 | 125,295 |

Evidence counts measure extraction coverage, not extraction accuracy. No gold structured-address annotations are supplied.

| File | Country | Rows | House evidence | Postal evidence |
|---|---|---:|---:|---:|---:|
| test_source1.tsv | France | 259,452 | 223,034 | 19 |
| test_source1.tsv | India | 809,986 | 438,363 | 32 |
| test_source1.tsv | US | 663,106 | 557,329 | 34 |
| test_source2.tsv | France | 703,378 | 529,207 | 38 |
| test_source2.tsv | India | 2,312,565 | 1,399,677 | 101 |
| test_source2.tsv | US | 1,871,330 | 1,436,150 | 46 |
| test_source3.tsv | France | 731,615 | 553,268 | 36 |
| test_source3.tsv | India | 2,405,000 | 1,387,547 | 288 |
| test_source3.tsv | US | 1,945,701 | 1,498,564 | 79 |
| train_source1.tsv | India | 883,188 | 478,997 | 34 |
| train_source1.tsv | US | 1,323,633 | 1,112,189 | 82 |
| train_source2.tsv | India | 2,017,799 | 1,183,543 | 104 |
| train_source2.tsv | US | 3,016,817 | 2,262,945 | 74 |
| train_source3.tsv | India | 2,115,547 | 1,180,393 | 264 |
| train_source3.tsv | US | 3,170,056 | 2,391,800 | 145 |

## Validation and reproducibility

All input sizes and CRCs match Stage 0. Every output was re-read to verify its 25-column schema, flags, row count, and SHA256 digest of the original four fields in order. Row counts, country counts, and raw missingness match Stage 0. Name/address digit preservation was asserted on every record before writing.

`report.json` contains provenance counts, per-country coverage, deterministic examples of ambiguous evidence, source fingerprints, and round-trip digests. The transformed TSV paths are recorded there. No labeled pairs were used to select normalization rules, and no model score is reported in this stage.

## Review gate

Stage 3 is pending approval. Next, build the retrieval routes and measure per-S1 candidate recall. If recall is weak, inspect misses by script, country, missing address, and numeric ambiguity. Preserve raw/basic views as alternative retrieval inputs; do not compensate by accepting weak matches.
