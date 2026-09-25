# Business entity resolution

Stages 0?2 are complete. Stage 3 candidate retrieval is implemented; see its reports
for the exact completed evaluation scope. Stages 4?8 require review and approval.
No external data or classifier training is used by Stages 0?3.

## Environment

Validated on Windows with Python 3.10.11. Install the pinned software dependencies:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

Commands below run from the workspace root containing `code/`. No network calls are
made by the audit. The supplied ZIP and PDF remain unchanged. The archive is unpacked
into `student_resource/`; macOS metadata is excluded. Each extracted member is read
to EOF to verify its ZIP CRC. The complete original inventory is saved in the report.

## Reproduce Stage 0

```powershell
python -X utf8 code/business_entity_resolution/src/stage0.py --pdf "C:\Users\alamu\Downloads\6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf" --archive "C:\Users\alamu\Downloads\6ab10eb3b23ba_student_resource.zip" --report-dir reports/stage0-rerun
python -X utf8 code/business_entity_resolution/src/render_stage0.py --report-dir reports/stage0-rerun
python -X utf8 -m unittest discover -s code/business_entity_resolution/src -p "test_stage0.py" -v
```

The initial run used the pre-existing local PDF reader, with the additional argument
`--pdf-tools-directory .tmp_pdf_tools`. This is unnecessary after installing the pinned
requirements. Choose a fresh report directory for each run: an existing audit database
is never overwritten. `--resource-dir` defaults to `student_resource`; extraction
replaces the supplied resource files there. `--chunk-size` defaults to 100000.

## Results and validation

The audit reads all seven TSVs, explicitly using `sep="\t"`, with string types and
automatic NA conversion disabled. A separate strict CSV pass verifies every row's
column count; the pandas row count must agree. Raw data is never transformed.
Blank/whitespace fields and whole-field null-like strings are counted separately;
the latter are diagnostics, not automatic missing-value replacements. Name length
means Unicode code points in the original string, with exact histogram quantiles.

`reports/stage0/audit.json` contains row counts, country distributions, missingness,
name lengths, match-count distributions, source types, ID-integrity checks, the
analytic all-empty macro F_0.5 baseline, and per-country baselines.
`problem_statement.txt` contains the full PDF text with page boundaries, and
`archive_inventory.json` lists every original ZIP member. `audit.sqlite` stores only
IDs, countries, sources, and ground-truth relationships for exact disk-backed checks.
The database is a local audit artifact, not part of the final submission package.

The audit exits nonzero for malformed files, duplicate IDs, invalid target prefixes,
incomplete labels, dangling labels, or targets assigned to multiple reference entities.
Cross-country labeled pairs and train/test ID overlap are reported for investigation
without silently modifying the dataset. PASS establishes structural and referential
consistency; it cannot establish that every supplied label is semantically correct.

The Stage 0 all-empty baseline is analytically the singleton fraction. Stage 1 now
independently verifies it using the general per-entity scorer.

## Reproduce Stage 1

Run Stage 0 first. Stage 1 checks the CRC and byte size of the current S1/ground-truth
files against the archive inventory from the passing audit, then records SHA256 hashes.
It reuses Stage 0's finding that no positive target belongs to multiple S1 entities.
It reads only training inputs; it does not inspect test records.

```powershell
python -X utf8 -m unittest discover -s code/business_entity_resolution/src -p "test_stage1.py" -v
python -X utf8 code/business_entity_resolution/src/stage1.py --output-dir reports/stage1-rerun --seed 20260925 --validation-fraction 0.2
```

The initial artifact directory is `reports/stage1`. Use a fresh output directory to
rebuild. Stage 1 uses only Python's standard library; no additional dependencies.

- `splits.tsv`: exactly one row per training S1, its country and true match count,
  plus `grouped`, `transfer_01`, and `transfer_02` role columns on the supplied data.
- `splits.sqlite`: bounded-memory access to each entity and its entire positive set.
- `manifest.json`: seed, split definitions, input fingerprints, split TSV fingerprint.
- `report.json` / `report.md`: fold sizes, singleton rates, per-country baselines,
  disjointness checks, and per-stratum counts.

The grouped holdout is stratified by country and exact match count. Stable SHA256
ordering of `[seed, entity_id]` makes assignments independent of input row order.
Within each stratum, the requested validation count is rounded half up, with at least
one entity per role when possible; a one-entity stratum remains in training.

Country modes are generated from sorted observed labels, with no country whitelist:
for this dataset, `transfer_01` is India → US and `transfer_02` is US → India. Each
transfer uses all S1s from its training country and all S1s from its validation country,
independently of the grouped split. Additional countries would get additional modes;
an entity outside a particular transfer's two countries is explicitly `excluded`.

Use `splits.iter_entities(db, modes, mode, role)` to load a selected fold. It yields
`(S1 ID, country, complete comma-separated positive IDs)`; never split individual
positive pairs independently. Fit pair models only on training S1s and learned text
transformations only on the training partition. Under strict country transfer, fit
learned transformations on training-country records only, then transform validation
without refitting. The shared candidate pool can retain distractors; their presence
does not authorize fitting on validation S1 labels. Later grouped CV/OOF training must
retain the same S1 grouping rule.

## Score local validation predictions

```powershell
python -X utf8 code/business_entity_resolution/src/score_predictions.py --baseline empty --mode all
python -X utf8 code/business_entity_resolution/src/score_predictions.py --baseline perfect --mode grouped
python -X utf8 code/business_entity_resolution/src/score_predictions.py --predictions path/to/validation_predictions.tsv --mode grouped --role validation --output reports/validation_metrics.json
python -X utf8 code/business_entity_resolution/src/score_predictions.py --predictions path/to/us_predictions.tsv --mode transfer_01 --role validation
```

`--split-dir` defaults to `reports/stage1`; change it when evaluating a rerun.
Prediction TSVs use exactly `source1_entity_id` and `matched_entity_ids`. They must
cover exactly the selected evaluation population, including empty predictions, in
any row order. Missing/extra S1 rows, duplicate rows or IDs within lists, malformed
columns, wrong source prefixes, and padded IDs are errors. An empty evaluation
population is an error. A well-formed predicted target absent from the true set is
a false positive. This local metric tool does not check target membership in S2/S3;
the stricter submission ID-existence validator is still required in Stage 8.

Scores are computed per S1 and then averaged equally using compensated summation:
`5 TP / (5 TP + 4 FP + FN)` for nonempty truth. Empty truth scores 1 only when the
prediction is also empty. `false_singleton_rate` means true singletons incorrectly
given any match, divided by true singletons; it is null if there are no true singletons.
`missed_non_singleton_rate` separately measures non-singletons predicted empty.
Every score also reports country slices. Perfect-label baselines are oracle checks,
not model results. Validation used for threshold tuning is not an unbiased final
estimate; later decision models require grouped OOF predictions/nested evaluation.

## Reproduce Stage 2

```powershell
python -X utf8 -m unittest discover -s code/business_entity_resolution/src -p "test_stage2.py" -v
python -X utf8 code/business_entity_resolution/src/stage2.py --workers 4 --artifact-dir artifacts/stage2-rerun --report-dir reports/stage2-rerun
```

The initial output directories are `artifacts/stage2` and `reports/stage2`. Both rerun
directories must be empty/fresh. Stage 2 uses only the standard library. Four local
worker processes handle independent source files with bounded memory. Lower `--workers`
if needed. No learned transformation, API, lookup, transliteration service, or label
fitting is involved. Country labels never select normalization or extraction rules.

All six normalized source TSVs preserve the original folder/file names, with 25 columns.
The first four columns remain `entity_id`, `business_name`, `business_address`, `country`
with exactly the original field values and row order. The original dataset is untouched.
New columns are:

| Columns | Meaning |
|---|---|
| `name_clean` | NFKC, case folding, decimal-digit canonicalization, punctuation and whitespace cleanup, `&` → `and`; no legal-word replacement |
| `name_normalized` | Basic name plus recognized trailing legal-suffix canonicalization; suffixes are expanded, not deleted |
| `address_normalized` | Same basic text cleanup; no ambiguous street-abbreviation expansion |
| `address_numbers` | Standalone numeric expressions, retaining supported slash/range/fraction structures and letter suffixes |
| `house_numbers`, `house_number_candidates`, `house_number_source` | Accepted premise-number hypotheses, broader candidates, and extraction provenance |
| `street_numbers` | Street ordinals and explicitly numbered streets, kept distinct from premise numbers |
| `postal_codes`, `postal_candidates`, `postal_code_source` | Accepted numeric-postcode hypotheses, broader candidates, and extraction provenance |
| `name_missing`, `address_missing` | Raw field is blank/whitespace; literal `NA` and `null` strings are not silently declared missing |
| `name_null_like`, `address_null_like` | Whole-field null-like string diagnostics; raw and normalized strings are retained |
| `name_normalized_empty`, `address_normalized_empty` | Cleaned view is empty, including punctuation-only inputs |
| `house_number_missing`, `postal_code_missing` | No accepted extracted value; this does not imply a real-world absence or a conflict |
| `house_number_ambiguous`, `postal_code_ambiguous` | Multiple possible values or candidates without an accepted value; zero is not a guarantee of correctness |

List-valued evidence fields use JSON arrays inside TSV cells, with `[]` for no values.
Flags use `0`/`1`. Read TSVs with `sep="\t"`, string dtypes, and automatic NA conversion
disabled; use `json.loads` for evidence lists. Ordinary TSV quoting preserves tabs,
newlines, and quotes inside the raw fields.

Case folding/NFKC preserve accented names and non-Latin combining marks; no accent
stripping or Latin-only filtering occurs. Unicode decimal digits become ASCII in
normalized text, with zero padding preserved; raw digits remain untouched. Numeric
punctuation is captured in evidence before the text views replace punctuation with
spaces. Known suffix variants normalize only in the trailing legal-token sequence,
so words inside distinctive names are not globally rewritten. Unrecognized suffixes
remain in the text. These conservative views do not resolve cross-script transliterations.

House provenance is `explicit_premise_label`, `leading_number_heuristic`, or empty.
Postal provenance is `explicit_postal_label`, `trailing_numeric_heuristic`, or empty.
Supported postal shapes are 5 digits, 6 digits, and 5+4 digits, independently of country.
Alphanumeric postal formats remain available in text but are not extracted as numeric
codes. House/building/plot labels, address order, and numeric shape are fallible cues:
later pair features must retain provenance and ambiguity. Never treat a missing value
as disagreement, and never interpret a heuristic as verified address data.

Every source's size and CRC must match Stage 0. Every row checks digit preservation
before writing. Each full output is re-read to validate its schema, boolean flags, row
count, and order-sensitive SHA256 digest of the four raw fields. Raw missingness and
country distributions must equal Stage 0. Files have a `.partial.tsv` suffix until
verification passes; a failed/incomplete run must not be consumed downstream.

`reports/stage2/report.md` and `report.json` report exact extraction coverage, missingness,
country slices, source provenance counts, and deterministic examples of ambiguous
evidence. Coverage is not extraction accuracy: no structured-address gold annotations
are available. Model scores and candidate recall belong to later stages. Normalized
TSVs are intermediate artifacts, not submission files.

## Submission requirements retained for later stages

The final package must include both output TSVs, this runnable package, and the filled
`Documentation_template.md`. The provided validator skips target existence by default
and treats missing candidates/subset violations as warnings. Stage 8 must run it with
`--check-ids` and also enforce both required files and strict match-subset consistency.
The supplied PDF's exact F_0.5 formula is authoritative: its denominator weights a false
positive four times a false negative, despite its informal description of "2x" precision.


## Stage 3: disk-backed candidate retrieval

Run from the repository root with Python 3.10+ and SQLite 3.34+ compiled with
FTS5/trigram support. Stage 3 itself uses only the Python standard library.
Dependencies for the entire package are pinned in `requirements.txt`.
The previously discovered Stage 4?6 files and `run_pipeline.sh` are unvalidated
drafts: do not use them to skip the stage review gates.

```powershell
python -m unittest discover -s code/business_entity_resolution/src -p test_stage3.py -v
python -u code/business_entity_resolution/src/stage3.py --limit 1000 --output-dir reports/stage3/grouped_pilot
```

The first run builds reusable indexes over **all** 2,206,821 training S1,
5,034,616 S2, and 5,285,603 S3 records. Indexing and retrieval resume from atomic
batch checkpoints when the identical command is repeated. Source files and
configuration must remain unchanged. A changed retrieval configuration needs a
new output directory; source changes need fresh indexes. OS-held locks reject
concurrent writers to the same run/index. Interrupted IDF fitting also resumes.

`--limit 1000` selects validation S1 entities deterministically by the Stage 1
split hash. It never truncates either candidate pool. This is a **pilot**, not
full-validation recall. `--limit 0` selects all entities in the requested role.

```powershell
python -u code/business_entity_resolution/src/stage3.py --limit 0 --output-dir reports/stage3/grouped_validation
python -u code/business_entity_resolution/src/stage3.py --role train --limit 0 --output-dir reports/stage3/grouped_train
python -u code/business_entity_resolution/src/stage3.py --mode transfer_01 --limit 1000 --output-dir reports/stage3/transfer_01_pilot
python -u code/business_entity_resolution/src/stage3.py --mode transfer_02 --limit 1000 --output-dir reports/stage3/transfer_02_pilot
```

These are separate runs; running a command is not evidence that its report has
completed. Transfer modes refit IDF exclusively on the training country. Grouped
IDF is fitted exclusively on training-role S1s. Candidate-pool posting counts are
unlabeled index statistics used to select retrieval anchors. No country filter
is applied to queries or candidate pools; country labels remain open strings.

Five routes run independently against S2 and S3 and are unioned per S1:

- Name and address character-trigram routes use rare shared trigram anchors.
- Combined retrieval unions those shortlists and adds cross-field conjunctions.
- Rare-token retrieval includes tokens with document frequency **one**.
- Component retrieval uses postal equality and house-number plus text-token
  intersections. These are heuristic retrieval hints, not verified matches.

All routes rerank bounded shortlists with actual L2-normalized TF-IDF cosine:
raw trigram counts times `log((training_documents+1)/(document_frequency+1))+1`.
Unseen trigrams use document frequency zero. Combined vectors sum name and
address trigram counts. This is **approximate retrieval**, not a global exact
nearest-neighbor scan. Posting caps and shortlist caps can lose true neighbors;
measure their effect using the recall report. No neural model or external data
is used at this stage.

Tune `--topk-name`, `--topk-address`, `--topk-combined`, `--topk-rare`, and
`--topk-components` (default 20 per route **per source**). Zero disables the
route's output. Tune `--anchors` (8), `--postings-per-anchor` (96), `--shortlist`
(96), `--rare-max-df` (100), and `--component-max-df` (5000) separately. Posting
lists and equal-score ties are resolved deterministically. Numeric agreement
never automatically accepts a match.

Outputs in each run directory:

- `config.json`: exact retrieval parameters and input fingerprints.
- `candidates.sqlite`: query manifest, per-source completion flags, unique union
  pairs, and JSON route metadata preserving each rank and cosine score.
- `candidate_pairs.tsv`: one row for every evaluated **training** S1, including
  empty sets. This is a Stage 3 artifact, not a test submission.
- `per_entity_recall.tsv`: every evaluated S1, true/found counts, candidate count,
  recall (blank for true singletons), and perfect-classifier F0.5 ceiling.
- `report.json`: scope and denominator, macro recall over non-singletons,
  per-country and per-route recall, micro recall, all-empty baseline, reduction
  against the complete S2+S3 cross product, and oracle macro F0.5 ceiling.

Macro candidate recall excludes true singletons, whose recall is undefined.
The oracle ceiling includes them with score one by assuming perfect rejection
of all false matches. It is not a trained-model validation score. Macro recall
and macro F0.5 both weight S1 entities equally. Later stages must preserve the
exact candidate set actually fed into the final inference model in the final
submission's `candidate_pairs.tsv`; early blocking output alone is insufficient.

Stop after reviewing Stage 3 results. If recall is weak, inspect missed true
pairs and increase route/shortlist budgets on a separate output run before
training a classifier. Do not compensate for missing candidates by lowering
acceptance thresholds.

To inspect missed true pairs and compare smaller equal per-route budgets after a
completed run:

```powershell
python code/business_entity_resolution/src/analyze_stage3.py --run-dir reports/stage3/grouped_pilot
```

`diagnostics.json` reports source-specific entity-macro recall and budget costs;
`missed_pairs.tsv` contains the normalized text of missing gold pairs for review.
Comparisons cannot evaluate a larger budget than the one retrieved initially.
