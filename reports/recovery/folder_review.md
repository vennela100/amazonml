# Folder review and resumption point

The next unfinished stage is **Stage 3: candidate generation**. Stages 0–2 have
completed reports and artifacts. The existence of Stage 4–6 source files does not
establish completion of those stages.

## Verified completed work

| Stage | Evidence | Result |
|---|---|---|
| 0: audit | `reports/stage0/audit.json` and `report.md` | All seven input TSVs audited; all-empty macro F_0.5 = 0.0558482088 |
| 1: scorer and splits | `reports/stage1/report.json`, `splits.sqlite`, `splits.tsv`, `file_scorer_check.json` | 1,765,458 training and 441,363 validation S1 entities; exact scorer and country-transfer modes |
| 2: normalization | `reports/stage2/report.json`, six per-file reports, six complete `artifacts/stage2/{train,test}/*.tsv` files | 24,229,173 rows normalized and re-read; raw fields/order, country counts, missingness, and digits preserved |

Stage 2's original process completed with exit code 0. All 15 Stage 2 tests were
rerun successfully during this review. All 10 existing Stage 3 tests also pass;
they do not cover the production-scale issues below.

Accepted house/premise hypotheses occur in 16,633,006 records. Accepted numeric
postcode hypotheses occur in only 1,376 records. These are extraction coverage
counts, not accuracy estimates. Keep raw text, ambiguity flags, and provenance.

## Running work observed during this review

Two Python processes were running `src/stage3.py`, both with input
`artifacts/stage2`, `--split train`, `--sources s2,s3`, routes
`name,address,rare`, and `--report-dir reports/stage3 --overwrite`:

| PID at inspection | Hash features | Query chunk size | Approximate resident memory |
|---|---:|---:|---:|
| 24488 | 150,000 | 50,000 | 2.0 GiB |
| 34284 | 131,072 | 200,000 | 2.0 GiB |

No completed full-data Stage 3 report or candidate database was found. These jobs
can collide when writing their shared output destination. The current on-disk
implementation persists candidates only after processing all routes for a source.
Process start times precede some code modifications, so neither process should be
assumed to execute exactly the current file contents. This review did not stop them.

Stopping them requires a decision because uncheckpointed computation would be lost.
The requested recommendation is to stop both and resume with one corrected,
checkpointed run in a fresh directory.

## Stage 3 defects to fix before a new full run

1. `build_hash_vectorizer()` returns L2-normalized hashed character counts. There
   is no inverse-document-frequency weighting, despite the TF-IDF function names.
2. `tfidf_blocking()` retains global candidate heaps, and `process_source()` retains
   the full union in nested Python dictionaries. Memory therefore grows with the
   number of entities and candidates, not just the two sparse chunks. Query text
   is re-vectorized for every pool chunk.
3. Postal/address-component blocking is absent. The active commands also omit
   combined name/address retrieval. All five requested routes remain necessary.
4. `rare_token_blocking()` excludes document-frequency-one tokens, losing the
   most distinctive available evidence. Its union is not capped per query, and
   only name tokens are considered.
5. Route-specific ranks and scores are discarded in favor of a single maximum
   score and route-name list. Downstream retrieval features need the original
   evidence per route.
6. `candidate_meta` is replaced when another source is saved, so it does not
   represent the final S2/S3 union. Entities without candidates are omitted.
7. Recall evaluation loads candidates into memory and assumes the entire selected
   split was queried. It does not verify an explicit query-population manifest.
   Report macro recall for non-singletons separately from any convention that
   assigns singleton recall 1, and save per-S1 results including zero-candidate rows.
8. The smoke test uses only the first 5,000 S1 and first 20,000 records per candidate
   source, but evaluates against all 441,363 validation S1 entities. Its resulting
   recall is not a valid full-data estimate. A valid pilot needs explicitly matched
   query/evaluation scope and a declared candidate pool; truncated-pool coverage
   must be separated from retrieval misses.

## Later drafts: recorded for their own review gates

- `stage5.py:macro_f05` and `stage6.py:compute_metrics` skip entities with no
  positive candidate labels. This violates singleton scoring and omits entities
  and true matches lost during candidate generation. Reuse the Stage 1 scorer
  against complete ground truth and the complete evaluation population.
- Stage 5 explicitly leaves honoring the Stage 1 train/validation roles as a TODO
  and currently uses all supplied labeled pairs. Do not treat its proposed metrics
  as held-out results for the existing split.
- Stage 5 chooses a threshold on raw OOF probabilities; Stage 6 applies a calibrator
  before using that threshold. Threshold and inference scores must use the same scale.
- Stage 6's submission writer produces headerless S1/S2-or-S3 pair rows, omits empty
  entities, and does not produce the two required one-row-per-S1 output TSVs. Its
  default feature input is the labeled training artifact, not a test inference pipeline.
- `run_pipeline.sh` calls modules that have no matching CLI, points to the wrong
  dataset root for this workspace, and runs past the requested review gates. It is
  not a validated end-to-end reproduction command.
- `requirements.txt` was changed from exact version pins to ranges. Restore tested
  exact pins when repairing the runnable package.

## Concrete resumption sequence

1. Resolve the two existing jobs before allowing another writer to their directory.
2. Preserve completed Stage 0–2 artifacts and the existing draft code.
3. Repair Stage 3 with real IDF weighting, all retrieval routes, bounded candidate
   budgets, retained route metadata, persisted query coverage, and checkpoints.
4. Test deterministic retrieval, rare tokens, empty entities, route unions, and
   evaluation scope. Run a declared pilot against the full candidate pools to
   establish memory, runtime, and recall before committing to all queries.
5. Produce per-S1 recall, country slices, and reduction ratios, then pause at the
   Stage 3 review gate. Do not launch the later training/submission drafts yet.
