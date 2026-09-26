# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve business entities with a classic **blocking + learned-matcher** pipeline:
a fast approximate-retrieval stage narrows ~10M Source 2/3 records to a few hundred
candidates per Source 1 entity, then a gradient-boosted classifier (CatBoost) scores
each candidate pair and a precision-tuned threshold produces the final matches. Every
component is **country-agnostic**, so the pipeline handles the unseen test country
(France) with no special casing. All decisions are driven by the exact competition
metric — macro F₀.₅ scored per Source 1 entity over the full population, including
singletons.

---

## 2. Methodology

### 2.1 Problem Analysis

Key observations from a full-data audit (no sampling):
- Names carry heavy noise: legal-suffix variation (Ltd/Limited, Pvt/Private),
  word-order transposition ("L.L.C. Emerald" vs "Emerald L.L.C."), punctuation
  (`&` vs `+`), typos ("Emera1d"), and **transliteration** (Latin ↔ Devanagari).
- Addresses vary by abbreviation (Rd/Road), reordering, missing PIN/state, and
  landmark references; house numbers appear in multiple formats.
- ~2–5% of entities are singletons (no true match); the metric rewards predicting
  those as empty, and penalizes false merges 4× more than misses.
- Country is an **open label** — training is US + India, test adds France.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage retrieve-then-match).
**Core Innovation:** the competition metric is the single source of truth end to
end — scoring is set-based per entity over the full population (so singletons and
matches lost in retrieval are charged correctly), and the decision threshold is
selected on that exact metric on a held-out split, on the same probability scale
used at inference.

---

## 3. Candidate Generation (Blocking)

For each Source 1 entity we retrieve candidates from Source 2 and Source 3 using an
inverted index (SQLite FTS5, trigram + word tokenization) with **five complementary
routes**, then take their union:

- **name** — trigram TF-IDF cosine over normalized business name
- **address** — trigram TF-IDF cosine over normalized address
- **combined** — name + address together (the strongest single route)
- **rare** — shared low-document-frequency tokens (distinctive words)
- **components** — matching house numbers / postal codes

- **Blocking keys:** approximate trigram anchors + FTS posting lists, reranked by
  train-fitted IDF cosine; per-route top-k budgets bound the candidate count.
- **Candidate pairs generated:** written to `candidate_pairs.tsv` (one row per S1).
- **Ensuring recall:** multiple independent routes are unioned so a match missed by
  one signal (e.g. a corrupted name) can still be found by address or rare tokens;
  blocking recall and the oracle F₀.₅ ceiling are measured on a validation split.

---

## 4. Matching Model

**Features used** (~60, all country-agnostic; `features.py`):
- **Name:** rapidfuzz WRatio / partial / token-set ratios, token Jaccard,
  character-3gram Jaccard, length ratios, empty-field flags.
- **Address:** the same similarity family over normalized addresses.
- **Numeric:** house-number and postal-code agreement / conflict / missingness.
- **Rare tokens:** shared distinctive-token counts and fractions.
- **Retrieval evidence:** per-route cosine score and inverse rank, number of routes
  that fired, source flag (S2/S3).
- **Interactions:** e.g. strong-name-weak-address, house-agree-and-name-strong.

**Model type:** CatBoost gradient-boosted trees (Apache-2.0, well under 8B params),
trained with entity-grouped cross-validation so all pairs of one S1 stay in the same
fold (no leakage).
**Threshold selection:** macro F₀.₅ swept on a held-out validation split over the
full population; ties broken toward the higher (more precise) threshold, consistent
with F₀.₅'s 4× precision weighting. Threshold and inference use the same raw
probability scale.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro, held-out validation):** [fill from full-scale validation run]
- **Blocking recall / oracle F₀.₅ ceiling:** [fill from full-scale run]
- **Common false positives (wrong merges):** same-name different-branch records
  (e.g. "Rogers Corporation" vs "Rogers Corporation Services") and same-address
  different-business records.
- **Common false negatives (missed matches):** entities where the name is heavily
  corrupted *and* the city/address diverges, leaving too little shared signal.

---

## 6. Conclusion

A precision-first retrieve-then-match pipeline, with the exact F₀.₅ metric wired into
both threshold selection and evaluation, resolves noisy multi-source business records
and generalizes to an unseen country without any external data. The main lever for
further gains is reducing false merges on singletons and near-duplicate branches.

---

## Appendix

### A. Code Artefacts

Runnable pipeline in `code/business_entity_resolution/` (all source in `src/`):

- `stage0`–`stage2`: audit, exact scorer + splits, country-agnostic normalization.
- `stage3` / `blocking_index` / `blocking_retrieval`: index + candidate retrieval
  (training/validation); `stage3_inference`: the same for the unlabeled test set.
- `stage4` + `features`: pairwise feature matrix (+ population and truth for scoring).
- `stage5`: CatBoost training, grouped CV, threshold selection.
- `stage6`: predictions → `matching_results.tsv` + `candidate_pairs.tsv`.
- `scoring` / `evaluation`: the exact macro-F₀.₅ metric (single source of truth).

Reproduction and exact run commands are in `code/business_entity_resolution/README.md`.

### B. Additional Results

*[Add blocking recall by country, threshold sweep curve, feature importances from the
full-scale run.]*

---

**Fair play:** the pipeline uses only the provided data — no external databases, APIs,
geocoding, or internet augmentation of any kind.
