# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Code Queens
**Team Members:** (Madanapalle Institute of Technology and Science (MITS), Madanapalle, Andhra Pradesh)
- ALAMUR VENNELA (Team Leader)
- Nara Rajani
- Gollapalli Rasmitha
- Korivi Vijaya Lakshmi

**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We treat the problem as an **assignment problem from the pool side**: every Source 2 / Source 3 record is attached to *at most one* Source 1 entity (or none). A fast, country-partitioned TF-IDF blocking stage retrieves the top-10 Source 1 candidates for every pool record (96.4% recall on training), a LightGBM pair classifier scores each candidate with ~50 string, numeric and candidate-context features, and each pool record is assigned to its best-scoring candidate only if the probability clears a threshold tuned directly for macro F0.5. Held-out macro F0.5 on 65,900 unseen training Source 1 entities: **0.9349**; public leaderboard F0.5: **0.9355**.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA on the full training set (2.21M S1, 5.03M S2, 5.29M S3 records):

- **Each pool record belongs to at most one S1 entity.** The 7,638,365 matched ids in the ground truth are all distinct, so the task is a many-to-one assignment, not arbitrary pairwise linking. About 26% of pool records match no S1 at all; they are *distractor* businesses.
- **Singletons are rare (5.6%)**, and matched S1 entities have 3.5 pool records on average (up to 11).
- **Name noise:** typos ("Svl Ttuaesb Pvt Ltd"), word transpositions, legal suffix changes (Pvt/Private/Ltd), punctuation, website/handle forms (`bestinfotech.com`, `@jexfirst`), Devanagari/Tamil transliterations (≈9–11% of pool names are non-Latin), and completely unrelated trade names ("Dovacira") at the correct address.
- **Address noise:** abbreviations (St/Street/"SAINT", Rd, Dr, R./Rue, BD), component reordering, missing components, leading zeros in house numbers ("053" vs "53"), state names in native script, and ~3% empty addresses.
- Because names can be totally different, **address evidence alone must be able to carry a match**, and vice versa for empty addresses.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier + constrained (one-S1-per-pool-record) assignment.
**Core Innovation:** Inverting the query direction, so each pool record picks its best S1. This enforces the data's many-to-one structure, removes the double-assignment false merges that per-S1 retrieval produces, and gives the classifier "competition" features (margin to the second-best S1, S1-side rank) that are very strong precision signals for F0.5.

---

## 3. Candidate Generation (Blocking)

- **Normalization** (`src/prep.py`): lowercase, accent stripping for Latin scripts (Indic vowel signs are preserved), `&`→`and`, legal-suffix stop-words removed from name tokens, address abbreviation map (st→street, rd→road, r→rue, bd→boulevard, …), house numbers stripped of leading zeros, and alphanumerics split (`177b`→`177`,`b`).
- **Blocking tokens** per record: name tokens (`n:`), concatenated-name tokens (`c:`) that match domains/handles, address unigrams (`a:`), and address bigrams within a comma component (`b:`, e.g. `920_stinocher`), which are highly discriminative.
- **Retrieval** (`src/block.py`): per country (country is an open label, no hard-coding; France is handled identically), a sublinear TF-IDF over all tokens, dropping tokens with document frequency > 3000 (non-discriminative, and they blow up the product). Then a multithreaded sparse top-n matrix product (`sparse_dot_topn`) returns the **top-10 S1 records by cosine for every pool record**.
- **Candidate pairs generated:** 101.2M on train and 95.8M on test (≈10 per pool record, versus 2.2M × 10.3M ≈ 2.3×10¹³ possible pairs).
- **Recall on training** (fraction of matched pool records whose true S1 is among their candidates): recall@1 0.921, @3 0.949, @5 0.957, **@10 0.964**.
- `candidate_pairs.tsv` lists, for each S1, exactly the pool records for which it was one of the 10 candidates. This is the full set scored by the model; every matched id is a subset of it.

---

## 4. Matching Model

**Features (≈50, `src/features.py`):**

- *Name:* RapidFuzz ratio, token-set, token-sort, partial ratio, Jaro-Winkler; similarity of space-free concatenations (catches domains/handles); token overlap counts and coverage on both sides; IDF-weighted name cosine; lengths; non-Latin-script flag.
- *Address:* ratio, token-set, token-sort, partial; token overlap/coverage; IDF-weighted address cosine; **house/plot number features**: shared-number count, Jaccard, conflict flag (both have numbers but none shared), subset flag, first-number equality.
- *Candidate context* (computed over the full candidate set): blocking cosine, rank, gap to the pool record's best candidate, top-1 vs top-2 margin, number of candidates, how many pool records pick this S1 at rank 0 and their summed score, rank of this pair among all pairs pointing at the same S1, and "relative-to-best" versions of key similarities (feature minus the max over the pool record's candidates).
- Country is intentionally **not** a feature, so the model transfers to France, which is unseen in training.

**Model type:** LightGBM binary classifier (num_leaves 127, lr 0.05, feature/bagging fraction 0.8, 700 trees), trained on 8.9M candidate pairs from 900k pool records disjoint from the evaluation entities.

**Decision rule / threshold selection:** each pool record → its highest-probability candidate, kept only if p ≥ t. t is chosen by sweeping and computing the **exact competition metric** (macro F0.5 over S1 entities, singletons scored 1/0) on a held-out set of 3% of training S1 entities (65,900) using all pool records that could be assigned to them.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, held-out training S1 entities):** 0.9349 (precision 0.974, recall 0.935)
- Threshold sweep: F0.5 is flat between t≈0.22 and 0.34 (chosen t = 0.28) and drops slowly beyond it; high thresholds trade recall for precision faster than F0.5 rewards.
- **Held-out error breakdown** (227,662 true pool→S1 links of the evaluation entities): 11,422 lost at blocking, 2,367 scored below threshold, 1,060 assigned to the wrong S1; 5,708 false links, 4,197 of them from distractor records.
- **Model iterations:** v1 (6M pairs, 500 trees, no IDF-cosine / S1-rank features) 0.9330 → v2 (8.9M pairs, 700 trees, + IDF-weighted name/address cosine, S1-side rank) **0.9349**.
- **Common false positives (wrong merges):** distractor records that are near-copies of an S1 business with a slightly different house number on the same street ("22911 johnson road" vs "22902 johnson road", "144 mcauley court" vs "141"), and records at a shared commercial building address.
- **Common false negatives:** mostly records with an **empty address** whose name is shared by several S1 entities in different cities (e.g. "miller foods inc"). These are genuinely ambiguous, and leaving them unmatched is the F0.5-optimal choice. Also random trade names at a shared building address ("umbrafayearc" at an address hosting several S1 businesses), and non-Latin-script names with partial addresses.

---

## 6. Conclusion

Exploiting the many-to-one structure (assign each pool record to one S1) turned a hard pairwise problem into a ranking-plus-threshold problem where candidate competition features give very strong precision. Cheap, high-recall sparse TF-IDF blocking with address bigrams and a gradient-boosted matcher over fuzzy, numeric and context features reach ~0.93+ macro F0.5 on held-out data, with a full-test runtime of roughly 1.5 hours on a 12-thread, 16 GB laptop.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| File | Role |
|---|---|
| `src/prep.py` | Load TSVs, normalize names/addresses, build blocking tokens (per source) |
| `src/block.py` | Per-country TF-IDF + sparse top-10 retrieval → candidate pairs |
| `src/features.py` | Pair features (RapidFuzz, sparse token/number overlaps, candidate context) |
| `src/train.py` | Label candidates from ground truth, train LightGBM, tune threshold on exact macro F0.5 |
| `src/predict.py` | Score test candidates, assign, write `matching_results.tsv` and `candidate_pairs.tsv` |
| `src/errors.py` | Error analysis on held-out predictions |
| `run_pipeline.sh` | End-to-end entry point: `bash run_pipeline.sh <dataset_dir> <work_dir> <output_dir>` |

No external data, APIs or pretrained models are used; all libraries are MIT/BSD/Apache-2.0.

### B. Additional Results

Blocking recall@k on training: @1 0.9206, @2 0.9409, @3 0.9490, @5 0.9566, @10 0.9640.
