"""Stage 3: Candidate generation via multi-route TF-IDF blocking.

Memory-aware design for 16GB RAM machine with ~3-4 GB available:
- Process one source pool (S2 or S3) at a time
- Use max_features to cap TF-IDF vocabulary
- Process S1 queries in chunks for sparse_dot_topn
- Store candidates in SQLite, not in memory
- Evaluate recall against ground truth from splits database
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from scoring import parse_ids

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Column indices in the 25-column normalized TSV
COL_ID = 0
COL_NAME_NORM = 5
COL_ADDR_NORM = 6
COL_COUNTRY = 3

NORM_HEADER = [
    'entity_id', 'business_name', 'business_address', 'country',
    'name_clean', 'name_normalized', 'address_normalized',
    'address_numbers', 'house_numbers', 'house_number_candidates',
    'house_number_source', 'street_numbers', 'postal_codes',
    'postal_candidates', 'postal_code_source',
    'name_missing', 'address_missing', 'name_null_like', 'address_null_like',
    'name_normalized_empty', 'address_normalized_empty',
    'house_number_missing', 'house_number_ambiguous',
    'postal_code_missing', 'postal_code_ambiguous'
]


def stream_tsv(path, columns):
    """Yield selected columns from a normalized TSV, streaming."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if header != NORM_HEADER:
            raise ValueError(f'{path}: unexpected header {header[:5]}...')
        for row in reader:
            yield tuple(row[c] for c in columns)


def load_pool(path):
    """Load pool IDs and normalized texts. Returns (ids, names, addresses, countries)."""
    ids, names, addresses, countries = [], [], [], []
    for entity_id, name, addr, country in stream_tsv(
            path, [COL_ID, COL_NAME_NORM, COL_ADDR_NORM, COL_COUNTRY]):
        ids.append(entity_id)
        names.append(name)
        addresses.append(addr)
        countries.append(country)
    log.info(f'Loaded pool: {len(ids):,} records from {Path(path).name}')
    return ids, names, addresses, countries


def load_s1(path):
    """Load S1 IDs and normalized texts."""
    return load_pool(path)


def build_hash_vectorizer(n_features=2**17):
    """Character n-gram hashing vectorizer — no fit, no vocabulary, constant memory.

    Uses HashingVectorizer with L2 norm so vectors are unit-length for cosine.
    n_features=131072 gives low collision rate for char 3-5 grams.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import Normalizer
    return HashingVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        n_features=n_features,
        norm='l2',          # unit vectors -> cosine = dot product
        alternate_sign=False,
        dtype=np.float32,
    )


def tfidf_blocking(s1_texts, pool_texts, top_k, chunk_size, route_name,
                   max_features=2**17, threshold=0.05):
    """Fully streaming hash-based blocking. Peak RAM = two small chunks only.

    Design:
    - For each (S1_chunk × pool_chunk) pair, compute cosine similarities.
    - Accumulate top-k per S1 entity across all pool chunks using a global dict.
    - Neither the full pool matrix nor the full S1 matrix is ever in memory.
    - Pool chunks = POOL_CHUNK rows each (~80 MB); S1 chunks = S1_CHUNK rows (~10 MB).
    """
    POOL_CHUNK = 300_000   # ~50 MB per pool chunk at 20 nnz/row × float32
    vec = build_hash_vectorizer(n_features=max_features)
    n_pool = len(pool_texts)
    n_s1 = len(s1_texts)
    log.info(f'Route {route_name}: streaming hash blocking '
             f'(pool={n_pool:,}, s1={n_s1:,}, n_features={max_features}, '
             f'pool_chunk={POOL_CHUNK:,}, s1_chunk={chunk_size:,})')

    # candidates[s1_idx] = sorted list of (score, pool_idx), max length top_k
    # We keep the list sorted ascending (smallest score first) for efficient merge.
    import heapq
    candidates: dict[int, list] = {}  # s1_idx -> [(score, pool_idx), ...]
    total_pairs = 0
    pool_offset = 0

    for p_start in range(0, n_pool, POOL_CHUNK):
        p_end = min(p_start + POOL_CHUNK, n_pool)
        pool_chunk_texts = pool_texts[p_start:p_end]
        pool_chunk_mat = vec.transform(pool_chunk_texts)  # (chunk, n_feat)
        log.info(f'Route {route_name}: pool chunk [{p_start:,}:{p_end:,}] '
                 f'nnz={pool_chunk_mat.nnz:,}, mem={pool_chunk_mat.data.nbytes/1024**2:.0f}MB')

        for s_start in range(0, n_s1, chunk_size):
            s_end = min(s_start + chunk_size, n_s1)
            s1_chunk_mat = vec.transform(s1_texts[s_start:s_end])

            # (s1_chunk, pool_chunk) cosine similarities, top-k per row
            sim = sp_matmul_topn(
                s1_chunk_mat,
                pool_chunk_mat.T,   # (n_feat, pool_chunk) CSC view
                top_n=top_k,
                threshold=threshold,
                n_threads=4,
                sort=True,
            )

            for i in range(s_end - s_start):
                row_s = sim.indptr[i]
                row_e = sim.indptr[i + 1]
                if row_e <= row_s:
                    continue
                s1_idx = s_start + i
                # Convert pool-chunk-local indices to global pool indices
                local_scores = sim.data[row_s:row_e]
                global_pool_idxs = sim.indices[row_s:row_e] + p_start

                if s1_idx not in candidates:
                    candidates[s1_idx] = []

                heap = candidates[s1_idx]
                for score, pidx in zip(local_scores.tolist(), global_pool_idxs.tolist()):
                    if score < threshold:
                        continue
                    total_pairs += 1
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, pidx))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, pidx))

            del s1_chunk_mat, sim
            gc.collect()

        del pool_chunk_mat
        gc.collect()

        if p_end % (POOL_CHUNK * 3) < POOL_CHUNK or p_end == n_pool:
            log.info(f'Route {route_name}: pool progress {p_end:,}/{n_pool:,}, '
                     f's1_with_cands={len(candidates):,}')

    # Convert heaps to sorted (descending score) lists
    result = {}
    for s1_idx, heap in candidates.items():
        result[s1_idx] = sorted([(pidx, score) for score, pidx in heap],
                                 key=lambda x: -x[1])

    n_cands = sum(len(v) for v in result.values())
    log.info(f'Route {route_name}: {n_cands:,} total candidates for '
             f'{len(result):,}/{n_s1:,} S1 entities')
    return result


def rare_token_blocking(s1_names, pool_names, max_df=100, max_candidates_per_token=500):
    """Find candidates sharing rare name tokens (df < max_df in pool)."""
    log.info(f'Building rare token index on {len(pool_names):,} pool names...')

    # Build token document frequency
    token_df = Counter()
    token_to_pool = defaultdict(list)
    for idx, name in enumerate(pool_names):
        tokens = set(name.split())
        for token in tokens:
            if len(token) >= 3:  # skip very short tokens
                token_df[token] += 1
                if token_df[token] <= max_candidates_per_token:
                    token_to_pool[token].append(idx)

    # Filter to rare tokens
    rare_tokens = {t for t, df in token_df.items() if 2 <= df <= max_df}
    rare_index = {t: pool_indices for t, pool_indices in token_to_pool.items()
                  if t in rare_tokens}
    log.info(f'Rare tokens: {len(rare_tokens):,} (df 2-{max_df})')

    del token_df, token_to_pool
    gc.collect()

    # Match S1 against rare tokens
    candidates = {}
    total = 0
    for s1_idx, name in enumerate(s1_names):
        tokens = set(name.split())
        matches = {}
        for token in tokens:
            if token in rare_index:
                for pool_idx in rare_index[token]:
                    if pool_idx not in matches:
                        matches[pool_idx] = 0.5  # fixed score for rare token match
                    matches[pool_idx] = max(matches[pool_idx], 0.5)
        if matches:
            candidates[s1_idx] = [(idx, score) for idx, score in matches.items()]
            total += len(candidates[s1_idx])

    log.info(f'Rare token route: {total:,} candidates for {len(candidates):,} S1 entities')
    return candidates


def merge_route_candidates(all_cands, new_cands, route_name, s1_offset=0):
    """Merge new candidates into the accumulator. Tracks routes and best score."""
    for s1_idx, pairs in new_cands.items():
        real_idx = s1_idx + s1_offset
        if real_idx not in all_cands:
            all_cands[real_idx] = {}
        for pool_idx, score in pairs:
            if pool_idx not in all_cands[real_idx]:
                all_cands[real_idx][pool_idx] = {'score': score, 'routes': [route_name]}
            else:
                entry = all_cands[real_idx][pool_idx]
                entry['score'] = max(entry['score'], score)
                if route_name not in entry['routes']:
                    entry['routes'].append(route_name)


def save_candidates_to_db(db_path, s1_ids, pool_ids, all_cands, source_label):
    """Save candidate pairs to SQLite in batches."""
    log.info(f'Saving candidates for source {source_label} to {db_path}...')
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA cache_size=-65536')
        db.execute('''CREATE TABLE IF NOT EXISTS candidates (
            s1_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            routes TEXT NOT NULL,
            best_score REAL NOT NULL,
            source TEXT NOT NULL
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS candidate_meta (
            s1_id TEXT NOT NULL,
            n_candidates INTEGER NOT NULL,
            routes_used TEXT NOT NULL,
            PRIMARY KEY (s1_id)
        )''')

        batch = []
        meta_batch = []
        for s1_idx, pool_entries in all_cands.items():
            s1_id = s1_ids[s1_idx]
            routes_used = set()
            for pool_idx, info in pool_entries.items():
                p_id = pool_ids[pool_idx]
                routes_str = ','.join(info['routes'])
                routes_used.update(info['routes'])
                batch.append((s1_id, p_id, routes_str, info['score'], source_label))
                if len(batch) >= 50_000:
                    db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?)', batch)
                    db.commit()
                    batch.clear()
            meta_batch.append((s1_id, len(pool_entries), ','.join(sorted(routes_used))))
            if len(meta_batch) >= 50_000:
                db.executemany('INSERT OR REPLACE INTO candidate_meta VALUES (?,?,?)', meta_batch)
                db.commit()
                meta_batch.clear()

        if batch:
            db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?)', batch)
        if meta_batch:
            db.executemany('INSERT OR REPLACE INTO candidate_meta VALUES (?,?,?)', meta_batch)
        db.commit()

    log.info(f'Saved {sum(len(v) for v in all_cands.values()):,} candidate pairs')


def load_ground_truth(splits_db_path):
    """Load ground truth from splits database. Returns dict[s1_id] -> set of matched IDs."""
    truth = {}
    with closing(sqlite3.connect(str(splits_db_path))) as db:
        for s1_id, matched_ids_str in db.execute(
                'SELECT id, matched_ids FROM entities ORDER BY id'):
            truth[s1_id] = parse_ids(matched_ids_str)
    log.info(f'Loaded ground truth for {len(truth):,} S1 entities')
    return truth


def evaluate_recall(db_path, splits_db_path, split_role='validation'):
    """Evaluate candidate recall against ground truth."""
    log.info(f'Evaluating recall for {split_role} entities...')

    # Load ground truth with split info
    truth = {}
    with closing(sqlite3.connect(str(splits_db_path))) as db:
        for s1_id, matched_ids_str, role in db.execute(
                'SELECT id, matched_ids, grouped_role FROM entities'):
            if role == split_role or split_role == 'all':
                truth[s1_id] = parse_ids(matched_ids_str)

    # Load generated candidates
    candidates = defaultdict(set)
    with closing(sqlite3.connect(str(db_path))) as db:
        for s1_id, cand_id in db.execute('SELECT s1_id, candidate_id FROM candidates'):
            if s1_id in truth:
                candidates[s1_id].add(cand_id)

    # Compute recall per entity
    total_entities = 0
    total_with_matches = 0
    total_true_matches = 0
    total_recalled = 0
    perfect_recall = 0
    zero_recall = 0
    recall_values = []
    per_country_stats = defaultdict(lambda: {
        'entities': 0, 'with_matches': 0, 'true_matches': 0,
        'recalled': 0, 'perfect': 0, 'zero': 0
    })

    # Load country info
    country_map = {}
    with closing(sqlite3.connect(str(splits_db_path))) as db:
        for s1_id, country, role in db.execute(
                'SELECT id, country, grouped_role FROM entities'):
            if role == split_role or split_role == 'all':
                country_map[s1_id] = country

    for s1_id, true_matches in truth.items():
        total_entities += 1
        country = country_map.get(s1_id, 'unknown')

        if not true_matches:
            # Singleton — recall is vacuously 1.0
            recall_values.append(1.0)
            perfect_recall += 1
            per_country_stats[country]['entities'] += 1
            per_country_stats[country]['perfect'] += 1
            continue

        total_with_matches += 1
        per_country_stats[country]['entities'] += 1
        per_country_stats[country]['with_matches'] += 1

        cand_set = candidates.get(s1_id, set())
        recalled = len(true_matches & cand_set)
        total_true_matches += len(true_matches)
        total_recalled += recalled
        per_country_stats[country]['true_matches'] += len(true_matches)
        per_country_stats[country]['recalled'] += recalled

        recall = recalled / len(true_matches)
        recall_values.append(recall)

        if recall == 1.0:
            perfect_recall += 1
            per_country_stats[country]['perfect'] += 1
        elif recall == 0.0:
            zero_recall += 1
            per_country_stats[country]['zero'] += 1

    # Compute statistics
    overall_recall = total_recalled / total_true_matches if total_true_matches else 0
    macro_recall = np.mean(recall_values) if recall_values else 0
    median_recall = np.median(recall_values) if recall_values else 0

    # Candidate count stats
    cand_counts = [len(candidates.get(s1_id, set())) for s1_id in truth]
    total_candidates = sum(cand_counts)

    result = {
        'split_role': split_role,
        'total_entities': total_entities,
        'entities_with_matches': total_with_matches,
        'total_true_matches': total_true_matches,
        'total_recalled': total_recalled,
        'overall_recall': overall_recall,
        'macro_recall': macro_recall,
        'median_recall': median_recall,
        'perfect_recall_entities': perfect_recall,
        'perfect_recall_fraction': perfect_recall / total_entities if total_entities else 0,
        'zero_recall_entities': zero_recall,
        'zero_recall_fraction': zero_recall / total_entities if total_entities else 0,
        'total_candidates': total_candidates,
        'avg_candidates_per_entity': total_candidates / total_entities if total_entities else 0,
        'per_country': {k: dict(v) for k, v in per_country_stats.items()},
        'recall_distribution': {
            'p10': float(np.percentile(recall_values, 10)) if recall_values else 0,
            'p25': float(np.percentile(recall_values, 25)) if recall_values else 0,
            'p50': float(np.percentile(recall_values, 50)) if recall_values else 0,
            'p75': float(np.percentile(recall_values, 75)) if recall_values else 0,
            'p90': float(np.percentile(recall_values, 90)) if recall_values else 0,
            'p99': float(np.percentile(recall_values, 99)) if recall_values else 0,
        }
    }

    log.info(f'Recall evaluation:')
    log.info(f'  Overall recall: {overall_recall:.4f}')
    log.info(f'  Macro recall: {macro_recall:.4f}')
    log.info(f'  Perfect recall entities: {perfect_recall:,}/{total_entities:,} '
             f'({perfect_recall/total_entities*100:.1f}%)')
    log.info(f'  Zero recall entities: {zero_recall:,}/{total_with_matches:,} '
             f'({zero_recall/total_with_matches*100:.1f}% of non-singletons)')
    log.info(f'  Total candidates: {total_candidates:,}')
    log.info(f'  Avg candidates/entity: {total_candidates/total_entities:.1f}')

    for country, stats in sorted(per_country_stats.items()):
        country_recall = stats['recalled'] / stats['true_matches'] if stats['true_matches'] else 1.0
        log.info(f'  {country}: recall={country_recall:.4f}, '
                 f'perfect={stats["perfect"]}, zero={stats.get("zero", 0)}')

    return result


def load_column(path, col_idx):
    """Load a single column from the normalized TSV. Returns list of strings."""
    values = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            values.append(row[col_idx])
    return values


def load_ids(path):
    """Load entity IDs only."""
    return load_column(path, COL_ID)


def process_source(s1_ids, s1_names, s1_addresses,
                   pool_path, source_label, db_path,
                   routes, top_k_name, top_k_address, top_k_combined,
                   chunk_size, max_features, threshold):
    """Run all retrieval routes for one source pool.

    Memory-optimized: loads only the columns needed for each route, frees
    them before the next route. Peak memory is one column of pool strings
    plus the TF-IDF matrix, not all columns at once.
    """
    log.info(f'\n{"="*60}')
    log.info(f'Processing source: {source_label} from {pool_path}')
    log.info(f'{"="*60}')

    # Always need pool IDs for candidate storage
    pool_ids = load_ids(pool_path)
    n_pool = len(pool_ids)
    log.info(f'Pool {source_label}: {n_pool:,} records')

    all_cands = {}  # s1_idx -> {pool_idx -> {score, routes}}

    if 'name' in routes:
        pool_names = load_column(pool_path, COL_NAME_NORM)
        cands = tfidf_blocking(
            s1_names, pool_names, top_k_name, chunk_size,
            f'{source_label}_name', max_features, threshold
        )
        merge_route_candidates(all_cands, cands, f'{source_label}_name')
        del cands
        # Keep pool_names for rare route if needed
        if 'rare' not in routes:
            del pool_names
        gc.collect()

    if 'address' in routes:
        pool_addresses = load_column(pool_path, COL_ADDR_NORM)
        cands = tfidf_blocking(
            s1_addresses, pool_addresses, top_k_address, chunk_size,
            f'{source_label}_address', max_features, threshold
        )
        merge_route_candidates(all_cands, cands, f'{source_label}_address')
        del cands
        # Keep pool_addresses for combined route if needed
        if 'combined' not in routes:
            del pool_addresses
        gc.collect()

    if 'combined' in routes:
        # Load whichever columns we don't have yet
        if 'name' not in routes or 'rare' not in routes:
            pool_names_for_combined = load_column(pool_path, COL_NAME_NORM)
        else:
            pool_names_for_combined = pool_names  # still in scope from name route
        if 'address' not in routes:
            pool_addresses_for_combined = load_column(pool_path, COL_ADDR_NORM)
        else:
            pool_addresses_for_combined = pool_addresses  # still in scope

        s1_combined = [n + ' ||| ' + a for n, a in zip(s1_names, s1_addresses)]
        pool_combined = [n + ' ||| ' + a for n, a in
                         zip(pool_names_for_combined, pool_addresses_for_combined)]
        cands = tfidf_blocking(
            s1_combined, pool_combined, top_k_combined, chunk_size,
            f'{source_label}_combined', max_features, threshold
        )
        merge_route_candidates(all_cands, cands, f'{source_label}_combined')
        del cands, s1_combined, pool_combined, pool_names_for_combined
        try:
            del pool_addresses_for_combined
        except NameError:
            pass
        gc.collect()

    if 'rare' in routes:
        if 'name' not in routes:
            pool_names = load_column(pool_path, COL_NAME_NORM)
        # pool_names still in scope from name route (we kept it)
        cands = rare_token_blocking(s1_names, pool_names)
        merge_route_candidates(all_cands, cands, f'{source_label}_rare')
        del cands, pool_names
        gc.collect()

    # Save to DB
    save_candidates_to_db(db_path, s1_ids, pool_ids, all_cands, source_label)

    n_cands = sum(len(v) for v in all_cands.values())
    n_s1_with_cands = len(all_cands)
    brute_force = len(s1_ids) * n_pool
    reduction = 1 - n_cands / brute_force if brute_force else 0

    log.info(f'Source {source_label}: {n_cands:,} candidates for {n_s1_with_cands:,} S1 entities')
    log.info(f'Reduction ratio: {reduction:.6f} ({n_cands:,} / {brute_force:,})')

    del all_cands, pool_ids
    gc.collect()

    return {'source': source_label, 'candidates': n_cands,
            's1_with_candidates': n_s1_with_cands,
            'brute_force_pairs': brute_force, 'reduction_ratio': reduction}


def run(args):
    started = time.monotonic()

    data_dir = Path(args.data_dir)
    split = args.split
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    db_path = report_dir / 'candidates.sqlite'
    if db_path.exists():
        if args.overwrite:
            db_path.unlink()
            log.info(f'Removed existing {db_path}')
        else:
            raise FileExistsError(f'{db_path} exists; use --overwrite or fresh dir')

    routes = args.routes.split(',')
    sources = args.sources.split(',')

    # File paths
    s1_file = data_dir / split / f'{split}_source1.tsv'
    source_files = {}
    for src in sources:
        src_num = src.replace('s', 'source')  # s2 -> source2
        source_files[src] = data_dir / split / f'{split}_{src_num}.tsv'

    # Verify files exist
    for label, path in [('S1', s1_file)] + list(source_files.items()):
        if not path.exists():
            raise FileNotFoundError(f'{label}: {path}')

    # Load S1 (shared across all sources) - column-by-column to save memory
    log.info(f'Loading S1 from {s1_file}...')
    s1_ids = load_column(s1_file, COL_ID)
    s1_names = load_column(s1_file, COL_NAME_NORM)
    s1_addresses = load_column(s1_file, COL_ADDR_NORM)
    log.info(f'Loaded {len(s1_ids):,} S1 entities')

    # Process each source
    source_reports = {}
    for src_label, pool_path in source_files.items():
        result = process_source(
            s1_ids, s1_names, s1_addresses,
            pool_path, src_label, db_path,
            routes, args.top_k_name, args.top_k_address, args.top_k_combined,
            args.chunk_size, args.max_features, args.threshold
        )
        source_reports[src_label] = result

    # Create indexes for fast lookup
    log.info('Creating database indexes...')
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute('CREATE INDEX IF NOT EXISTS idx_s1 ON candidates(s1_id)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_cand ON candidates(candidate_id)')
        db.commit()

    # Evaluate recall if we have ground truth (training split only)
    recall_result = None
    if split == 'train' and args.splits_db:
        splits_db = Path(args.splits_db)
        if splits_db.exists():
            recall_result = evaluate_recall(db_path, splits_db, args.eval_role)

    elapsed = time.monotonic() - started

    # Build report
    report = {
        'stage': 3,
        'status': 'PASS',
        'split': split,
        'routes': routes,
        'sources': sources,
        'top_k': {'name': args.top_k_name, 'address': args.top_k_address,
                  'combined': args.top_k_combined},
        'chunk_size': args.chunk_size,
        'max_features': args.max_features,
        'threshold': args.threshold,
        'source_reports': source_reports,
        'recall': recall_result,
        'elapsed_seconds': elapsed
    }

    report_path = report_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    # Render markdown report
    render_report(report, report_dir / 'report.md')

    log.info(f'\nSTAGE 3 PASS: {report_dir / "report.md"}')
    log.info(f'Elapsed: {elapsed:.0f}s')


def render_report(report, path):
    """Generate markdown summary."""
    lines = ['# Stage 3: Candidate Generation / Blocking', '',
             f'**Status**: {report["status"]}', '',
             '## Configuration', '',
             f'- Routes: {report["routes"]}',
             f'- Sources: {report["sources"]}',
             f'- Top-K: name={report["top_k"]["name"]}, '
             f'address={report["top_k"]["address"]}, '
             f'combined={report["top_k"]["combined"]}',
             f'- Max features: {report["max_features"]:,}',
             f'- Threshold: {report["threshold"]}',
             f'- Elapsed: {report["elapsed_seconds"]:.0f}s', '',
             '## Source Results', '',
             '| Source | Candidates | S1 with candidates | Brute force | Reduction |',
             '|---|---:|---:|---:|---:|']

    for src, info in report['source_reports'].items():
        lines.append(f'| {src} | {info["candidates"]:,} | '
                     f'{info["s1_with_candidates"]:,} | '
                     f'{info["brute_force_pairs"]:,} | '
                     f'{info["reduction_ratio"]:.6f} |')

    if report.get('recall'):
        r = report['recall']
        lines += ['', '## Recall Evaluation', '',
                  f'- Split role: {r["split_role"]}',
                  f'- Total entities: {r["total_entities"]:,}',
                  f'- Entities with matches: {r["entities_with_matches"]:,}',
                  f'- **Overall recall**: {r["overall_recall"]:.4f}',
                  f'- **Macro recall**: {r["macro_recall"]:.4f}',
                  f'- Perfect recall: {r["perfect_recall_entities"]:,} '
                  f'({r["perfect_recall_fraction"]*100:.1f}%)',
                  f'- Zero recall: {r["zero_recall_entities"]:,} '
                  f'({r["zero_recall_fraction"]*100:.1f}%)',
                  f'- Total candidates: {r["total_candidates"]:,}',
                  f'- Avg candidates/entity: {r["avg_candidates_per_entity"]:.1f}', '',
                  '### Recall Distribution', '',
                  f'- P10: {r["recall_distribution"]["p10"]:.4f}',
                  f'- P25: {r["recall_distribution"]["p25"]:.4f}',
                  f'- P50: {r["recall_distribution"]["p50"]:.4f}',
                  f'- P75: {r["recall_distribution"]["p75"]:.4f}',
                  f'- P90: {r["recall_distribution"]["p90"]:.4f}',
                  f'- P99: {r["recall_distribution"]["p99"]:.4f}', '',
                  '### Per-Country Results', '',
                  '| Country | Entities | With matches | Recall | Perfect | Zero |',
                  '|---|---:|---:|---:|---:|---:|']
        for country, stats in sorted(r['per_country'].items()):
            cr = stats['recalled'] / stats['true_matches'] if stats['true_matches'] else 1.0
            lines.append(f'| {country} | {stats["entities"]:,} | '
                         f'{stats["with_matches"]:,} | {cr:.4f} | '
                         f'{stats["perfect"]:,} | {stats.get("zero", 0):,} |')

    lines += ['', '## Next Steps', '',
              '- If recall is below 95%, increase top-K or add more routes',
              '- Inspect zero-recall entities for patterns (common names, missing addresses)',
              '- Stage 4 will compute pairwise features on these candidates', '']

    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=str,
                        default='artifacts/stage2')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--sources', type=str, default='s2,s3',
                        help='Comma-separated sources')
    parser.add_argument('--routes', type=str, default='name,address,combined,rare',
                        help='Comma-separated routes')
    parser.add_argument('--top-k-name', type=int, default=50)
    parser.add_argument('--top-k-address', type=int, default=30)
    parser.add_argument('--top-k-combined', type=int, default=30)
    parser.add_argument('--chunk-size', type=int, default=50000)
    parser.add_argument('--max-features', type=int, default=131072,
                        help='HashingVectorizer n_features (power of 2 recommended)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Minimum cosine similarity threshold')
    parser.add_argument('--report-dir', type=str, default='reports/stage3')
    parser.add_argument('--splits-db', type=str,
                        default='reports/stage1/splits.sqlite')
    parser.add_argument('--eval-role', type=str, default='validation',
                        choices=['train', 'validation', 'all'])
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
