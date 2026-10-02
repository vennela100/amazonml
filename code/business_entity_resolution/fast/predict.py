"""Score test candidates, assign each pool record to its best S1 (if prob >= threshold),
and write matching_results.tsv + candidate_pairs.tsv for every test S1 entity.

Processed one country at a time (blocking never crosses countries) to bound memory."""
import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import keep_awake, load_recs
from features import RecordCache, pair_features
from train import assign, load_cands


def write_lists(path, header, s1_ids, s1_rows, pairs_s1, pairs_ids):
    """One row per S1 entity; comma-joined ids (deduplicated)."""
    df = pd.DataFrame({"s": pairs_s1, "m": pairs_ids}).drop_duplicates()
    lists = df.groupby("s").m.agg(",".join)
    col = pd.Series(s1_rows).map(lists).fillna("").values
    out = pd.DataFrame({"source1_entity_id": s1_ids, header: col})
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


def score_country(recs_all, cands_all, rows, model, pool_chunk, t0):
    """rows: global record rows of one country.  Returns prob for the country's pairs."""
    loc = np.full(len(recs_all), -1, np.int32)
    loc[rows] = np.arange(len(rows), dtype=np.int32)
    m = loc[cands_all["pool"]] >= 0
    idx = np.flatnonzero(m)
    recs = recs_all.iloc[rows].reset_index(drop=True)
    cands = {"pool": loc[cands_all["pool"][idx]], "s1": loc[cands_all["s1"][idx]],
             "score": cands_all["score"][idx], "rank": cands_all["rank"][idx]}
    cache = RecordCache(recs)
    print(f"  cache built ({len(rows):,} recs, {len(idx):,} pairs) {time.time()-t0:.0f}s", flush=True)
    prob = np.zeros(len(idx), np.float32)
    feats = model.feature_name()
    bounds = np.searchsorted(cands["pool"], np.unique(cands["pool"])[::pool_chunk])
    bounds = np.r_[bounds, len(cands["pool"])]
    for i in range(len(bounds) - 1):
        sel = np.arange(bounds[i], bounds[i + 1])
        X = pair_features(recs, cands, sel, cache)[feats]
        prob[sel] = model.predict(X, num_threads=12)
        del X
        print(f"  chunk {i+1}/{len(bounds)-1} {time.time()-t0:.0f}s", flush=True)
    return idx, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", required=True, help="dir with lgb.txt + decision.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--pool-chunk", type=int, default=300_000)
    a = ap.parse_args()
    keep_awake()
    t0 = time.time()
    recs = load_recs(a.work, "test", ["id", "src", "country", "nname", "naddr"])
    cands = load_cands(os.path.join(a.work, "test_cands.npz"))
    model = lgb.Booster(model_file=os.path.join(a.model, "lgb.txt"))
    thr = a.threshold if a.threshold is not None else json.load(open(os.path.join(a.model, "decision.json")))["threshold"]
    prob_path = os.path.join(a.work, f"test_prob_{os.path.basename(os.path.normpath(a.model))}.npy")
    if os.path.exists(prob_path):
        prob = np.load(prob_path)
    else:
        prob = np.zeros(len(cands["pool"]), np.float32)
        for c in recs.country.cat.categories:
            rows = np.flatnonzero((recs.country == c).values)
            print(f"{c}: {len(rows):,} records", flush=True)
            idx, p = score_country(recs, cands, rows, model, a.pool_chunk, t0)
            prob[idx] = p
            del idx, p
            gc.collect()
        np.save(prob_path, prob)
    ap_, as1, _ = assign(cands["pool"], cands["s1"], prob, thr)
    ids = recs.id.values
    s1_rows = np.flatnonzero(recs.src.values == 1)
    os.makedirs(a.out, exist_ok=True)
    write_lists(os.path.join(a.out, "matching_results.tsv"), "matched_entity_ids",
                ids[s1_rows], s1_rows, as1, ids[ap_])
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids",
                ids[s1_rows], s1_rows, cands["s1"], ids[cands["pool"]])
    print(f"assigned {len(ap_):,} pool records at thr {thr}; "
          f"S1 with matches {pd.Series(as1).nunique():,}/{len(s1_rows):,}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
