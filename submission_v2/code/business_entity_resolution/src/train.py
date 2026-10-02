"""Train the pair matcher on training candidates and pick the assignment threshold by
exact macro F0.5 on a held-out set of Source-1 entities.

Decision rule: each pool record is assigned to its highest-probability S1 candidate
if that probability >= threshold (a pool record matches at most one S1 in the data).
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import keep_awake, load_recs
from features import RecordCache, pair_features


def load_cands(path):
    z = np.load(path)
    c = {k: z[k] for k in ("pool", "s1", "score", "rank")}
    c["pool"] = c["pool"].astype(np.int32)
    c["s1"] = c["s1"].astype(np.int32)
    o = np.lexsort((c["rank"], c["pool"]))
    return {k: v[o] for k, v in c.items()}


def assign(pool, s1, prob, thr):
    """argmax per pool row, keep if prob>=thr -> (pool_rows, s1_rows)."""
    o = np.lexsort((-prob, pool))
    p, s, pr = pool[o], s1[o], prob[o]
    first = np.r_[True, p[1:] != p[:-1]]
    keep = first & (pr >= thr)
    return p[keep], s[keep], pr[keep]


def macro_f05(eval_s1, truth_s1_of_pool, ap, as1):
    """eval_s1: S1 rows scored.  truth_s1_of_pool: array over rec rows (-1 = none)."""
    E = pd.Index(eval_s1)
    ntrue = pd.Series(truth_s1_of_pool[truth_s1_of_pool >= 0]).value_counts()
    ntrue = ntrue.reindex(E, fill_value=0).values
    inE = E.get_indexer(as1) >= 0
    ap, as1 = ap[inE], as1[inE]
    tp_mask = truth_s1_of_pool[ap] == as1
    npred = pd.Series(as1).value_counts().reindex(E, fill_value=0).values
    ntp = pd.Series(as1[tp_mask]).value_counts().reindex(E, fill_value=0).values
    fp = npred - ntp
    fn = ntrue - ntp
    denom = 5 * ntp + 4 * fn + fp
    f = np.where(denom > 0, 5 * ntp / np.maximum(denom, 1), 1.0)
    return f.mean(), dict(singleton_frac=float((ntrue == 0).mean()),
                          false_merge_singletons=float(((ntrue == 0) & (npred > 0)).sum() / max((ntrue == 0).sum(), 1)),
                          precision=float(ntp.sum() / max(npred.sum(), 1)),
                          recall=float(ntp.sum() / max(ntrue.sum(), 1)))


def predict_chunked(model, recs, cands, sel, cache, chunk=3_000_000):
    """sel sorted & grouped by pool row; split on pool boundaries."""
    out = np.zeros(len(sel), np.float32)
    p = cands["pool"][sel]
    i = 0
    while i < len(sel):
        j = min(i + chunk, len(sel))
        while j < len(sel) and p[j] == p[j - 1]:
            j += 1
        out[i:j] = model.predict(pair_features(recs, cands, sel[i:j], cache), num_threads=12)
        print(f"  predicted {j:,}/{len(sel):,}", flush=True)
        i = j
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--work", required=True)
    ap_.add_argument("--gt", required=True)
    ap_.add_argument("--eval-frac", type=float, default=0.03)
    ap_.add_argument("--train-pool", type=int, default=700_000)
    ap_.add_argument("--trees", type=int, default=600)
    ap_.add_argument("--out", default="model")
    a = ap_.parse_args()
    keep_awake()
    t0 = time.time()
    recs = load_recs(a.work, "train", ["id", "src", "nname", "naddr"])
    cands = load_cands(os.path.join(a.work, "train_cands.npz"))
    n = len(recs)
    row = pd.Series(np.arange(n), index=recs.id.values)
    gt = pd.read_csv(a.gt, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    truth = np.full(n, -1, np.int64)
    truth[row[ex.m.values].values] = row[ex.source1_entity_id.values].values
    y_all = (truth[cands["pool"]] == cands["s1"])
    cache = RecordCache(recs)
    print(f"record cache built {time.time()-t0:.0f}s", flush=True)
    # --- blocking diagnostics
    matched_pool = np.flatnonzero((truth >= 0) & (recs.src.values != 1))
    hit = np.zeros(n, bool)
    hit[cands["pool"][y_all]] = True
    r0 = np.zeros(n, bool)
    r0[cands["pool"][y_all & (cands["rank"] == 0)]] = True
    print(f"blocking recall (matched pool rows with true S1 in cands): {hit[matched_pool].mean():.4f}; "
          f"rank0 acc {r0[matched_pool].mean():.4f}; pairs {len(y_all):,}", flush=True)
    for k in (1, 2, 3, 5, 10):
        hk = np.zeros(n, bool)
        hk[cands["pool"][y_all & (cands["rank"] < k)]] = True
        print(f"  recall@{k}: {hk[matched_pool].mean():.4f}", flush=True)
    # --- eval / train split
    rng = np.random.default_rng(42)
    s1_rows = np.flatnonzero(recs.src.values == 1)
    E = s1_rows[rng.random(len(s1_rows)) < a.eval_frac]
    inE = np.zeros(n, bool)
    inE[E] = True
    eval_pool = np.unique(cands["pool"][inE[cands["s1"]] & (cands["rank"] < 3)])
    is_eval_pool = np.zeros(n, bool)
    is_eval_pool[eval_pool] = True
    cand_pools = np.unique(cands["pool"])
    tr_pool = cand_pools[~is_eval_pool[cand_pools]]
    tr_pool = rng.choice(tr_pool, size=min(a.train_pool, len(tr_pool)), replace=False)
    is_tr = np.zeros(n, bool)
    is_tr[tr_pool] = True
    sel_tr = np.flatnonzero(is_tr[cands["pool"]])
    sel_ev = np.flatnonzero(is_eval_pool[cands["pool"]])
    print(f"train pairs {len(sel_tr):,}  eval pairs {len(sel_ev):,}  eval S1 {len(E):,}", flush=True)
    Xtr = pair_features(recs, cands, sel_tr, cache)
    ytr = y_all[sel_tr]
    print(f"features train done {time.time()-t0:.0f}s pos rate {ytr.mean():.3f}", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=12, verbose=-1)
    model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=a.trees)
    del Xtr
    pev = predict_chunked(model, recs, cands, sel_ev, cache)
    print(f"eval predicted {time.time()-t0:.0f}s", flush=True)
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.96, 0.02):
        apool, as1, _ = assign(cands["pool"][sel_ev], cands["s1"][sel_ev], pev, thr)
        f, info = macro_f05(E, truth, apool, as1)
        print(f"thr {thr:.2f}: macroF0.5 {f:.5f} {info}", flush=True)
        if f > best[0]:
            best = (f, float(thr))
    print("BEST", best, flush=True)
    os.makedirs(os.path.join(a.work, a.out), exist_ok=True)
    model.save_model(os.path.join(a.work, a.out, "lgb.txt"))
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
    print(imp.head(25).to_string())
    json.dump({"threshold": best[1], "eval_f05": best[0]}, open(os.path.join(a.work, a.out, "decision.json"), "w"))
    # keep eval preds for error analysis
    np.savez(os.path.join(a.work, a.out, "eval_preds.npz"), sel=sel_ev, prob=pev, E=E)


if __name__ == "__main__":
    main()
