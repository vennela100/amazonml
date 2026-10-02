"""Error analysis on held-out eval predictions from train.py."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import load_recs
from train import assign, load_cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    recs = load_recs(a.work, "train", ["id", "src", "nname", "naddr"])
    cands = load_cands(os.path.join(a.work, "train_cands.npz"))
    z = np.load(os.path.join(a.work, a.model, "eval_preds.npz"))
    sel, prob, E = z["sel"], z["prob"], z["E"]
    thr = json.load(open(os.path.join(a.work, a.model, "decision.json")))["threshold"]
    n = len(recs)
    row = pd.Series(np.arange(n), index=recs.id.values)
    gt = pd.read_csv(a.gt, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    truth = np.full(n, -1, np.int64)
    truth[row[ex.m.values].values] = row[ex.source1_entity_id.values].values
    inE = np.zeros(n, bool)
    inE[E] = True
    P, S = cands["pool"][sel], cands["s1"][sel]
    ap_, as1, apr = assign(P, S, prob, thr)
    assigned = np.full(n, -1, np.int64)
    assigned[ap_] = as1
    # all pool rows whose truth in E
    tp_rows = np.flatnonzero((truth >= 0) & inE[np.maximum(truth, 0)])
    incand = np.zeros(n, bool)
    incand[P[truth[P] == S]] = True
    best = pd.DataFrame({"p": P, "s": S, "pr": prob}).sort_values("pr", ascending=False).drop_duplicates("p").set_index("p")
    fn_missing = tp_rows[~incand[tp_rows]]
    fn_rest = tp_rows[incand[tp_rows] & (assigned[tp_rows] != truth[tp_rows])]
    fn_wrong = fn_rest[assigned[fn_rest] >= 0]
    fn_below = fn_rest[assigned[fn_rest] < 0]
    fp_rows = ap_[inE[as1] & (truth[ap_] != as1)]
    fp_distr = fp_rows[truth[fp_rows] < 0]
    print(f"true pool rows for E: {len(tp_rows):,}")
    print(f"FN not in cands: {len(fn_missing):,}  FN wrong S1: {len(fn_wrong):,}  FN below thr: {len(fn_below):,}")
    print(f"FP into E: {len(fp_rows):,} (of which distractor pool rows: {len(fp_distr):,})")
    rng = np.random.default_rng(0)

    def show(title, rows, other=None):
        print(f"\n===== {title}")
        for r in rng.choice(rows, size=min(a.n, len(rows)), replace=False):
            t = truth[r]
            print(f"POOL  {recs.nname.values[r]} | {recs.naddr.values[r]}")
            if t >= 0:
                print(f"  TRUE {recs.nname.values[t]} | {recs.naddr.values[t]}")
            if r in best.index:
                b = best.loc[r]
                print(f"  BEST({b.pr:.2f}) {recs.nname.values[int(b.s)]} | {recs.naddr.values[int(b.s)]}")

    show("FN not in candidates", fn_missing)
    show("FN assigned to wrong S1", fn_wrong)
    show("FN below threshold", fn_below)
    show("FP distractor merged", fp_distr)


if __name__ == "__main__":
    main()
