"""Candidate generation: for every pool (S2/S3) record, top-K Source-1 records by
TF-IDF cosine over name+address tokens, computed per country with a multithreaded
sparse top-n matmul.  Writes <out>/<split>_cands.npz (pool_row, s1_row, score, rank)
where rows index into <split>_recs.pkl.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import keep_awake, load_recs


def add_name_bigrams(toks):
    """Name word-pairs ('m:pediatric_care'): rare and discriminative even when the
    single name words are too common to survive the df cap."""
    n = [t[2:] for t in toks.split() if t.startswith("n:")]
    if len(n) < 2:
        return toks
    return toks + " " + " ".join("m:" + a + "_" + b for a, b in zip(n, n[1:]))


def run_country(recs, cidx, k, max_df, threads, chunk):
    sub = recs.iloc[cidx]
    is1 = (sub.src.values == 1)
    s1_rows = cidx[is1]
    pl_rows = cidx[~is1]
    vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, max_df=max_df,
                          dtype=np.float32, lowercase=False)
    X = vec.fit_transform(sub.toks.values)
    print(f"  tfidf {X.shape} nnz {X.nnz:,}", flush=True)
    S = X[is1]
    P = X[~is1]
    del X
    ST = S.T.tocsr()
    out_p, out_s, out_v, out_r = [], [], [], []
    for st in range(0, P.shape[0], chunk):
        M = sp_matmul_topn(P[st:st + chunk], ST, top_n=k, threshold=0.01,
                           sort=True, n_threads=threads)
        M = M.tocsr()
        cnt = np.diff(M.indptr)
        rloc = np.repeat(np.arange(M.shape[0]), cnt)
        rank = np.arange(M.nnz) - np.repeat(M.indptr[:-1], cnt)
        out_p.append(pl_rows[st + rloc])
        out_s.append(s1_rows[M.indices])
        out_v.append(M.data.astype(np.float32))
        out_r.append(rank.astype(np.int8))
        if (st // chunk) % 10 == 0:
            print(f"  {st + chunk:,}/{P.shape[0]:,} pool rows", flush=True)
    return (np.concatenate(out_p), np.concatenate(out_s),
            np.concatenate(out_v), np.concatenate(out_r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-df", type=int, default=3000, help="absolute df cap")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--sample-pool", type=float, default=1.0,
                    help="fraction of pool rows to query (train speed-up)")
    ap.add_argument("--name-bigrams", action="store_true")
    ap.add_argument("--out-name", default=None, help="candidates file name (default <split>_cands.npz)")
    a = ap.parse_args()
    keep_awake()
    recs = load_recs(a.work, a.split, ["src", "country", "toks"])
    if a.name_bigrams:
        recs["toks"] = [add_name_bigrams(t) for t in recs.toks.values]
        print("name bigrams added", flush=True)
    rng = np.random.default_rng(0)
    parts = []
    for c in recs.country.cat.categories:
        cidx = np.flatnonzero((recs.country == c).values)
        if a.sample_pool < 1.0:
            keep = (recs.src.values[cidx] == 1) | (rng.random(len(cidx)) < a.sample_pool)
            cidx = cidx[keep]
        t = time.time()
        r = run_country(recs, cidx, a.k, a.max_df, a.threads, a.chunk)
        print(f"{c}: {len(cidx):,} recs -> {len(r[0]):,} cands in {time.time()-t:.0f}s", flush=True)
        parts.append(r)
    p, s, v, rk = (np.concatenate(x) for x in zip(*parts))
    np.savez(os.path.join(a.work, a.out_name or f"{a.split}_cands.npz"), pool=p, s1=s, score=v, rank=rk)


if __name__ == "__main__":
    main()
