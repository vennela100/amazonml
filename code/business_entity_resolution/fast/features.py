"""Pair features for (pool record, S1 candidate) pairs."""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

_digits = re.compile(r"\d+")


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32).astype(np.float32)


def _numsets(arr):
    return [frozenset(_digits.findall(s)) for s in arr]


def _first_num(arr):
    out = []
    for s in arr:
        m = _digits.search(s)
        out.append(m.group(0) if m else "")
    return out


def _tokset(arr):
    return [frozenset(s.split()) for s in arr]


def _context_cache(nrec, cands):
    """Per-record / per-pair context arrays over the FULL candidate set (computed once)."""
    if "_ctx" in cands:
        return cands["_ctx"]
    pool, s1, score, rank = cands["pool"], cands["s1"], cands["score"], cands["rank"]
    c = {}
    m0, m1 = rank == 0, rank == 1
    c["top1"] = np.zeros(nrec, np.float32)
    c["top1"][pool[m0]] = score[m0]
    c["sec"] = np.zeros(nrec, np.float32)
    c["sec"][pool[m1]] = score[m1]
    c["ncand"] = np.bincount(pool, minlength=nrec).astype(np.float32)
    c["deg0"] = np.bincount(s1[m0], minlength=nrec).astype(np.float32)
    c["degc"] = np.bincount(s1, minlength=nrec).astype(np.float32)
    c["claim"] = np.bincount(s1[m0], weights=score[m0], minlength=nrec).astype(np.float32)
    # rank of each pair among all pairs pointing at the same S1 (by score, desc)
    o = np.lexsort((-score, s1))
    s1o = s1[o]
    start = np.r_[0, np.flatnonzero(s1o[1:] != s1o[:-1]) + 1]
    grp_start = np.repeat(start, np.diff(np.r_[start, len(s1o)]))
    inrank = np.empty(len(s1), np.float32)
    inrank[o] = np.arange(len(s1o)) - grp_start
    del o, s1o, grp_start
    c["inrank"] = inrank
    c["top_in"] = np.zeros(nrec, np.float32)
    top_pairs = np.flatnonzero(inrank == 0)
    c["top_in"][s1[top_pairs]] = score[top_pairs]
    cands["_ctx"] = c
    return c


def context_features(nrec, cands, sel, src):
    """Candidate-list context computed over the FULL candidate set, returned for sel."""
    c = _context_cache(nrec, cands)
    p, s, v = cands["pool"][sel], cands["s1"][sel], cands["score"][sel]
    return pd.DataFrame({
        "score": v, "rank": cands["rank"][sel].astype(np.float32),
        "gap_top1": c["top1"][p] - v, "margin12": c["top1"][p] - c["sec"][p], "ncand": c["ncand"][p],
        "s1_rank0_deg": c["deg0"][s], "s1_cand_deg": c["degc"][s], "s1_claim": c["claim"][s],
        "src": src[p].astype(np.float32),
        "s1_inrank": c["inrank"][sel], "s1_top_in_gap": c["top_in"][s] - v,
    })


class RecordCache:
    """Per-record token matrices so pair overlaps are sparse row products (C speed)."""

    def __init__(self, recs):
        from sklearn.feature_extraction.text import CountVectorizer
        cv = dict(binary=True, lowercase=False, dtype=np.float32)
        self.A = CountVectorizer(analyzer=str.split, **cv).fit_transform(recs.naddr.values).tocsr()
        self.N = CountVectorizer(analyzer=_digits.findall, **cv).fit_transform(recs.naddr.values).tocsr()
        self.T = CountVectorizer(analyzer=str.split, **cv).fit_transform(recs.nname.values).tocsr()
        # IDF (over this split) for weighted overlaps: cos = sum_shared idf^2 / (|p||s|)
        for nm in ("A", "T"):
            M = getattr(self, nm)
            df = np.bincount(M.indices, minlength=M.shape[1]).astype(np.float32)
            idf2 = np.log((M.shape[0] + 1) / (df + 1)) + 1.0
            idf2 = (idf2 * idf2).astype(np.float32)
            setattr(self, "idf2" + nm, idf2)
            setattr(self, "norm" + nm, np.sqrt(M @ idf2).astype(np.float32))
        self.lenA = np.diff(self.A.indptr).astype(np.float32)
        self.lenN = np.diff(self.N.indptr).astype(np.float32)
        self.lenT = np.diff(self.T.indptr).astype(np.float32)
        first = pd.Series(recs.naddr.values).str.extract(r"(\d+)", expand=False).fillna("")
        codes, _ = pd.factorize(first)
        codes[first.values == ""] = -1
        self.first = codes
        self.nlen = recs.nname.str.len().values.astype(np.float32)
        self.nonlatin = recs.nname.str.contains(r"[ऀ-෿]", regex=True).values.astype(np.float32)


def _inter(M, p, s):
    return np.asarray(M[p].multiply(M[s]).sum(axis=1)).ravel().astype(np.float32)


def _wcos(M, idf2, norm, p, s):
    sh = M[p].multiply(M[s]).tocsr()
    num = sh @ idf2
    return (num / np.maximum(norm[p] * norm[s], 1e-6)).astype(np.float32)


def _strings(recs, pool, s1, cache):
    pn = recs.nname.values[pool]
    sn = recs.nname.values[s1]
    pa = recs.naddr.values[pool]
    sa = recs.naddr.values[s1]
    f = {}
    f["n_ratio"] = _cp(pn, sn, fuzz.ratio)
    f["n_tset"] = _cp(pn, sn, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(pn, sn, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(pn, sn, fuzz.partial_ratio)
    f["n_jw"] = _cp(pn, sn, JaroWinkler.normalized_similarity)
    pc = np.array([x.replace(" ", "") for x in pn], dtype=object)
    sc = np.array([x.replace(" ", "") for x in sn], dtype=object)
    f["n_concat"] = _cp(pc, sc, fuzz.ratio)
    f["n_concat_partial"] = _cp(pc, sc, fuzz.partial_ratio)
    f["a_ratio"] = _cp(pa, sa, fuzz.ratio)
    f["a_tset"] = _cp(pa, sa, fuzz.token_set_ratio)
    f["a_tsort"] = _cp(pa, sa, fuzz.token_sort_ratio)
    f["a_partial"] = _cp(pa, sa, fuzz.partial_ratio)
    c = cache
    inter = _inter(c.N, pool, s1)
    lp, ls = c.lenN[pool], c.lenN[s1]
    f["num_inter"] = inter
    f["num_p"] = lp
    f["num_s"] = ls
    f["num_jac"] = inter / np.maximum(lp + ls - inter, 1)
    f["num_conflict"] = ((lp > 0) & (ls > 0) & (inter == 0)).astype(np.float32)
    f["num_p_subset"] = ((lp > 0) & (inter == lp)).astype(np.float32)
    f["first_num_eq"] = ((c.first[pool] >= 0) & (c.first[pool] == c.first[s1])).astype(np.float32)
    ai = _inter(c.A, pool, s1)
    lpa, lsa = c.lenA[pool], c.lenA[s1]
    f["a_tok_inter"] = ai
    f["a_tok_cov_p"] = ai / np.maximum(lpa, 1)
    f["a_tok_cov_s"] = ai / np.maximum(lsa, 1)
    f["a_len_p"] = lpa
    f["a_len_s"] = lsa
    ni = _inter(c.T, pool, s1)
    lpn, lsn = c.lenT[pool], c.lenT[s1]
    f["n_tok_inter"] = ni
    f["n_tok_cov_p"] = ni / np.maximum(lpn, 1)
    f["n_tok_cov_s"] = ni / np.maximum(lsn, 1)
    f["n_len_p"] = c.nlen[pool]
    f["n_len_s"] = c.nlen[s1]
    f["p_nonlatin"] = c.nonlatin[pool]
    f["a_wcos"] = _wcos(c.A, c.idf2A, c.normA, pool, s1)
    f["n_wcos"] = _wcos(c.T, c.idf2T, c.normT, pool, s1)
    return pd.DataFrame(f)


def pair_features(recs, cands, sel, cache, chunk=1_000_000):
    """Features for candidate pairs cands[sel].  cands: dict pool/s1/score/rank arrays
    (full set, sorted by pool then rank).  Country-agnostic on purpose: France is
    unseen in training."""
    X = context_features(len(recs), cands, sel, recs.src.values)
    pool, s1 = cands["pool"][sel], cands["s1"][sel]
    parts = [_strings(recs, pool[i:i + chunk], s1[i:i + chunk], cache) for i in range(0, len(pool), chunk)]
    X = pd.concat([X, pd.concat(parts, ignore_index=True)], axis=1)
    # relative-to-best-candidate versions of key sims
    for c in ("n_tset", "a_tset", "num_jac", "n_concat", "n_ratio"):
        X[c + "_rel"] = X[c].values - X.groupby(pool)[c].transform("max").values
    return X
