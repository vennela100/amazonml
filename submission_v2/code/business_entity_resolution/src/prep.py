"""Fast normalization + tokenization for all records of a split.

Writes <out>/<split>_recs.pkl : DataFrame[id, src, country, name, addr, toks]
  name/addr : normalized strings (used by pair features)
  toks      : space-joined blocking tokens (n:/a:/c: prefixed)
"""
import argparse
import os
import re
import sys
import unicodedata
from multiprocessing import Pool

import pandas as pd

NAME_STOP = set("""pvt private ltd limited llc llp inc incorporated corp corporation co company
the of and sarl sas eurl sa sasu sci plc lp pc pllc sl gmbh ltda formerly dba""".split())

ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "ln": "lane", "ct": "court",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail",
    "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "apt": "unit", "ste": "unit", "suite": "unit", "fl": "floor",
    "r": "rue", "imp": "impasse", "che": "chemin", "rte": "route", "all": "allee",
    "nagr": "nagar", "clny": "colony", "saint": "street",
}
ADDR_STOP = set("""no number house flat plot h null near opp opposite behind unit floor po box
pin door dist district tq""".split())

_nonword = re.compile(r"[^\wऀ-෿]+", re.UNICODE)  # keep Indic vowel signs
_num = re.compile(r"^\d+$")
_numalpha = re.compile(r"^(\d+)([a-z]+)$")


def strip_accents(s):
    if any(ord(c) >= 0x0900 for c in s):  # Indic scripts: keep vowel signs
        return unicodedata.normalize("NFC", s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_name(s):
    s = strip_accents(s.lower()).replace("&", " and ")
    return " ".join(_nonword.sub(" ", s).split())


def name_tokens(raw):
    low = strip_accents(raw.lower())
    toks = _nonword.sub(" ", low.replace("&", " and ")).split()
    core = [t for t in toks if t not in NAME_STOP]
    out = ["n:" + t for t in core]
    # domains / handles: 'bestinfotech.com', '@jexfirst' -> concatenated token
    dom = re.sub(r"\.(com|in|net|org|co|fr|io|biz|us)\b.*$", "", low.strip().lstrip("@"))
    dom = _nonword.sub("", dom)
    if dom:
        out.append("c:" + dom)
    if len(core) >= 2:
        out.append("c:" + "".join(core))
        out.append("c:" + core[0] + core[1])
        out.append("c:" + "".join(t for t in toks))
    return out


def norm_addr_tokens(raw):
    s = strip_accents(raw.lower())
    comps = s.split(",")
    toks_all, bigr = [], []
    for c in comps:
        tt = []
        for t in _nonword.sub(" ", c).split():
            m = _numalpha.match(t)
            if m:
                tt.append(m.group(1).lstrip("0") or "0")
                tt.append(m.group(2))
                continue
            if _num.match(t):
                t = t.lstrip("0") or "0"
            t = ADDR_MAP.get(t, t)
            tt.append(t)
        tt = [t for t in tt if t not in ADDR_STOP]
        toks_all.extend(tt)
        for a, b in zip(tt, tt[1:]):
            bigr.append(a + "_" + b)
    return toks_all, bigr


def process_row(args):
    rid, name, addr = args
    nt = name_tokens(name)
    at, ab = norm_addr_tokens(addr)
    toks = nt + ["a:" + t for t in set(at)] + ["b:" + t for t in set(ab)]
    return norm_name(name), " ".join(at), " ".join(dict.fromkeys(toks))


def load(path, src):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    df.columns = ["id", "name", "addr", "country"]
    df["src"] = src
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for k in (1, 2, 3):
        dst = os.path.join(a.out, f"{a.split}_s{k}.pkl")
        if os.path.exists(dst):
            continue
        p = os.path.join(a.data, a.split, f"{a.split}_source{k}.tsv")
        df = load(p, k)
        print("loaded", p, len(df), flush=True)
        nn, na, tk = [], [], []
        with Pool(a.workers) as pool:
            for r in pool.imap(process_row, zip(df.id, df.name, df.addr), chunksize=5000):
                nn.append(r[0]); na.append(r[1]); tk.append(r[2])
        df = df.drop(columns=["name", "addr"])
        df["nname"], df["naddr"], df["toks"] = nn, na, tk
        del nn, na, tk
        df["src"] = df.src.astype("int8")
        df.to_pickle(dst)
        print("wrote", dst, flush=True)
        del df


if __name__ == "__main__":
    main()
