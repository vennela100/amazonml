import ctypes
import os
import sys

import pandas as pd


def keep_awake():
    """Ask Windows not to sleep while this process runs (auto-reverts on exit)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def load_recs(work, split, cols):
    """Concatenate per-source prep outputs (S1, S2, S3 order = global row index).
    The pickles are produced locally by prep.py (trusted input)."""
    parts = []
    for k in (1, 2, 3):
        df = pd.read_pickle(os.path.join(work, f"{split}_s{k}.pkl"))
        parts.append(df[cols])
        del df
    out = pd.concat(parts, ignore_index=True)
    if "country" in out:
        out["country"] = out.country.astype("category")
    return out
