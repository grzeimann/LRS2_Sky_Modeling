from __future__ import annotations

import os.path as op
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits


__all__ = [
    "infer_channel",
    "load_channel_index",
    "RECOMMENDED_MASTER_KEYS",
]


CHANNELS = ("uv", "orange", "red", "farred")

# Recommended header keywords to include in a master index file. These are common
# and broadly useful for filtering and analysis. The index should at minimum
# contain a 'path' column with the absolute path to the FITS exposure.
RECOMMENDED_MASTER_KEYS = [
    "PATH",
    "OBJECT",
    "DATE",
    "TIME",
    "EXPTIME",
    "AIRMASS",
    "RA",
    "DEC",
    "THROUGHP"
]

def infer_channel(hdr: Optional[fits.Header] = None, path: Optional[str] = None) -> Optional[str]:
    """Infer LRS2 channel string ('uv','orange','red','farred') from header or file path.

    Returns None if it cannot be determined.
    """
    # Try header fields first
    cand = None
    if hdr is not None:
        for key in ("CHANNEL", "ARM", "INSTRUME", "DETECTOR"):
            v = hdr.get(key)
            if v:
                cand = str(v).lower()
                break
    # Fallback to path name
    if cand is None and path:
        cand = str(path).lower()
    if cand:
        for ch in CHANNELS:
            if ch in cand:
                return ch
        # Common blue arm names sometimes encode 'uv' or 'orange' differently
        if "blue" in cand and "uv" in CHANNELS:
            return "uv"
        if "blue" in cand and "orange" in CHANNELS and "uv" not in cand:
            # ambiguous; leave None
            pass
    return None


def _standardize_paths(paths: Iterable[str], base_dir: str) -> List[str]:
    out = []
    for p in paths:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            out.append(None)
            continue
        s = str(p)
        if not s:
            out.append(None)
            continue
        if not op.isabs(s):
            s = op.abspath(op.join(base_dir, s))
        out.append(s)
    return out


def load_channel_index(channel: str, archive_dir: str = "archive", path: Optional[str] = None) -> pd.DataFrame:
    """Load a prebuilt exposure list for a given LRS2 channel from the archive.

    The archive text files are expected to be whitespace-delimited with no
    header line and per-row fields:
      path  object  date  time  exptime  airmass  ra  dec

    - The 'channel' is inferred from the filename (must match requested channel)
      or from the provided 'channel' argument.
    - The output DataFrame schema is canonical across the package with columns:
      path, object, exptime, dateobs, ra, dec, arm, channel

    Parameters
    ----------
    channel : str
        One of {uv, orange, red, farred}.
    archive_dir : str, default 'archive'
        Directory containing '<channel>_file_list.txt'.
    path : str, optional
        Override path to a specific list file. If provided, 'channel' is used
        only for validation/filtering.
    """
    ch = str(channel).lower()
    if ch not in CHANNELS:
        raise ValueError(f"Unknown channel '{channel}'. Expected one of {CHANNELS}.")

    list_path = path or op.join(archive_dir, f"{ch}_file_list.txt")
    if not op.exists(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")

    # Read whitespace-delimited with no header (use regex separator for future pandas versions)
    df_raw = pd.read_csv(list_path, sep=r"\s+", engine="python", header=None, comment="#", dtype=str)
    # Expect 9 columns; if more due to spaces in OBJECT, collapse adjacent columns
    # Heuristic: first column is path, last two are RA, DEC, and the two before are exptime, airmass
    if df_raw.shape[1] < 9:
        raise ValueError(f"Unexpected column count ({df_raw.shape[1]}) in {list_path}; expected >= 8.")

    # Build columns robustly
    # Combine middle columns into single object name if more than 8 total
    ncol = df_raw.shape[1]
    path_col = df_raw.iloc[:, 0]
    obj_parts = df_raw.iloc[:, ncol - 8]
    date_col = df_raw.iloc[:, ncol - 7]
    time_col = df_raw.iloc[:, ncol - 6]
    exptime_col = df_raw.iloc[:, ncol - 5]
    airmass_col = df_raw.iloc[:, ncol - 4]
    ra_col = df_raw.iloc[:, ncol - 3]
    dec_col = df_raw.iloc[:, ncol - 2]
    trans_col = df_raw.iloc[:, ncol - 1]

    out = pd.DataFrame()
    # Standardize paths relative to the list file location
    base_dir = op.dirname(op.abspath(list_path))
    out["path"] = _standardize_paths(path_col.tolist(), base_dir)
    out["object"] = obj_parts.str.strip()

    # exptime as float where possible
    def to_float(x):
        try:
            return float(str(x))
        except Exception:
            return np.nan

    out["exptime"] = exptime_col.apply(to_float)

    # dateobs combine
    out["dateobs"] = (date_col.astype(str).str.strip() + "T" + time_col.astype(str).str.strip())

    out["ra"] = ra_col.astype(str).str.strip()
    out["dec"] = dec_col.astype(str).str.strip()

    # determine channel per row using path; enforce match with requested channel
    rows_ch = [infer_channel(None, p) for p in out["path"].tolist()]
    out["channel"] = [rc if rc in CHANNELS else None for rc in rows_ch]
    out["transparency"] = trans_col.apply(to_float)
    # filter to requested channel
    mask = [c == ch for c in out["channel"].tolist()]
    out = out.loc[mask].reset_index(drop=True)

    return out
