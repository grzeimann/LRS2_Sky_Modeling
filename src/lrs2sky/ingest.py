from __future__ import annotations

import glob
import os
import os.path as op
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from astropy.io import fits


__all__ = [
    "find_sky_files",
    "infer_channel",
]


CHANNELS = ("uv", "orange", "red", "farred")

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


def _iter_dates(center_date: str, ndays: int) -> List[str]:
    d0 = datetime(int(center_date[:4]), int(center_date[4:6]), int(center_date[6:]))
    start = d0 - timedelta(days=int(ndays // 2))
    return [
        (start + timedelta(days=int(i))).strftime("%Y%m%d")
        for i in range(int(ndays))
    ]


def _is_sky_exposure(hdr: fits.Header, min_exptime: float) -> bool:
    exptime = float(hdr.get("EXPTIME", 0.0))
    if exptime < min_exptime:
        return False
    obj = (hdr.get("OBJECT") or "").lower()
    # Heuristic: OBJECT contains "sky" or target slot matches typical SKY slot
    if "sky" in obj:
        return True
    # Optional slot heuristic
    try:
        slot = obj.split("_")[-2]
        if slot == "066":
            return True
    except Exception:
        pass
    # Fallback: not identified as sky
    return False


def find_sky_files(
    folders: Iterable[str],
    pattern: str = "multi*{date}*{channel}.fits",
    date: str = "20220101",
    ndays: int = 365,
    min_exptime: float = 300.0,
    csv_out: Optional[str] = None,
    channel: Optional[str] = None,
) -> pd.DataFrame:
    """Search recursively for LRS2 FITS sky exposures and return a table.

    Parameters
    ----------
    folders : iterable of str
        Base folders to search in (non-recursive glob on pattern per date).
    pattern : str
        Filename pattern containing the substring '{date}'.
    date : str
        Center date (YYYYMMDD) for the date range.
    ndays : int
        Number of days to span around center date.
    min_exptime : float
        Minimum exposure time in seconds.
    csv_out : str, optional
        If provided, write the resulting table to this CSV path.

    Returns
    -------
    pandas.DataFrame
        Table with columns: path, object, exptime, dateobs, ra, dec, arm, channel
    """
    records = []
    dates = _iter_dates(date, ndays)
    # normalize channel string if provided
    ch = str(channel).lower() if channel else None
    if ch and ch not in CHANNELS:
        raise ValueError(f"Unknown channel '{channel}'. Expected one of {CHANNELS}.")
    for folder in folders:
        for d in dates:
            pat = pattern.format(date=d, channel=(ch or "*"))
            for path in sorted(glob.glob(op.join(folder, pat))):
                try:
                    with fits.open(path) as hdul:
                        hdr = hdul[0].header
                except Exception:
                    continue
                if not _is_sky_exposure(hdr, min_exptime=min_exptime):
                    continue
                obj = hdr.get("OBJECT", "")
                exptime = float(hdr.get("EXPTIME", np.nan))
                dateobs = hdr.get("DATE-OBS") or hdr.get("DATE", "")
                ra = hdr.get("RA") or hdr.get("OBJRA")
                dec = hdr.get("DEC") or hdr.get("OBJDEC")
                arm = hdr.get("INSTRUME") or hdr.get("DETECTOR") or ""
                ch_infer = infer_channel(hdr, path)
                # Skip mismatched channel if user requested specific channel
                if ch and ch_infer and ch_infer != ch:
                    continue
                records.append(
                    dict(
                        path=op.abspath(path),
                        object=obj,
                        exptime=exptime,
                        dateobs=dateobs,
                        ra=ra,
                        dec=dec,
                        arm=arm,
                        channel=ch_infer,
                    )
                )
    df = pd.DataFrame.from_records(records)
    if csv_out:
        os.makedirs(op.dirname(op.abspath(csv_out)) or ".", exist_ok=True)
        df.to_csv(csv_out, index=False)
    return df
