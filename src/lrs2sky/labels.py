from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_moon, get_sun
from astropy.time import Time
import astropy.units as u


__all__ = [
    "compute_labels_from_header",
]

# HET location (approximate) – update if needed
HET_LOCATION = EarthLocation(lat=30.6814 * u.deg, lon=-104.0147 * u.deg, height=2025 * u.m)


def _parse_time(header: Dict[str, Any]) -> Optional[Time]:
    for key in ("DATE-OBS", "DATE", "MJD-OBS"):
        if key in header and header[key]:
            val = header[key]
            if key == "MJD-OBS":
                try:
                    return Time(float(val), format="mjd", scale="utc")
                except Exception:
                    continue
            try:
                return Time(val, format="isot", scale="utc")
            except Exception:
                try:
                    return Time(val)
                except Exception:
                    continue
    return None


def _parse_radec(header: Dict[str, Any]) -> Optional[SkyCoord]:
    ra = header.get("RA") or header.get("OBJRA")
    dec = header.get("DEC") or header.get("OBJDEC")
    if ra is None or dec is None:
        return None
    try:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    except Exception:
        try:
            return SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        except Exception:
            return None


def _airmass(altaz) -> Optional[float]:
    try:
        secz = 1.0 / np.cos((90 * u.deg - altaz.alt).to(u.rad))
        return float(secz)
    except Exception:
        return None


def compute_labels_from_header(header: Dict[str, Any]) -> Dict[str, float]:
    """Compute metadata labels for a single exposure from its FITS header.

    Returns a dict with keys like: airmass, sun_alt, moon_alt, moon_illum,
    moon_sep, glat, elat, doy_sin, doy_cos, exptime, arm, transparency.
    Missing quantities are returned as np.nan.
    """
    out: Dict[str, float] = {}

    t = _parse_time(header)
    coord = _parse_radec(header)

    # Exposure time, arm, and transparency (THROUGHP)
    out["exptime"] = float(header.get("EXPTIME", np.nan))
    arm = header.get("INSTRUME") or header.get("DETECTOR") or header.get("ARM")
    out["arm_id"] = {"blue": 0.0, "red": 1.0}.get(str(arm).lower(), np.nan)
    try:
        out["transparency"] = float(header.get("THROUGHP", np.nan))
    except Exception:
        out["transparency"] = np.nan

    if t is None or coord is None:
        # Fill with NaNs if we cannot compute
        for k in [
            "airmass",
            "sun_alt",
            "moon_alt",
            "moon_illum",
            "moon_sep",
            "glat",
            "elat",
            "doy_sin",
            "doy_cos",
        ]:
            out[k] = np.nan
        return out

    altaz = coord.transform_to(AltAz(obstime=t, location=HET_LOCATION))
    out["airmass"] = _airmass(altaz)

    sun = get_sun(t).transform_to(AltAz(obstime=t, location=HET_LOCATION))
    moon_icrs = get_moon(t, location=HET_LOCATION)
    moon = moon_icrs.transform_to(AltAz(obstime=t, location=HET_LOCATION))

    out["sun_alt"] = float(sun.alt.to(u.deg))
    out["moon_alt"] = float(moon.alt.to(u.deg))

    # Simple illumination proxy from phase angle
    try:
        elong = coord.separation(moon_icrs)
        out["moon_sep"] = float(elong.to(u.deg))
    except Exception:
        out["moon_sep"] = np.nan

    try:
        phase_angle = get_sun(t).separation(moon_icrs).to(u.rad).value
        illum = (1 + np.cos(phase_angle)) / 2
        out["moon_illum"] = float(illum)
    except Exception:
        out["moon_illum"] = np.nan

    try:
        out["glat"] = float(coord.galactic.b.to(u.deg))
    except Exception:
        out["glat"] = np.nan

    try:
        ecl = coord.barycentrictrueecliptic
        out["elat"] = float(ecl.lat.to(u.deg))
    except Exception:
        out["elat"] = np.nan

    # Day-of-year encoding
    try:
        dt = t.to_datetime()
        doy = dt.timetuple().tm_yday
        ang = 2 * np.pi * (doy / 365.25)
        out["doy_sin"] = float(np.sin(ang))
        out["doy_cos"] = float(np.cos(ang))
    except Exception:
        out["doy_sin"] = np.nan
        out["doy_cos"] = np.nan

    return out
