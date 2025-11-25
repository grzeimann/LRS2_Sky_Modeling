from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import BarycentricTrueEcliptic
try:
    from astropy.coordinates.baseframe import NonRotationTransformationWarning  # type: ignore
except Exception:  # pragma: no cover - compatibility for older astropy
    class NonRotationTransformationWarning(Warning):
        pass
# Compatibility wrappers for solar/lunar positions across Astropy versions
try:  # Astropy with direct get_sun/get_moon
    from astropy.coordinates import get_moon as _get_moon, get_sun as _get_sun
    def _compat_get_moon(t, location=None):
        return _get_moon(t, location=location)
    def _compat_get_sun(t):
        return _get_sun(t)
except Exception:  # Fallback to generic body getter
    from astropy.coordinates import get_body as _get_body
    def _compat_get_moon(t, location=None):
        return _get_body("moon", t, location=location)
    def _compat_get_sun(t):
        return _get_body("sun", t)
from astropy.time import Time
import astropy.units as u
import warnings


__all__ = [
    "compute_labels_from_row",
]


# HET location (approximate) – update if needed
HET_LOCATION = EarthLocation(lat=30.6814 * u.deg, lon=-104.0147 * u.deg, height=2025 * u.m)


def _get_case_insensitive(row: pd.Series, *candidates: str, default=None):
    if row is None:
        return default
    lower_map = {str(k).lower(): k for k in row.index}
    for name in candidates:
        k = lower_map.get(str(name).lower())
        if k is not None:
            val = row[k]
            # Normalize NaN/empty to default
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            s = str(val)
            if s.strip() == "" or s.strip().lower() == "nan":
                continue
            return val
    return default


def _parse_time_from_row(row: pd.Series) -> Optional[Time]:
    # Prefer dateobs if present
    dateobs = _get_case_insensitive(row, "dateobs", "DATE-OBS", "DATEOBS")
    if dateobs is not None:
        val = str(dateobs).strip()
        # If value looks like "YYYY-MM-DD HH:MM:SS" convert to ISO "T"
        if " " in val and "T" not in val:
            val = val.replace(" ", "T", 1)
        try:
            return Time(val, format="isot", scale="utc")
        except Exception:
            try:
                return Time(val)
            except Exception:
                pass
    # Try DATE + TIME columns
    date = _get_case_insensitive(row, "date", "DATE")
    time = _get_case_insensitive(row, "time", "TIME")
    if date is not None and time is not None:
        val = f"{str(date).strip()}T{str(time).strip()}"
        try:
            return Time(val, format="isot", scale="utc")
        except Exception:
            try:
                return Time(val)
            except Exception:
                pass
    # Try MJD
    mjd = _get_case_insensitive(row, "mjd", "MJD-OBS", "mjd_obs")
    if mjd is not None:
        try:
            return Time(float(mjd), format="mjd", scale="utc")
        except Exception:
            return None
    return None


def _parse_radec_from_row(row: pd.Series) -> Optional[SkyCoord]:
    ra = _get_case_insensitive(row, "ra", "RA", "objra", "OBJRA")
    dec = _get_case_insensitive(row, "dec", "DEC", "objdec", "OBJDEC")
    if ra is None or dec is None:
        return None
    # Try sexagesimal RA in hours + DEC in deg
    try:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    except Exception:
        pass
    # Try both in degrees
    try:
        return SkyCoord(float(str(ra)), float(str(dec)), unit=(u.deg, u.deg))
    except Exception:
        return None


def _airmass(altaz) -> Optional[float]:
    try:
        secz = 1.0 / np.cos((90 * u.deg - altaz.alt).to(u.rad))
        # secz is a dimensionless Quantity in recent Astropy; use .value
        return secz.value
    except Exception:
        return None


def compute_labels_from_row(row: pd.Series) -> Dict[str, float]:
    """Compute metadata labels for a single exposure from a CSV row.

    Expected row fields (case-insensitive):
    - dateobs OR (DATE and TIME)
    - ra, dec (or OBJRA/OBJDEC)
    - exptime (optional)
    - transparency or THROUGHP (optional)
    - environmental scalars (optional): AMBTEMP, HUMIDITY, DEWPOINT, BAROMPRE, WINDDIR, WINDSPD

    Returns a dict with keys: airmass, sun_alt, moon_alt, moon_illum,
    moon_sep, glat, elat, doy_sin, doy_cos, exptime, transparency, millum,
    ambtemp, humidity, dewpoint, barompre, winddir, windspd.
    Missing quantities are returned as np.nan.
    """
    out: Dict[str, float] = {}

    # Helper to pull a scalar from the row and coerce to float
    def add_scalar(out_key: str, *candidates: str) -> None:
        val = _get_case_insensitive(row, *candidates)
        try:
            out[out_key] = float(val) if val is not None else np.nan
        except Exception:
            out[out_key] = np.nan

    # Scalars
    add_scalar("exptime", "exptime", "EXPTIME")
    add_scalar("transparency", "transparency", "THROUGHP")
    add_scalar("millum", "millum", "MILLUM")
    # New ingested environmental parameters
    add_scalar("ambtemp", "ambtemp", "AMBTEMP", "AMBIENT_T", "AMBIENTTEMP")
    add_scalar("humidity", "humidity", "HUMIDITY", "humidty", "HUMID")
    add_scalar("dewpoint", "dewpoint", "DEWPOINT")
    add_scalar("barompre", "BAROMPRE", "barompre", "barometricpressure", "BAROMETRICPRESSURE", "BAROMETRIC", "PRESSURE")
    add_scalar("winddir", "winddir", "WINDDIR", "WIND_DIR")
    add_scalar("windspd", "windspd", "WINDSPD", "WIND_SPEED", "WINDSPEED")
    add_scalar("structaz", "structaz", "STRUCTAZ", "STRUCTURE_AZ", "STRUCTUREAZ")

    # Time and coordinates
    t = _parse_time_from_row(row)
    coord = _parse_radec_from_row(row)

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

    # Airmass at HET
    try:
        altaz = coord.transform_to(AltAz(obstime=t, location=HET_LOCATION))
        out["airmass"] = _airmass(altaz)
    except Exception:
        out["airmass"] = np.nan

    # Sun and Moon geometry
    # Suppress extremely noisy NonRotationTransformationWarning unless debug is enabled
    sun_alt = np.nan
    moon_alt = np.nan
    moon_icrs = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonRotationTransformationWarning)
        try:
            sun = _compat_get_sun(t)
            sun_alt = (sun.transform_to(AltAz(obstime=t, location=HET_LOCATION)).alt.to(u.deg).value)
            sun_icrs = sun.icrs
        except Exception:
            sun_icrs = None
        try:
            moon = _compat_get_moon(t, location=HET_LOCATION)
            moon_alt = (moon.transform_to(AltAz(obstime=t, location=HET_LOCATION)).alt.deg)
            moon_icrs = moon.icrs
        except Exception:
            moon_icrs = None
    out["sun_alt"] = sun_alt
    out["moon_alt"] = moon_alt

    # Separation and illumination (work in a common frame to avoid warnings)
    try:
        if moon_icrs is not None:
            elong = coord.icrs.separation(moon_icrs)
            out["moon_sep"] = elong.deg
        else:
            out["moon_sep"] = np.nan
    except Exception:
        out["moon_sep"] = np.nan

    try:
        if (moon_icrs is not None) and ('sun_icrs' in locals()) and (sun_icrs is not None):
            phase_angle = sun_icrs.separation(moon_icrs).to(u.rad).value
            illum = (1 + np.cos(phase_angle)) / 2
            out["moon_illum"] = float(illum)
        else:
            out["moon_illum"] = np.nan
    except Exception:
        out["moon_illum"] = np.nan

    # Galactic and ecliptic latitude
    try:
        out["glat"] = coord.galactic.b.to(u.deg).value
    except Exception:
        out["glat"] = np.nan

    try:
        ecl = coord.transform_to(BarycentricTrueEcliptic())
        out["elat"] = ecl.lat.to(u.deg).value
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
