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
# Prefer built-in Moon illumination when available for correctness across versions
try:
    from astropy.coordinates import moon_illumination as _moon_illumination  # type: ignore
except Exception:
    try:
        from astropy.coordinates.moon import moon_illumination as _moon_illumination  # type: ignore
    except Exception:
        _moon_illumination = None  # type: ignore

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
        # Normalize a leading '+' in the time component (e.g., 'YYYY-MM-DDT+HH:MM:SS')
        if "T" in val:
            d, tpart = val.split("T", 1)
            tpart = tpart.lstrip("+")
            val = f"{d}T{tpart}"
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
        tstr = str(time).strip().lstrip("+")
        val = f"{str(date).strip()}T{tstr}"
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


def _compute_moon_illumination(
    t: Time,
    location: EarthLocation,
    sun_icrs,
    moon_icrs,
    sun_obj,
    moon_obj,
) -> Optional[float]:
    """Return fraction of lunar disk illuminated in [0,1].

    Preference order:
    1) astropy.coordinates.moon_illumination (version-dependent location handling)
    2) Phase-angle formula using distances: k = (1 + cos i)/2, with
       i = atan2(r_sun*sin ψ, r_moon - r_sun*cos ψ)
    3) Geocentric elongation approximation: k ≈ (1 - cos ψ)/2
    """
    # 1) Built-in if available
    try:
        if _moon_illumination is not None:
            # Some versions accept only Time; others accept (Time, location)
            try:
                k = _moon_illumination(t, location=location)  # type: ignore[arg-type]
            except Exception:
                k = _moon_illumination(t)  # type: ignore[misc]
            return float(np.clip(float(k), 0.0, 1.0))
    except Exception:
        pass
    # Require sun/moon positions for fallbacks
    if sun_icrs is None or moon_icrs is None:
        return None
    try:
        psi = sun_icrs.separation(moon_icrs).to(u.rad).value
        # 2) Phase-angle with distances if available
        try:
            r_sun = float(getattr(sun_obj, 'distance').to(u.AU).value)
            r_moon = float(getattr(moon_obj, 'distance').to(u.AU).value)
            num = r_sun * np.sin(psi)
            den = (r_moon - r_sun * np.cos(psi))
            i = float(np.arctan2(num, den))
            k = 0.5 * (1.0 + np.cos(i))
        except Exception:
            # 3) Geocentric elongation approximation
            k = 0.5 * (1.0 - np.cos(psi))
        return float(np.clip(k, 0.0, 1.0))
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
    - optional extinction coefficient for KSfeature: k, k_ext, extinction

    Returns a dict with keys: airmass, sun_alt, moon_alt, moon_illum,
    moon_sep, moon_airmass, KSfeature, glat, elat, doy_sin, doy_cos,
    exptime, transparency, millum, ambtemp, humidity, dewpoint, barompre,
    winddir, windspd.
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
    add_scalar("airmass", "airmass", "AIRMASS", "AIR_MASS", "air_mass")

    # Optional extinction coefficient (defaults applied later if NaN)
    add_scalar("k_ext", "k", "k_ext", "extinction", "EXTINCTION")
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
            "sun_alt",
            "moon_alt",
            "moon_illum",
            "moon_sep",
            "moon_airmass",
            "KSfeature",
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
        airmass = _airmass(altaz)
        prev_airmass = out.get("airmass", np.nan)
        if (airmass is not None) and np.isfinite(prev_airmass) and np.isfinite(airmass):
            airmass_difference = float(np.abs(airmass - prev_airmass))
            if airmass_difference > 0.1:
                print(f"Header airmass {prev_airmass:.2f}, Date-Obs airmass {airmass:.2f}, airmass difference {airmass_difference:.2f}")
        out["airmass"] = airmass if airmass is not None else np.nan
    except Exception:
        out["airmass"] = np.nan

    # Sun and Moon geometry
    # Suppress extremely noisy NonRotationTransformationWarning unless debug is enabled
    sun_alt = np.nan
    moon_alt = np.nan
    sun_obj = None
    moon_obj = None
    moon_icrs = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonRotationTransformationWarning)
        try:
            sun = _compat_get_sun(t)
            sun_alt = (sun.transform_to(AltAz(obstime=t, location=HET_LOCATION)).alt.to(u.deg).value)
            sun_icrs = sun.icrs
            sun_obj = sun
        except Exception:
            sun_icrs = None
            sun_obj = None
        try:
            moon = _compat_get_moon(t, location=HET_LOCATION)
            moon_alt = (moon.transform_to(AltAz(obstime=t, location=HET_LOCATION)).alt.deg)
            moon_icrs = moon.icrs
            moon_obj = moon
        except Exception:
            moon_icrs = None
            moon_obj = None
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
        k = _compute_moon_illumination(t, HET_LOCATION, sun_icrs, moon_icrs, sun_obj, moon_obj)
        out["moon_illum"] = float(k) if k is not None and np.isfinite(k) else np.nan
    except Exception:
        out["moon_illum"] = np.nan

    # Moon airmass from moon altitude (only when above horizon)
    try:
        if np.isfinite(moon_alt) and (moon_alt > 0) and (moon_alt < 89.999):
            z_rad = np.radians(90.0 - float(moon_alt))
            secz = 1.0 / np.cos(z_rad)
            out["moon_airmass"] = float(secz)
        else:
            out["moon_airmass"] = 100.00
    except Exception:
        out["moon_airmass"] = np.nan

    # KSfeature (L_moon) per provided formula
    try:
        illum = out.get("moon_illum", np.nan)
        alt_term = float(np.clip(np.sin(np.radians(moon_alt)), 0.0, None)) if np.isfinite(moon_alt) else np.nan
        sep = out.get("moon_sep", np.nan)
        sep_term = 1.0 / (1.0 + (float(sep) / 25.0) ** 2) if np.isfinite(sep) else np.nan
        k = out.get("k_ext", np.nan)
        if not np.isfinite(k):
            k = 0.15  # default atmospheric extinction coefficient
            out["k_ext"] = k
        m_air = out.get("moon_airmass", np.nan)
        trans_term = float(np.exp(-k * float(m_air))) if np.isfinite(k) and np.isfinite(m_air) else np.nan
        if np.isfinite(illum) and np.isfinite(alt_term) and np.isfinite(trans_term) and np.isfinite(sep_term):
            out["KSfeature"] = float(illum) * alt_term * trans_term * sep_term
        else:
            out["KSfeature"] = np.nan
    except Exception:
        out["KSfeature"] = np.nan

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
