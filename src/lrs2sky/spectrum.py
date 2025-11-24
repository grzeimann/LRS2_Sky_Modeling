from __future__ import annotations

from typing import Optional, Tuple, Sequence

import numpy as np
from astropy.io import fits

__all__ = [
    "extract_sky_spectrum",
    "rebin_to_grid",
]


def rebin_to_grid(wave: np.ndarray, flux: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Simple linear interpolation onto a common wavelength grid.

    Extrapolated values are set to NaN.
    """
    if wave is None or flux is None or len(wave) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    y = np.interp(grid, wave, flux, left=np.nan, right=np.nan)
    return y


def _guess_wavelength(header) -> Optional[np.ndarray]:
    # Try common WCS keywords: CRVAL1, CDELT1, CRPIX1, NAXIS1
    try:
        n = int(header.get("NAXIS1"))
        crval = float(header.get("CRVAL1"))
        cdelt = float(header.get("CDELT1"))
        crpix = float(header.get("CRPIX1", 1.0))
        pix = np.arange(1, n + 1)
        wave = crval + (pix - crpix) * cdelt
        return wave
    except Exception:
        return None


def extract_sky_spectrum(
    path: str,
    wavelength_grid: Optional[np.ndarray] = None,
    normalize: bool = True,
    continuum_region: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract a 1D sky spectrum from a FITS file.

    This is a minimal implementation. If a 1D spectrum is found, it is returned.
    If a wavelength grid is supplied, the spectrum is rebinned/interpolated.

    Returns (wave, flux) where either may be None if extraction fails.
    """
    try:
        with fits.open(path) as hdul:
            # Heuristic: look for first 1D data array
            wave = None
            flux = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                data = hdu.data
                if data is None:
                    continue
                if data.ndim == 1 and data.size > 10:
                    flux = np.asarray(data, dtype=float)
                    wave = _guess_wavelength(hdu.header) or wave
                    break
            if flux is None:
                return None, None
            if normalize:
                f = flux.copy()
                if continuum_region is not None and wave is not None:
                    lo, hi = continuum_region
                    m = (wave >= lo) & (wave <= hi)
                    med = np.nanmedian(f[m]) if m.any() else np.nanmedian(f)
                else:
                    med = np.nanmedian(f)
                if np.isfinite(med) and med > 0:
                    flux = f / med
            if wavelength_grid is not None:
                if wave is None:
                    # cannot rebin without wave; return NaNs
                    return wavelength_grid, np.full_like(wavelength_grid, np.nan)
                flux = rebin_to_grid(wave, flux, wavelength_grid)
                wave = wavelength_grid
            return wave, flux
    except Exception:
        return None, None
