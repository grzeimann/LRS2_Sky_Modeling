from __future__ import annotations

from typing import Optional, Tuple, Sequence, List

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location

try:
    import tables as tb  # optional dependency for spectral I/O (PyTables)
except Exception:  # pragma: no cover - optional
    tb = None

__all__ = [
    "extract_sky_biweight",
    "save_spectra_hdf5",
    "load_spectra_hdf5",
]


def extract_sky_biweight(
    path: str,
    central_width: int = 200,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[Tuple[int, int]]]:
    """Extract sky spectrum by robustly coadding fibers via biweight.

    Implementation choices per plan:
    - Use HDU[0].data with shape (n_fiber=280, n_wave)
    - Use wavelength from HDU[6].data[0]
    - Compute biweight_location across fibers at each wavelength with ignore_nan=True
    - Normalize by biweight of the central `central_width` pixels; return the norm and pixel bounds

    Returns
    -------
    wave : array or None
        Wavelength array of length n_wave.
    flux : array or None
        Normalized sky flux of length n_wave.
    norm : float or None
        Normalization scalar applied to raw coadd.
    pix_bounds : (i0, i1) or None
        Integer pixel bounds used for normalization.
    """
    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is None or data.ndim != 2:
                return None, None, None, None
            # data shape expected (n_fiber, n_wave)
            data = np.asarray(data, dtype=float)
            # wavelength from HDU[6]
            try:
                wave = np.asarray(hdul[6].data[0], dtype=float)
            except Exception:
                wave = None
            # robust coadd across fibers (axis=0 fibers -> result length n_wave)
            sky = biweight_location(data, axis=0, ignore_nan=True)
            # normalization using central pixels by index
            n_wave = sky.shape[0]
            w = int(central_width)
            w = max(1, min(w, n_wave))
            half = w // 2
            mid = n_wave // 2
            i0 = int(max(0, mid - half))
            i1 = int(min(n_wave, i0 + w))
            try:
                norm = float(biweight_location(sky[i0:i1], ignore_nan=True))
            except Exception:
                norm = float(np.nanmedian(sky[i0:i1])) if np.isfinite(np.nanmedian(sky[i0:i1])) else np.nan
            if np.isfinite(norm) and norm > 0:
                sky_norm = sky / norm
            else:
                sky_norm = sky
            return wave, sky_norm, norm, (i0, i1)
    except Exception:
        return None, None, None, None


def save_spectra_hdf5(
    out_path: str,
    wave: np.ndarray,
    flux2d: np.ndarray,
    exp_paths: Sequence[str],
    norms: Sequence[float],
    pix_bounds: Sequence[Tuple[int, int]],
) -> None:
    """Save spectra matrix and metadata to HDF5 using PyTables.

    Nodes:
    - /wave: CArray float64 (n_wave,)
    - /flux: CArray float64 (n_exp, n_wave)
    - /norm: CArray float64 (n_exp,)
    - /pix_bounds: CArray int64 (n_exp, 2)
    - /exp_path: VLArray of UTF-8 strings
    """
    if tb is None:
        raise RuntimeError("PyTables (tables) is required to write HDF5 outputs. Please install 'tables' or 'pytables'.")
    exp_paths = list(exp_paths)
    norms = np.asarray(list(norms), dtype=np.float64)
    pix = np.asarray(list(pix_bounds), dtype=np.int64)
    wave = np.asarray(wave, dtype=np.float64)
    flux2d = np.asarray(flux2d, dtype=np.float64)
    with tb.open_file(out_path, mode="w") as h5:
        atom_f = tb.Float64Atom()
        atom_i = tb.Int64Atom()
        # wave
        ca_wave = h5.create_carray(h5.root, "wave", atom_f, wave.shape)
        ca_wave[:] = wave
        # flux
        ca_flux = h5.create_carray(h5.root, "flux", atom_f, flux2d.shape)
        ca_flux[:] = flux2d
        # norm
        ca_norm = h5.create_carray(h5.root, "norm", atom_f, norms.shape)
        ca_norm[:] = norms
        # pix_bounds
        ca_pix = h5.create_carray(h5.root, "pix_bounds", atom_i, pix.shape)
        ca_pix[:] = pix
        # exp_path as variable-length UTF-8 strings
        vl = h5.create_vlarray(h5.root, "exp_path", tb.VLStringAtom())
        for s in exp_paths:
            if s is None:
                s = ""
            vl.append(str(s))


def load_spectra_hdf5(in_path: str):
    """Load spectra matrix and metadata from HDF5 using PyTables.

    Returns a dict with keys: wave, flux, norm, pix_bounds, exp_path
    """
    if tb is None:
        raise RuntimeError("PyTables (tables) is required to read HDF5 outputs. Please install 'tables' or 'pytables'.")
    with tb.open_file(in_path, mode="r") as h5:
        wave = h5.root.wave.read()
        flux = h5.root.flux.read()
        norm = h5.root.norm.read()
        pix_bounds = h5.root.pix_bounds.read()
        exp_list = h5.root.exp_path.read()
        # Ensure list of Python strings
        exp_path = [e.decode("utf-8") if isinstance(e, (bytes, bytearray)) else str(e) for e in exp_list]
        return {
            "wave": wave,
            "flux": flux,
            "norm": norm,
            "pix_bounds": pix_bounds,
            "exp_path": np.array(exp_path, dtype=object),
        }
