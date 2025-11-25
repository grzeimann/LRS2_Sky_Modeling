# lrs2sky

Minimal, modular tools to build a Cannon-style forward model of the night-sky spectrum for HET/LRS2.

Quick start
- Create a conda environment and install in editable mode.
- Ingest FITS files into a CSV index.
- Prototype label extraction and modeling.

Environment
1. conda env create -f environment.yml
2. conda activate lrs2sky
3. pip install -e .

CLI usage
- lrs2sky version
- lrs2sky ingest /data/root --date 20220101 --ndays 90 --channel orange --csv-out index_orange.csv
- lrs2sky labels index_orange.csv --out labels_orange.parquet
- lrs2sky spectra index_orange.csv --channel orange --out spectra_orange.h5 --central-width 200

Channel selection
- Channels supported: uv, orange, red, farred.
- Ingest: pass --channel to limit file discovery and to embed an inferred channel column in the CSV. Pattern supports {date} and {channel}.
- Spectra: pass --channel to only process exposures from that channel and to ensure a consistent wavelength grid.
- Channel is inferred from FITS header (CHANNEL/ARM/INSTRUME/DETECTOR) or from the filename.

What gets extracted
- Header labels: airmass, Sun/Moon geometry, gal/ecl latitudes, day-of-year encodings, exposure time, arm id, and transparency via THROUGHP.
- Sky spectra: biweight coadd across all 280 fibers from HDU[0] at each wavelength; wavelength taken from HDU[6].data[0].
- Normalization: biweight of the central 200 pixels (configurable via --central-width); the scalar and pixel bounds are saved.
- Output format: HDF5 written with PyTables: datasets wave (n_wave), flux (n_exp,n_wave), norm (n_exp), pix_bounds (n_exp,2), and exp_path (VLArray of UTF-8 strings).

Python usage
- from lrs2sky.ingest import find_sky_files
- from lrs2sky.labels import compute_labels_from_header
- from lrs2sky.spectrum import extract_sky_biweight, load_spectra_hdf5
- from lrs2sky.model import fit_basis, fit_coeff_models, predict_coeffs, reconstruct_spectrum

Notes
- Functions are initial implementations and designed to be extended per skycannon_instructions.md.
- Label computations use astropy and may require internet-free IERS settings for consistent alt/az; adjust as needed.
- For a new location (different data root), point ingest to your base folder(s); the tools do not assume a specific mount path.

Installation on older pip
- If you see an error like "File 'setup.py' or 'setup.cfg' not found... editable mode requires a setuptools-based build" on older pip (e.g., 21.x), we now include setup.cfg/setup.py for compatibility. Two options:
  1) Upgrade pip: python -m pip install --upgrade pip
  2) Use the provided setup.cfg by running: pip install -e .
- The package also supports modern PEP 517 builds via pyproject.toml on newer pip versions.