# lrs2sky

Minimal, modular tools to build a Cannon-style forward model of the night-sky spectrum for HET/LRS2.

Quick start
- Create a conda environment and install in editable mode.
- Ingest a channel-specific list into a CSV index.
- Compute labels from the CSV and prototype modeling.

Environment
1. conda env create -f environment.yml
2. conda activate lrs2sky
3. pip install -e .

Notes on Python versions
- Supports Python 3.9 and newer. If you are on a legacy system with Python 3.9, the included setup.cfg enables editable installs with older pip versions.

CLI usage
- lrs2sky version
- lrs2sky ingest --channel orange --csv-out index/index_orange.csv
  (reads archive/orange_file_list.txt by default; use --archive-dir or --list-path to override)
- lrs2sky labels index/index_orange.csv --out label/label_orange.parquet
- lrs2sky spectra index/index_orange.csv --channel orange --out spectra/spectra_orange.h5 --central-width 200

Ingest (channel list based)
- Channels supported: uv, orange, red, farred.
- The ingest command loads a prebuilt whitespace-delimited list file from the archive:
  archive/<channel>_file_list.txt
  Each row is expected to have: path  object  date  time  exptime  airmass  RA  DEC  THROUGHP
- Output CSV schema (columns): path, object, exptime, dateobs, ra, dec, channel, transparency.
- You can point to a custom list file with --list-path, or change the directory via --archive-dir.

What gets labeled
- The labels command reads the index CSV (no FITS I/O) and computes per-exposure metadata:
  Sun/Moon geometry, moon illumination, galactic/ecliptic latitudes, day-of-year encodings, airmass.

Python usage
- from lrs2sky.ingest import load_channel_index
- from lrs2sky.labels import compute_labels_from_row
- from lrs2sky.spectrum import extract_sky_biweight, load_spectra_hdf5, save_spectra_hdf5
- from lrs2sky.model import fit_basis, fit_coeff_models, predict_coeffs, reconstruct_spectrum

Notes
- Functions are initial implementations and designed to be extended per skycannon_plan.md.
- Label computations use astropy and may require internet-free IERS settings for consistent alt/az; adjust as needed.
- Ingest no longer scans directories; it reads channel list files under archive/ and filters to the requested channel.

Installation on older pip
- If you see an error like "File 'setup.py' or 'setup.cfg' not found... editable mode requires a setuptools-based build" on older pip (e.g., 21.x), we now include setup.cfg/setup.py for compatibility. Two options:
  1) Upgrade pip: python -m pip install --upgrade pip
  2) Use the provided setup.cfg by running: pip install -e .
- The package also supports modern PEP 517 builds via pyproject.toml on newer pip versions.