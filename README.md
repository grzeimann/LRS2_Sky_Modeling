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
- lrs2sky ingest /path/to/data --date 20220101 --ndays 90 --csv-out index.csv

Python usage
- from lrs2sky.ingest import find_sky_files
- from lrs2sky.labels import compute_labels_from_header
- from lrs2sky.spectrum import extract_sky_spectrum
- from lrs2sky.model import fit_basis, fit_coeff_models, predict_coeffs, reconstruct_spectrum

Notes
- Functions are initial implementations and designed to be extended per skycannon_instructions.md.
- Label computations use astropy and may require internet-free IERS settings for consistent alt/az; adjust as needed.