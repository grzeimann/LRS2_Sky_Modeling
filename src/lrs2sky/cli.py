import argparse
import sys
from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np

from . import __version__
from .ingest import find_sky_files, infer_channel
from .labels import compute_labels_from_header
from .spectrum import extract_sky_biweight, save_spectra_hdf5


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"lrs2sky {__version__}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    folders = args.folders
    if not folders:
        print("Please provide at least one folder to search", file=sys.stderr)
        return 2
    df = find_sky_files(
        folders=folders,
        pattern=args.pattern,
        date=args.date,
        ndays=args.ndays,
        min_exptime=args.min_exptime,
        csv_out=args.csv_out,
        channel=args.channel,
    )
    if args.csv_out:
        print(f"Wrote {len(df)} rows to {args.csv_out}")
    else:
        # print a small preview to stdout
        with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
            print(df.head(20))
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.index)
    rows = []
    errors = 0
    for i, row in df.iterrows():
        path = row.get("path") or row.get("PATH")
        if not path:
            errors += 1
            continue
        try:
            with fits.open(path) as hdul:
                lab = compute_labels_from_header(hdul[0].header)
        except Exception:
            errors += 1
            continue
        lab["path"] = path
        rows.append(lab)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.parquet or args.out.lower().endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"Wrote labels for {len(out)} exposures to {args.out} (errors: {errors})")
    return 0


def cmd_spectra(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.index)
    exp_paths = []
    flux_list = []
    norms = []
    pix_bounds = []
    wave_ref = None
    n_err = 0
    used_channel = args.channel
    for i, row in df.iterrows():
        path = row.get("path") or row.get("PATH")
        if not path:
            n_err += 1
            continue
        # If channel filter specified, validate/infer and skip mismatches
        if used_channel:
            try:
                with fits.open(path) as hdul:
                    ch_row = infer_channel(hdul[0].header, path)
            except Exception:
                ch_row = None
            if ch_row and ch_row != used_channel:
                continue
        wave, flux, norm, pb = extract_sky_biweight(path, central_width=args.central_width)
        if wave is None or flux is None:
            n_err += 1
            continue
        if wave_ref is None:
            wave_ref = wave
        else:
            # ensure same wavelength grid length; if mismatch, skip
            if len(wave) != len(wave_ref) or (np.nanmax(np.abs(wave - wave_ref)) > 1e-6):
                n_err += 1
                continue
        exp_paths.append(path)
        flux_list.append(flux)
        norms.append(norm if norm is not None else np.nan)
        pb = pb or (np.int64(0), np.int64(0))
        pix_bounds.append((int(pb[0]), int(pb[1])))
    if wave_ref is None or not flux_list:
        print("No spectra extracted; aborting.", file=sys.stderr)
        return 2
    flux2d = np.vstack([np.asarray(f, dtype=float) for f in flux_list])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_spectra_hdf5(args.out, wave_ref, flux2d, exp_paths, norms, pix_bounds)
    print(f"Wrote {flux2d.shape[0]} spectra to {args.out} (n_wave={flux2d.shape[1]}, errors: {n_err})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lrs2sky", description="LRS2 sky modeling utilities")
    sub = p.add_subparsers(dest="command")

    p_ver = sub.add_parser("version", help="Print version and exit")
    p_ver.set_defaults(func=cmd_version)

    p_ing = sub.add_parser("ingest", help="Ingest FITS files and write CSV index")
    p_ing.add_argument("folders", nargs="*", help="Folders to search")
    p_ing.add_argument("--pattern", default="multi*{date}*{channel}.fits", help="Filename pattern with {date} and {channel} placeholders (channel will be '*' if not provided)")
    p_ing.add_argument("--date", default="20220101", help="Center date YYYYMMDD")
    p_ing.add_argument("--ndays", type=int, default=365, help="Number of days to search centered on date")
    p_ing.add_argument("--min-exptime", type=float, default=300.0, help="Minimum exposure time in seconds")
    p_ing.add_argument("--csv-out", default=None, help="Path to CSV output")
    p_ing.add_argument("--channel", choices=["uv","orange","red","farred"], help="Limit ingestion to a specific LRS2 channel")
    p_ing.set_defaults(func=cmd_ingest)

    p_lbl = sub.add_parser("labels", help="Batch extract labels from index CSV to Parquet/CSV")
    p_lbl.add_argument("index", help="Path to index CSV with a 'path' column")
    p_lbl.add_argument("--out", required=True, help="Output Parquet or CSV path")
    p_lbl.add_argument("--parquet", action="store_true", help="Force Parquet output (otherwise inferred from extension)")
    p_lbl.set_defaults(func=cmd_labels)

    p_spec = sub.add_parser("spectra", help="Extract biweight sky spectra and save HDF5")
    p_spec.add_argument("index", help="Path to index CSV with a 'path' column")
    p_spec.add_argument("--out", required=True, help="Output HDF5 path (*.h5)")
    p_spec.add_argument("--central-width", type=int, default=200, help="Central pixel width for normalization biweight")
    p_spec.add_argument("--channel", choices=["uv","orange","red","farred"], help="Only process exposures matching this channel (validated via header/path)")
    p_spec.set_defaults(func=cmd_spectra)

    return p


essential_commands = {"version", "ingest", "labels", "spectra"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
