import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .ingest import find_sky_files


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
    )
    if args.csv_out:
        print(f"Wrote {len(df)} rows to {args.csv_out}")
    else:
        # print a small preview to stdout
        with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
            print(df.head(20))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lrs2sky", description="LRS2 sky modeling utilities")
    sub = p.add_subparsers(dest="command")

    p_ver = sub.add_parser("version", help="Print version and exit")
    p_ver.set_defaults(func=cmd_version)

    p_ing = sub.add_parser("ingest", help="Ingest FITS files and write CSV index")
    p_ing.add_argument("folders", nargs="*", help="Folders to search")
    p_ing.add_argument("--pattern", default="multi*{date}*orange.fits", help="Filename pattern with {date} placeholder")
    p_ing.add_argument("--date", default="20220101", help="Center date YYYYMMDD")
    p_ing.add_argument("--ndays", type=int, default=365, help="Number of days to search centered on date")
    p_ing.add_argument("--min-exptime", type=float, default=300.0, help="Minimum exposure time in seconds")
    p_ing.add_argument("--csv-out", default=None, help="Path to CSV output")
    p_ing.set_defaults(func=cmd_ingest)

    return p


essential_commands = {"version", "ingest"}


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
