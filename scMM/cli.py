"""Command-line entry points for scMM data processing."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .file.data import CyESIData


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="scmm-process",
        description="Process a CyESI mzML/mzXML file or a directory of files.",
    )
    parser.add_argument("input", type=Path, help="Input mzML/mzXML file or directory")
    parser.add_argument("output", type=Path, help="Directory in which to save the result")
    parser.add_argument("--ref-mz", type=float, required=True, help="Reference ion m/z")
    parser.add_argument("--ppm-tol", type=float, default=10.0, help="Alignment tolerance in ppm")
    parser.add_argument("--resolution", type=float, default=35_000.0)
    parser.add_argument("--cell-snr", type=float, default=5.0)
    parser.add_argument("--peak-snr", type=float, default=3.0)
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 uses all CPUs)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory with the same dataset name",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line data processing workflow."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    common = {
        "ref_mz": args.ref_mz,
        "ppm_tol": args.ppm_tol,
        "resolution": args.resolution,
        "cell_snr": args.cell_snr,
        "peak_snr": args.peak_snr,
    }
    if input_path.is_dir():
        data = CyESIData.load_from_filelist(input_path, n_jobs=args.jobs, **common)
    else:
        data = CyESIData.load_from_file(input_path, **common)

    result_path = data.save(args.output, overwrite=args.overwrite)
    logging.getLogger(__name__).info("Saved processed dataset to %s", result_path)
    return 0
