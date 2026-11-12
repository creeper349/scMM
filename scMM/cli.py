import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import render_quality_report
from .file.data import CyESIData


def _metadata_table(path):
    if path is None:
        return None
    table = pd.read_csv(path)
    if "source_file" not in table:
        raise ValueError("metadata CSV must contain a source_file column")
    return {
        str(row["source_file"]): {
            key: value
            for key, value in row.items()
            if key != "source_file" and pd.notna(value)
        }
        for _, row in table.iterrows()
    }


def _process(args):
    source = Path(args.input)
    kwargs = {
        "reference_mz": args.reference_mz,
        "reference_mode": args.reference_mode,
        "frame_metadata": args.frame_metadata,
        "metadata_by_file": _metadata_table(args.metadata),
        "ppm_tol": args.ppm,
        "cell_signal_threshold": args.cell_threshold,
        "feature_snr_threshold": args.feature_snr,
        "show_progress": not args.quiet,
        "msconvert_path": args.msconvert,
    }
    if source.is_dir():
        data = CyESIData.load_from_directory(
            source,
            feature_strategy=args.feature_strategy,
            feature_merge_ppm=args.feature_merge_ppm,
            **kwargs,
        )
    else:
        data = CyESIData.load_from_file(source, **kwargs)
    output = data.save(args.output)
    print(output)
    return 0


def _qc(args):
    data = CyESIData.read(args.input)
    output = render_quality_report(data, args.output)
    print(output)
    return 0


def _export(args):
    data = CyESIData.read(args.input)
    output = data.export_csv(args.output)
    print(output)
    return 0


def _inspect(args):
    data = CyESIData.read(args.input)
    summary = {
        "cells": data.n_obs,
        "features": data.n_vars,
        "sources": sorted(data.obs.get("source_file", pd.Series(dtype=str)).astype(str).unique()),
        "obs_columns": data.obs.columns.tolist(),
        "var_columns": data.var.columns.tolist(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="scmm",
        description="CyESI single-cell mass-spectrometry processing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="process mzML/mzXML/vendor RAW data")
    process.add_argument("input", help="input file or directory")
    process.add_argument("output", help="output H5AD file")
    process.add_argument(
        "--reference-mz",
        nargs="+",
        type=float,
        required=True,
        help="one or more reference m/z values",
    )
    process.add_argument(
        "--reference-mode",
        choices=["union", "intersection"],
        default="union",
    )
    process.add_argument("--ppm", type=float, default=10.0)
    process.add_argument("--cell-threshold", type=float, default=5.0)
    process.add_argument("--feature-snr", type=float, default=3.0)
    process.add_argument(
        "--feature-strategy",
        choices=["shared", "independent"],
        default="shared",
        help="shared summed axis or independent per-file peak picking",
    )
    process.add_argument("--feature-merge-ppm", type=float, default=10.0)
    process.add_argument(
        "--msconvert",
        help="legacy compatibility option; Thermo RAW files use direct RawFileReader access",
    )
    process.add_argument(
        "--frame-metadata",
        nargs="*",
        default=[],
        choices=["frame_id", "rt", "acquisition_timestamp", "acquisition_time"],
    )
    process.add_argument(
        "--metadata",
        help="CSV with source_file plus experiment metadata columns",
    )
    process.add_argument("--quiet", action="store_true")
    process.set_defaults(func=_process)

    qc = subparsers.add_parser("qc", help="create a PDF quality report")
    qc.add_argument("input", help="input H5AD file")
    qc.add_argument("output", help="output PDF file")
    qc.set_defaults(func=_qc)

    export = subparsers.add_parser("export", help="export X/obs/var CSV files")
    export.add_argument("input", help="input H5AD file")
    export.add_argument("output", help="output directory")
    export.set_defaults(func=_export)

    inspect = subparsers.add_parser("inspect", help="show dataset summary")
    inspect.add_argument("input", help="input H5AD file")
    inspect.set_defaults(func=_inspect)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
