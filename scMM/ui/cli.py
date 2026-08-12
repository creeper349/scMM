"""Command-line launcher for the guided scMM web interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scMM.application import StorageRoot


def parse_storage_root(value: str) -> tuple[str, Path]:
    """Parse a LABEL=PATH storage-root specification."""
    label, separator, path_text = value.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("storage must use LABEL=PATH, for example Raw=/mnt/ms")
    return label.strip(), Path(path_text.strip()).expanduser()


def build_parser() -> argparse.ArgumentParser:
    """Build the web-server argument parser."""
    parser = argparse.ArgumentParser(
        prog="scmm-ui",
        description="Launch the guided scMM raw-data preview web interface.",
    )
    parser.add_argument(
        "--storage",
        action="append",
        type=parse_storage_root,
        metavar="LABEL=PATH",
        help="Server-mounted directory exposed to the file selector; repeat for multiple roots",
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="Listening address; use 0.0.0.0 for LAN/Tailscale access",
    )
    parser.add_argument("--port", type=int, default=5006, help="Listening port (default: 5006)")
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        metavar="HOST[:PORT]",
        help="Additional browser origin accepted by Panel; repeat when needed",
    )
    parser.add_argument("--show", action="store_true", help="Open a local browser after launch")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and run the Panel server."""
    args = build_parser().parse_args(argv)
    root_specs = args.storage or [("当前目录", Path.cwd())]
    roots = tuple(StorageRoot(label, path) for label, path in root_specs)

    try:
        import panel as pn
    except ImportError as exc:
        raise RuntimeError(
            "Web UI dependencies are not installed; install scMM[ui] or use environment.yml"
        ) from exc

    from .app import create_app

    serve_options: dict[str, object] = {
        "address": args.address,
        "port": args.port,
        "show": args.show,
        "title": "scMM 数据查看",
    }
    if args.allow_websocket_origin:
        serve_options["websocket_origin"] = args.allow_websocket_origin
    pn.serve(lambda: create_app(roots), **serve_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
