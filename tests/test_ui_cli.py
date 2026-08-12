from pathlib import Path

import pytest

from scMM.ui.cli import build_parser, parse_output_root, parse_storage_root


def test_parse_storage_root_supports_labels_and_paths() -> None:
    label, path = parse_storage_root(" 原始数据 = /mnt/lab data ")

    assert label == "原始数据"
    assert path == Path("/mnt/lab data")


@pytest.mark.parametrize("value", ["raw", "=/mnt/raw", "Raw="])
def test_parse_storage_root_rejects_invalid_syntax(value) -> None:
    with pytest.raises(Exception, match="LABEL=PATH"):
        parse_storage_root(value)


def test_parse_output_root_supports_named_paths() -> None:
    label, path = parse_output_root("Results=/mnt/scmm-results")

    assert label == "Results"
    assert path == Path("/mnt/scmm-results")


def test_ui_cli_parses_server_options() -> None:
    args = build_parser().parse_args(
        [
            "--storage",
            "Raw=/mnt/raw",
            "--storage",
            "Archive=/mnt/archive",
            "--address",
            "0.0.0.0",
            "--output",
            "Results=/tmp/results",
            "--port",
            "5100",
            "--allow-websocket-origin",
            "scmm-node:5100",
        ]
    )

    assert args.storage == [("Raw", Path("/mnt/raw")), ("Archive", Path("/mnt/archive"))]
    assert args.address == "0.0.0.0"
    assert args.output == [("Results", Path("/tmp/results"))]
    assert args.port == 5100
    assert args.allow_websocket_origin == ["scmm-node:5100"]
