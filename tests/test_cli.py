import pytest

from scMM.cli import build_parser


def test_cli_requires_reference_mz() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.mzML", "output"])


def test_cli_parses_processing_options() -> None:
    args = build_parser().parse_args(
        ["raw", "results", "--ref-mz", "734.5929", "--jobs", "4", "--overwrite"]
    )

    assert args.ref_mz == 734.5929
    assert args.jobs == 4
    assert args.overwrite is True
