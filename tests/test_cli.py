from scMM.cli import build_parser


def test_process_cli_accepts_multiple_references():
    args = build_parser().parse_args(
        [
            "process",
            "input.mzML",
            "output.h5ad",
            "--reference-mz",
            "100.1",
            "200.2",
            "--reference-mode",
            "intersection",
        ]
    )
    assert args.reference_mz == [100.1, 200.2]
    assert args.reference_mode == "intersection"
