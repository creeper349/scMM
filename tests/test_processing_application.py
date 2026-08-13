from pathlib import Path

import pytest

from scMM.application import (
    OutputCatalog,
    OutputRoot,
    ProcessingParameters,
    ProcessingPlanner,
    ProcessingRequest,
    StorageCatalog,
    StorageRoot,
)


def _planner(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "results"
    raw.mkdir()
    output.mkdir()
    source = raw / "sample.mzML"
    source.write_bytes(b"<mzML><spectrum></spectrum></mzML>")
    planner = ProcessingPlanner(
        StorageCatalog((StorageRoot("Raw", raw),)),
        OutputCatalog((OutputRoot("Results", output),)),
    )
    return planner, source, output


def test_processing_parameters_validate_and_expand_presets() -> None:
    params = ProcessingParameters.from_preset("sensitive", 734.5929, n_jobs=2)

    assert params.cell_snr == 3.0
    assert params.max_zero_frac == 0.95
    assert params.load_kwargs()["n_jobs"] == 2

    with pytest.raises(ValueError, match="ref_mz"):
        ProcessingParameters(ref_mz=0)
    with pytest.raises(ValueError, match="max_zero_frac"):
        ProcessingParameters(ref_mz=100, max_zero_frac=1.1)
    with pytest.raises(ValueError, match="n_jobs"):
        ProcessingParameters(ref_mz=100, n_jobs=0)
    with pytest.raises(ValueError, match="Unknown"):
        ProcessingParameters.from_preset("unknown", 100)  # type: ignore[arg-type]


def test_processing_preflight_resolves_safe_input_and_output(tmp_path: Path) -> None:
    planner, source, output = _planner(tmp_path)
    request = ProcessingRequest(
        storage_label="Raw",
        input_path="sample.mzML",
        output_label="Results",
        parameters=ProcessingParameters(ref_mz=100),
    )

    plan = planner.preflight(request)

    assert plan.input_path == source.resolve()
    assert plan.output_root == output.resolve()
    assert plan.result_path == output.resolve() / "sample"
    assert plan.input_size_bytes == len(b"<mzML><spectrum></spectrum></mzML>")


def test_processing_preflight_rejects_truncated_input(tmp_path: Path) -> None:
    planner, source, _ = _planner(tmp_path)
    source.write_bytes(b"<indexedmzML><mzML><spectrum></spectrum></mzML>")
    request = ProcessingRequest(
        storage_label="Raw",
        input_path="sample.mzML",
        output_label="Results",
        parameters=ProcessingParameters(ref_mz=100),
    )

    with pytest.raises(ValueError, match=r"XML 文档不完整.*重新转换"):
        planner.preflight(request)


def test_processing_preflight_rejects_conflicts_and_output_escape(tmp_path: Path) -> None:
    planner, _, output = _planner(tmp_path)
    (output / "sample").mkdir()
    request = ProcessingRequest(
        storage_label="Raw",
        input_path="sample.mzML",
        output_label="Results",
        parameters=ProcessingParameters(ref_mz=100),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        planner.preflight(request)
    with pytest.raises(ValueError, match="Invalid result name"):
        planner.outputs.resolve_target("Results", "../escape")

    overwrite_plan = planner.preflight(ProcessingRequest(**{**request.__dict__, "overwrite": True}))
    assert "overwritten" in overwrite_plan.warnings[0]


def test_output_catalog_rejects_hidden_names_and_symlink_targets(tmp_path: Path) -> None:
    output = tmp_path / "results"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    catalog = OutputCatalog((OutputRoot("Results", output),))

    with pytest.raises(ValueError, match="Invalid result name"):
        catalog.resolve_target("Results", ".scmm-tasks")

    (output / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="symbolic link"):
        catalog.resolve_target("Results", "escape")


def test_output_catalog_rejects_duplicate_and_unknown_roots(tmp_path: Path) -> None:
    root = OutputRoot("Results", tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        OutputCatalog((root, root))
    with pytest.raises(KeyError, match="Unknown output root"):
        OutputCatalog((root,)).root("Missing")
