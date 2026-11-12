import pytest
from pathlib import Path

from scMM.file.data import CyESIData
from scMM.workflows import ProcessingConfig, parse_reference_mz, run_processing


def test_parse_reference_mz_and_config():
    assert parse_reference_mz("100.1, 200.2;300.3") == [100.1, 200.2, 300.3]
    config = ProcessingConfig("input", "output.h5ad", [100.1])
    assert config.feature_strategy == "shared"
    assert (config.mz_min, config.mz_max) == (100.0, 1000.0)


def test_parse_reference_mz_rejects_empty_input():
    with pytest.raises(ValueError):
        parse_reference_mz(" , ")


def test_processing_range_validation_happens_before_extraction(tmp_path):
    source = tmp_path / "input.mzML"
    source.touch()
    config = ProcessingConfig(
        str(source),
        str(tmp_path / "output.h5ad"),
        [50.0],
        mz_min=100.0,
        mz_max=500.0,
        create_qc=False,
    )
    with pytest.raises(ValueError, match="strictly inside"):
        run_processing(config)


def test_processing_range_rejects_reversed_limits(tmp_path):
    source = tmp_path / "input.mzML"
    source.touch()
    config = ProcessingConfig(
        str(source),
        str(tmp_path / "output.h5ad"),
        [200.0],
        mz_min=500.0,
        mz_max=100.0,
        create_qc=False,
    )
    with pytest.raises(ValueError, match="mz_min < mz_max"):
        run_processing(config)


def test_processing_range_is_forwarded_to_file_loader(tmp_path, monkeypatch):
    source = tmp_path / "input.mzML"
    source.touch()
    captured = {}

    class Result:
        n_obs = 2
        n_vars = 3

        def save(self, path):
            return Path(path)

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(CyESIData, "load_from_file", fake_load)
    config = ProcessingConfig(
        str(source),
        str(tmp_path / "output.h5ad"),
        [200.0],
        mz_min=50.0,
        mz_max=1500.0,
        create_qc=False,
    )
    run_processing(config)
    assert captured["mz_range"] == (50.0, 1500.0)
