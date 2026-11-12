import pyopenms as oms

from scMM.file.io import load_single_file


def test_raw_dispatches_to_thermo_reader_without_conversion(tmp_path, monkeypatch):
    source = tmp_path / "sample.RAW"
    source.touch()
    experiment = oms.MSExperiment()
    observed = {}

    def fake_reader(path):
        observed["path"] = path
        return experiment, {
            "timestamp": 1.0,
            "instrument": "Orbitrap",
            "raw_reader": "Thermo RawFileReader",
        }

    monkeypatch.setattr("scMM.file.io.load_thermo_raw", fake_reader)
    loaded, metadata = load_single_file(
        str(source), format="auto", msconvert_path="unused.exe"
    )

    assert loaded is experiment
    assert observed["path"] == str(source)
    assert metadata["source_file"] == "sample.RAW"
    assert metadata["converted_from_raw"] is False
    assert metadata["raw_reader"] == "Thermo RawFileReader"
