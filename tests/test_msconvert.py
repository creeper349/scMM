from pathlib import Path
from types import SimpleNamespace

import pytest

from scMM.file.msconvert import MSConvertError, convert_raw, find_msconvert


def test_find_msconvert_accepts_explicit_file(tmp_path):
    executable = tmp_path / "msconvert.exe"
    executable.touch()
    assert find_msconvert(executable) == executable.resolve()


def test_convert_raw_builds_command_and_finds_output(tmp_path, monkeypatch):
    executable = tmp_path / "msconvert.exe"
    executable.touch()
    source = tmp_path / "sample.raw"
    source.touch()
    destination = tmp_path / "converted"
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        destination.mkdir(exist_ok=True)
        (destination / "sample.mzML").touch()
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("scMM.file.msconvert.subprocess.run", fake_run)
    result = convert_raw(source, destination, executable=executable)
    assert result == destination / "sample.mzML"
    assert "--mzML" in observed["command"]
    assert "--outdir" in observed["command"]


def test_find_msconvert_has_actionable_error(monkeypatch):
    monkeypatch.delenv("SCMM_MSCONVERT", raising=False)
    monkeypatch.setattr("scMM.file.msconvert.shutil.which", lambda _: None)
    monkeypatch.setattr("scMM.file.msconvert.os.environ", {})
    with pytest.raises(MSConvertError, match="ProteoWizard"):
        find_msconvert()
