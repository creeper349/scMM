from types import SimpleNamespace

import numpy as np
import pytest

from scMM.file.acquisition import read_acquisition_settings, resolve_ms_files


def write_settings(path, lower=100, upper=1500, resolution=50000):
    path.write_text(f'''<mzML xmlns="urn:mzml"><run><spectrumList>
    <spectrum><cvParam accession="MS:1000511" value="2"/>
    <scanList><scan><cvParam accession="MS:1000800" value="1000"/>
    <scanWindow><cvParam accession="MS:1000501" value="1"/>
    <cvParam accession="MS:1000500" value="2000"/></scanWindow></scan></scanList></spectrum>
    <spectrum><cvParam accession="MS:1000511" value="1"/>
    <scanList><scan><cvParam accession="MS:1000800" value="{resolution}"/>
    <scanWindow><cvParam accession="MS:1000501" value="{lower}"/>
    <cvParam accession="MS:1000500" value="{upper}"/></scanWindow></scan></scanList></spectrum>
    </spectrumList></run></mzML>''', encoding="utf8")


def test_metadata_uses_ms1_and_multiple_inputs(tmp_path):
    first, second = tmp_path / "a.mzML", tmp_path / "b.mzML"
    write_settings(first)
    write_settings(second, lower=50, upper=1200, resolution=35000)
    assert resolve_ms_files([tmp_path, first]) == [first, second]
    result = read_acquisition_settings([first, second])
    assert (result["mz_min"], result["mz_max"], result["resolution"]) == (50, 1500, 35000)
    assert len(result["files"]) == 2


def test_missing_metadata_and_mzxml(tmp_path):
    source = tmp_path / "empty.mzML"
    source.write_text('<mzML><spectrum><cvParam accession="MS:1000511" value="1"/></spectrum></mzML>')
    assert set(read_acquisition_settings([source])) == {"files"}
    source = tmp_path / "scan.mzXML"
    source.write_text('<mzXML><scan msLevel="1" lowMz="80" highMz="900" resolution="60000"/></mzXML>')
    result = read_acquisition_settings([source])
    assert (result["mz_min"], result["mz_max"], result["resolution"]) == (80, 900, 60000)


def test_raw_acquisition_reads_trailer_and_disposes(monkeypatch, tmp_path):
    from scMM.file.thermo_raw import read_raw_acquisition_settings
    disposed = []
    raw = SimpleNamespace(
        RunHeaderEx=SimpleNamespace(FirstSpectrum=1, LastSpectrum=3),
        SelectInstrument=lambda *args: None,
        GetFilterForScanNumber=lambda i: SimpleNamespace(MSOrder=2 if i == 1 else 1),
        GetScanStatsForScanNumber=lambda i: SimpleNamespace(LowMass=100, HighMass=1200),
        GetTrailerExtraInformation=lambda i: SimpleNamespace(Labels=["Orbitrap Resolution:"], Values=["70,000"]),
        Dispose=lambda: disposed.append(True))
    monkeypatch.setattr("scMM.file.thermo_raw._load_api", lambda: (SimpleNamespace(FileFactory=lambda p: raw), SimpleNamespace(MS=1)))
    assert read_raw_acquisition_settings(tmp_path / "a.raw") == ([(100., 1200.)] * 2, [70000.] * 2, 2)
    assert disposed == [True]


@pytest.mark.parametrize("strategy", ["shared", "independent"])
def test_batch_file_selection_reaches_both_extraction_paths(tmp_path, monkeypatch, strategy):
    import pandas as pd
    import pyopenms as oms
    import scMM.file.data as module
    from scMM.file.data import CyESIData

    paths = [tmp_path / "a.mzML", tmp_path / "b.raw"]
    for path in paths:
        path.touch()
    observed = []
    def cells(cls, *args, source_file, **kwargs):
        source = source_file.name
        return CyESIData(X=np.array([[1., 2.]], dtype=np.float32),
                         obs=pd.DataFrame({"source_file": [source]}, index=[source]),
                         var=pd.DataFrame({"mz": [100., 200.]}, index=["100.0", "200.0"]),
                         layers={"feature_snr": np.array([[1., 2.]], dtype=np.float32)})
    def load(path, **kwargs):
        observed.append(str(path))
        return oms.MSExperiment(), {"path": str(path)}
    def sum_spec(*args, **kwargs):
        spectrum = oms.MSSpectrum()
        spectrum.set_peaks((np.array([99., 100., 101.]), np.array([0., 1., 0.])))
        return spectrum
    monkeypatch.setattr(module, "load_single_file", load)
    monkeypatch.setattr(module, "sum_spec", sum_spec)
    monkeypatch.setattr(module, "filter_spectrum", lambda spec, **kwargs: spec)
    monkeypatch.setattr(module, "extract_peaks", lambda *args, **kwargs: (np.array([100., 200.]), np.array([1., 2.])))
    monkeypatch.setattr(module, "align_frame", lambda *args, **kwargs: (pd.DataFrame([[1., 2.]], columns=[100., 200.]), pd.DataFrame(index=[0])))
    monkeypatch.setattr(CyESIData, "_from_aligned_frames", classmethod(cells))
    if strategy == "independent":
        monkeypatch.setattr(CyESIData, "load_from_file", classmethod(lambda cls, path, **kwargs: cells(cls, source_file=path)))
    result = CyESIData.load_from_directory(paths, reference_mz=[100.], feature_strategy=strategy, show_progress=False)
    assert result.shape == (2, 2)
    assert set(result.obs.source_file) == {"a.mzML", "b.raw"}
    assert result.uns["file_meta"]["name"] == "selected_files"
