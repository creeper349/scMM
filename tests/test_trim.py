from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
import pytest

from scMM.file.trim import selected_frames
from scMM.file.io import align_frame, sum_spec


def write_source(path):
    experiment = oms.MSExperiment()
    for index, level in enumerate([1, 2, 1, 3, 1]):
        spectrum = oms.MSSpectrum()
        spectrum.setRT(index * 30.)
        spectrum.setMSLevel(level)
        spectrum.setNativeID(f"scan={index + 1}")
        spectrum.set_peaks((np.array([199.99, 200., 200.01]), np.array([0., index + 1., 0.])))
        experiment.addSpectrum(spectrum)
    oms.MzMLFile().store(str(path), experiment)
    return experiment


def test_selection_filters_time_and_frames_without_renumbering(tmp_path):
    exp = write_source(tmp_path / "source.mzML")
    assert [i for i, _ in selected_frames(exp, rt_range=(30, 120), frame_range=(1, 4))] == [2, 4]
    assert [i for i, _ in selected_frames(exp, frame_range=(0, 2))] == [0, 2]
    summed = sum_spec(exp, mz_range=(199.9, 200.1), rt_range=(30, 120), frame_range=(1, 4))
    assert summed.getMetaValue("n_summed_spectra") == 2
    full = sum_spec(exp, mz_range=(199.9, 200.1))
    np.testing.assert_allclose(summed.get_peaks()[1], full.get_peaks()[1] * (8 / 9), rtol=1e-6)
    aligned, metadata = align_frame(exp, [200.], rt_range=(30, 120), frame_range=(1, 4))
    assert aligned.index.tolist() == [2, 4]
    assert metadata.native_id.tolist() == ["scan=3", "scan=5"]
    np.testing.assert_allclose(aligned.iloc[:, 0], [3., 5.])
    assert exp.size() == 5
    with pytest.raises(ValueError):
        list(selected_frames(exp, frame_range=(4, 2)))
    with pytest.raises(ValueError, match="No MS1"):
        sum_spec(exp, rt_range=(300, 400))


@pytest.mark.parametrize("strategy", ["single", "shared", "independent"])
def test_frame_selection_applies_to_summing_and_cell_extraction(tmp_path, monkeypatch, strategy):
    import scMM.file.data as module
    from scMM.file.data import CyESIData
    source = tmp_path / "source.mzML"
    write_source(source)
    monkeypatch.setattr(module, "filter_spectrum", lambda spec, **kwargs: spec)
    monkeypatch.setattr(module, "extract_peaks", lambda *args, **kwargs: (np.array([200.]), np.array([8.])))
    observed = []
    def cells(cls, frame, obs, *, source_file, **kwargs):
        observed.append(frame.index.tolist())
        assert obs.frame_id.tolist() == [2, 4]
        return CyESIData(X=frame.to_numpy(), obs=obs,
                         var=pd.DataFrame({"mz": [200.]}, index=["200.0"]),
                         layers={"feature_snr": frame.to_numpy()}, uns={"processing": {}})
    monkeypatch.setattr(CyESIData, "_from_aligned_frames", classmethod(cells))
    kwargs = dict(reference_mz=[200.], mz_range=(199.9, 200.1), rt_range=(30, 120),
                  frame_range=(1, 4), log_memory=False, show_progress=False)
    if strategy == "single":
        result = CyESIData.load_from_file(source, **kwargs)
    else:
        result = CyESIData.load_from_directory([source], feature_strategy=strategy, **kwargs)
    assert observed == [[2, 4]]
    assert result.n_obs == 2
    spectrum = next(iter(result.uns["summed_spectra"].values()))
    expected = sum_spec(write_source(source), mz_range=(199.9, 200.1), rt_range=(30, 120), frame_range=(1, 4))
    np.testing.assert_allclose(spectrum["intensity"], expected.get_peaks()[1])
    assert [p.name for p in tmp_path.iterdir()] == ["source.mzML"]


def test_import_panel_passes_frame_selection_to_processing(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from scMM.file.data import CyESIData
    source = tmp_path / "source.mzML"
    write_source(source)
    configs = []
    def process(config, **kwargs):
        configs.append(config)
        return CyESIData(X=np.ones((1, 1))), Path(config.output_path), None
    monkeypatch.setattr("scMM.workflows.run_processing", process)
    app = AppTest.from_string('''
import streamlit as st
from scMM.webapp import _state, page_data
_state(st)
page_data()
''', default_timeout=30).run()
    assert not app.exception
    panel = next(p for p in app.expander if p.label.startswith("文件裁剪"))
    assert panel.proto.expanded is False
    radio = next(w for w in app.radio if w.label == "输入模式")
    radio.set_value(radio.options[1]).run()
    app.text_input(key="input_path").set_value(str(source)).run()
    app.button(key="trim_load").click().run()
    next(w for w in app.radio if w.label == "裁剪方式").set_value("时间与帧序号").run()
    next(w for w in app.slider if w.label.startswith("保留时间")).set_value((.5, 2.))
    next(w for w in app.slider if w.label.startswith("保留原始帧")).set_value((2, 4))
    next(b for b in app.button if b.label == "开始峰提取").click().run()
    assert not app.exception and not app.error
    assert configs[0].file_ranges[str(source)]["frame_range"] == [2, 4]
    assert configs[0].file_ranges[str(source)]["rt_range"] == [30., 120.]
    assert not any("裁剪输出" in w.label for w in app.text_input)


@pytest.mark.parametrize("strategy", ["shared", "independent"])
def test_each_file_uses_its_own_frame_interval(tmp_path, monkeypatch, strategy):
    import scMM.file.data as module
    from scMM.file.data import CyESIData
    paths = [tmp_path / "a.mzML", tmp_path / "b.mzML"]
    for path in paths:
        write_source(path)
    monkeypatch.setattr(module, "filter_spectrum", lambda spec, **kwargs: spec)
    monkeypatch.setattr(module, "extract_peaks", lambda *args, **kwargs: (np.array([200.]), np.array([8.])))
    observed = {}
    def cells(cls, frame, obs, *, source_file, **kwargs):
        observed[Path(source_file).name] = obs.frame_id.tolist()
        return CyESIData(X=frame.to_numpy(), obs=obs,
                         var=pd.DataFrame({"mz": [200.]}, index=["200.0"]),
                         layers={"feature_snr": frame.to_numpy()}, uns={"processing": {}})
    monkeypatch.setattr(CyESIData, "_from_aligned_frames", classmethod(cells))
    result = CyESIData.load_from_directory(paths, reference_mz=[200.], feature_strategy=strategy,
                mz_range=(199.9, 200.1), log_memory=False, show_progress=False,
                file_ranges={str(paths[0]): {"frame_range": [0, 2]}, str(paths[1]): {"rt_range": [90., 120.]}})
    assert observed == {"a.mzML": [0, 2], "b.mzML": [4]}
    assert result.n_obs == 3
