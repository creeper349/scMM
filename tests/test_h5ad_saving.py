import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from scMM.file.data import CyESIData
from scMM.webapp import _atomic_save_h5ad


def example_data():
    return CyESIData(X=np.ones((3, 2), dtype="float32"),
                     obs=pd.DataFrame(index=["a", "b", "c"]),
                     var=pd.DataFrame({"mz": [100., 200.]}, index=["f1", "f2"]))


def test_atomic_h5ad_save_preserves_original_on_failure(tmp_path, monkeypatch):
    data = example_data()
    original = data.save(tmp_path / "original.h5ad")
    contents = original.read_bytes()

    def fail(self, path):
        path.write_bytes(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(CyESIData, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        _atomic_save_h5ad(data, original)
    assert original.read_bytes() == contents
    assert list(tmp_path.iterdir()) == [original]


def test_loaded_h5ad_can_save_obs_var_layers_back_to_original(tmp_path):
    source = example_data().save(tmp_path / "source.h5ad")
    app = AppTest.from_string('''
import streamlit as st
from scMM.webapp import _state, page_data, _h5ad_save_controls
_state(st)
page_data()
if st.session_state.data is not None:
    _h5ad_save_controls(st, st.session_state.data)
''', default_timeout=30).run()
    next(widget for widget in app.radio if widget.label == "输入模式").set_value("加载已有 H5AD")
    app.text_input(key="input_path").set_value(str(source))
    next(button for button in app.button if button.label == "加载 H5AD").click().run()
    assert not app.exception
    assert app.session_state.h5ad_path == str(source.resolve())
    data = app.session_state.data
    data.obs["group"] = ["control", "treated", "treated"]
    data.var["name"] = ["A", "B"]
    data.normalize("total", scale=1.)
    app.button(key="results_save_h5ad").click().run()
    assert not app.exception
    assert not app.error
    restored = CyESIData.read(source)
    assert restored.obs.group.tolist() == ["control", "treated", "treated"]
    assert restored.var.name.tolist() == ["A", "B"]
    assert "normalized_total" in restored.layers
    np.testing.assert_allclose(restored.X.sum(axis=1), 1.)


def test_processing_output_folder_dialog_preserves_filename(tmp_path, monkeypatch):
    output = tmp_path / "selected"
    output.mkdir()
    monkeypatch.setattr("scMM.file.dialogs.choose_local_path", lambda kind: str(output))
    app = AppTest.from_string('''
import streamlit as st
from scMM.webapp import _state, page_data
_state(st)
page_data()
''', default_timeout=30).run()
    next(widget for widget in app.radio if widget.label == "输入模式").set_value("处理单个 mzML / mzXML / RAW 文件").run()
    app.text_input(key="processing_output_path").set_value("previous/my_result.h5ad").run()
    next(button for button in app.button if button.label == "📁 选择 H5AD 输出文件夹").click().run()
    assert not app.exception
    assert app.text_input(key="processing_output_path").value == str(output / "my_result.h5ad")
    configurations = []

    def process(config, *, progress_callback):
        configurations.append(config)
        return example_data(), Path(config.output_path), None

    monkeypatch.setattr("scMM.workflows.run_processing", process)
    app.text_input(key="input_path").set_value(str(tmp_path / "input.mzML"))
    next(button for button in app.button if button.label == "开始峰提取").click().run()
    assert not app.exception
    assert configurations[0].output_path == str(output / "my_result.h5ad")
    assert app.session_state.h5ad_path == str(output / "my_result.h5ad")
