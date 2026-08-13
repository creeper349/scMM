import numpy as np
import pyopenms as oms
import pytest

pytest.importorskip("panel")
pytest.importorskip("plotly")

from scMM.application import OutputRoot, StorageRoot
from scMM.ui.app import PreviewWorkspace, create_app


def _write_raw_file(path) -> None:
    experiment = oms.MSExperiment()
    for rt, intensity in [(1.0, 10.0), (2.0, 20.0)]:
        spectrum = oms.MSSpectrum()
        spectrum.setMSLevel(1)
        spectrum.setRT(rt)
        spectrum.set_peaks(
            (
                np.asarray([99.9, 100.0, 100.1]),
                np.asarray([0.0, intensity, 0.0], dtype=np.float32),
            )
        )
        experiment.addSpectrum(spectrum)
    oms.MzMLFile().store(str(path), experiment)


def _workspace(tmp_path) -> PreviewWorkspace:
    output = tmp_path / "results"
    output.mkdir(exist_ok=True)
    return PreviewWorkspace(
        (StorageRoot("Raw", tmp_path),),
        (OutputRoot("Results", output),),
    )


def test_ui_workspace_loads_selected_file_and_populates_downloads(tmp_path) -> None:
    raw_path = tmp_path / "preview.mzML"
    _write_raw_file(raw_path)
    workspace = _workspace(tmp_path)
    assert workspace._selector is not None
    workspace._selector.value = [str(raw_path)]

    workspace._load_selected(None)

    assert workspace.preview is not None
    assert workspace.tabs.active == 1
    assert workspace.tic["intensity"].tolist() == [10.0, 20.0]
    assert workspace.eic_download.disabled is False
    assert workspace.spectrum_download.filename == "preview_spectrum.csv"
    assert "TIC" in workspace.tic_pane.object.layout.title.text


def test_create_app_returns_template_with_isolated_session(tmp_path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    app = create_app(
        (StorageRoot("Raw", tmp_path),),
        (OutputRoot("Results", output),),
    )

    assert type(app).__name__ == "FastListTemplate"
    assert app.title == "scMM 数据查看"
    assert app.sidebar_width == 680
    assert 'id = "scmm-sidebar-resizer"' in app.sidebar_footer


def test_sidebar_restores_default_controls_and_panel_file_selector(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    sidebar = workspace.sidebar()

    default_width_controls = (
        workspace.root_select,
        workspace.ms_level,
        workspace.mz_min,
        workspace.mz_max,
        workspace.target_mz,
        workspace.ppm,
        workspace.rt_range,
    )
    assert all(widget.width == 300 for widget in default_width_controls[:-1])
    assert workspace.rt_range.width is None
    assert all(widget.sizing_mode == "stretch_width" for widget in default_width_controls)
    assert workspace._selector is not None
    assert type(workspace._selector).__name__ == "FileSelector"
    assert workspace._selector.root_directory == str(tmp_path)
    assert workspace._selector.only_files is True
    assert workspace._selector.sizing_mode == "stretch_width"
    assert all(type(item).__name__ != "FlexBox" for item in sidebar)


def test_panel_file_selector_accepts_a_nested_file(tmp_path) -> None:
    nested = tmp_path / "batch"
    nested.mkdir()
    raw_path = nested / "nested.mzML"
    _write_raw_file(raw_path)
    workspace = _workspace(tmp_path)
    assert workspace._selector is not None

    workspace._selector.value = [str(raw_path)]

    assert workspace._selector.value == [str(raw_path)]
    assert workspace.load_button.disabled is False
    assert "nested.mzML" in workspace.selection_text.object
