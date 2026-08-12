import numpy as np
import pyopenms as oms
import pytest

pytest.importorskip("panel")
pytest.importorskip("plotly")

from scMM.application import StorageRoot
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


def test_ui_workspace_loads_selected_file_and_populates_downloads(tmp_path) -> None:
    raw_path = tmp_path / "preview.mzML"
    _write_raw_file(raw_path)
    workspace = PreviewWorkspace((StorageRoot("Raw", tmp_path),))
    workspace.file_select.value = raw_path.name

    workspace._load_selected(None)

    assert workspace.preview is not None
    assert workspace.tabs.active == 1
    assert workspace.tic["intensity"].tolist() == [10.0, 20.0]
    assert workspace.eic_download.disabled is False
    assert workspace.spectrum_download.filename == "preview_spectrum.csv"
    assert "TIC" in workspace.tic_pane.object.layout.title.text


def test_create_app_returns_template_with_isolated_session(tmp_path) -> None:
    app = create_app((StorageRoot("Raw", tmp_path),))

    assert type(app).__name__ == "FastListTemplate"
    assert app.title == "scMM 数据查看"


def test_sidebar_uses_compact_responsive_controls(tmp_path) -> None:
    workspace = PreviewWorkspace((StorageRoot("Raw", tmp_path),))
    sidebar = workspace.sidebar()

    compact = (
        workspace.ms_level,
        workspace.mz_min,
        workspace.mz_max,
        workspace.target_mz,
        workspace.ppm,
    )
    assert all(widget.width == 96 for widget in compact)
    assert all(widget.height == 54 for widget in compact)
    assert all(widget.sizing_mode == "fixed" for widget in compact)
    assert workspace.root_select.max_width == 180
    assert workspace.root_select.width is None
    assert workspace.file_select.width == 180
    assert workspace.file_select.sizing_mode is None
    assert workspace.rt_range.width == 180
    assert workspace.rt_range.sizing_mode is None
    assert type(sidebar[7]).__name__ == "FlexBox"
    assert type(sidebar[10]).__name__ == "FlexBox"


def test_compact_browser_navigates_directories(tmp_path) -> None:
    nested = tmp_path / "batch"
    nested.mkdir()
    raw_path = nested / "nested.mzML"
    _write_raw_file(raw_path)
    workspace = PreviewWorkspace((StorageRoot("Raw", tmp_path),))

    workspace.file_select.value = "batch"

    assert workspace._browser_directory == nested.relative_to(tmp_path)
    assert workspace.file_select.options["📄 nested.mzML"] == "batch/nested.mzML"
    assert workspace.up_button.disabled is False

    workspace.file_select.value = "batch/nested.mzML"

    assert workspace._selected_path == "batch/nested.mzML"
    assert workspace.load_button.disabled is False
