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
    workspace._selector.value = [str(raw_path)]

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
