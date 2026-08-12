from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("panel")

from scMM.application import OutputRoot, StorageCatalog, StorageRoot
from scMM.ui.processing import GuidedProcessingPanel


def _panel(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "results"
    raw.mkdir()
    output.mkdir()
    source = raw / "sample.mzML"
    source.write_bytes(b"raw")
    panel = GuidedProcessingPanel(
        StorageCatalog((StorageRoot("Raw", raw),)),
        (OutputRoot("Results", output),),
    )
    panel.set_input("Raw", "sample.mzML")
    return panel


def test_processing_panel_requires_fresh_preflight_and_confirmation(tmp_path: Path) -> None:
    panel = _panel(tmp_path)

    panel._preflight()

    assert "预检通过" in panel.preflight_text.object
    assert panel.confirm.disabled is False
    assert panel.submit_button.disabled is True

    panel.confirm.value = True
    assert panel.submit_button.disabled is False

    panel.ppm_tol.value = 8.0
    assert panel.confirm.disabled is True
    assert panel.submit_button.disabled is True
    assert "重新" in panel.preflight_text.object


def test_processing_panel_reports_validation_failure(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    panel.ref_mz.value = 0

    panel._preflight()

    assert "预检未通过" in panel.preflight_text.object
    assert "ref_mz" in panel.preflight_text.object


def test_processing_panel_submits_and_recovers_task_status(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    panel._preflight()
    panel.confirm.value = True
    task = Mock(
        task_id="a" * 32,
        status="running",
        created_at="2026-08-12T00:00:00+00:00",
        input_path="/raw/sample.mzML",
        result_path="/results/sample",
        error=None,
    )

    with (
        patch.object(panel.tasks, "submit", return_value=task) as submit,
        patch.object(panel.tasks, "list", return_value=(task,)),
        patch.object(panel.tasks, "get", return_value=task),
        patch.object(panel.tasks, "read_log", return_value="processing"),
    ):
        panel._submit()

    submit.assert_called_once()
    assert panel._active_task_id == task.task_id
    assert "处理中" in panel.status_text.object
    assert panel.log_text.value == "processing"


def test_processing_panel_uses_compact_controls_and_responsive_groups(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    layout = panel.panel()

    assert panel.output_select.width == 180
    assert panel.ref_mz.width == 150
    assert panel.log_text.sizing_mode == "stretch_width"
    assert any(type(item).__name__ == "FlexBox" for item in layout)
