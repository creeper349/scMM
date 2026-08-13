from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
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
    source.write_bytes(b"<mzML><spectrum></spectrum></mzML>")
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


def test_processing_panel_loads_quality_and_safe_result_downloads(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    result = panel.outputs.roots[0].path / "sample"
    result.mkdir()
    (result / ".meta").write_text("{}", encoding="utf-8")
    for filename in ("data.csv", "cell-quality.csv", "scmm-manifest.json"):
        (result / filename).write_text("content", encoding="utf-8")
    report = SimpleNamespace(
        summary=SimpleNamespace(
            cell_count=3,
            feature_count=2,
            zero_fraction=0.25,
            median_total_intensity=10.0,
            median_detected_features=2.0,
            embedding_warnings=(),
        ),
        cells=pd.DataFrame(
            {
                "cell_index": [0, 1, 2],
                "total_intensity": [5.0, 10.0, 15.0],
                "detected_features": [1, 2, 2],
            }
        ),
        features=pd.DataFrame({"mz": [100.0, 200.0], "detection_rate": [1.0, 0.5]}),
        embedding=pd.DataFrame(
            {
                "cell_index": [0, 1, 2],
                "PCA1": [-1.0, 0.0, 1.0],
                "PCA2": [0.5, -1.0, 0.5],
            }
        ),
    )
    task = Mock(task_id="b" * 32, result_path=str(result))

    with patch("scMM.ui.processing.load_quality_report", return_value=report):
        panel._load_quality(task)

    assert panel.quality_section.visible is True
    assert "细胞事件" in panel.quality_summary.object
    assert panel.artifact_downloads["data.csv"].disabled is False
    assert panel.artifact_downloads["feature_meta.csv"].disabled is True


def test_processing_panel_rejects_result_outside_output_root(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".meta").write_text("{}", encoding="utf-8")

    with pytest.raises(PermissionError, match="输出根目录"):
        panel._safe_result_path(str(outside))
