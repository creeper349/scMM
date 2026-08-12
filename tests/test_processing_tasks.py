import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from scMM.application import (
    OutputCatalog,
    OutputRoot,
    ProcessingParameters,
    ProcessingPlanner,
    ProcessingRequest,
    ProcessingTaskManager,
    StorageCatalog,
    StorageRoot,
    TaskBusyError,
)
from scMM.application.tasks import ProcessingTask, update_task
from scMM.application.worker import run_request


def _manager(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "results"
    state = tmp_path / "tasks"
    raw.mkdir()
    output.mkdir()
    source = raw / "sample.mzML"
    source.write_bytes(b"raw")
    planner = ProcessingPlanner(
        StorageCatalog((StorageRoot("Raw", raw),)),
        OutputCatalog((OutputRoot("Results", output),)),
    )
    manager = ProcessingTaskManager(planner, state)
    request = ProcessingRequest(
        storage_label="Raw",
        input_path="sample.mzML",
        output_label="Results",
        parameters=ProcessingParameters(ref_mz=100, n_jobs=1),
    )
    return manager, request, output


def test_task_manager_launches_detached_worker_and_persists_state(tmp_path: Path) -> None:
    manager, request, _ = _manager(tmp_path)
    process = Mock(pid=4321)

    with patch("scMM.application.tasks.subprocess.Popen", return_value=process) as popen:
        task = manager.submit(request)

    assert task.status == "running"
    assert task.pid == 4321
    assert ProcessingTask.from_json(task.state_path) == task
    command = popen.call_args.args[0]
    assert command[1:3] == ["-m", "scMM.application.worker"]
    assert "--start-gate" in command
    assert popen.call_args.kwargs["start_new_session"] is True
    assert json.loads(Path(task.request_path).read_text())["parameters"]["ref_mz"] == 100


def test_task_manager_blocks_a_second_active_task(tmp_path: Path) -> None:
    manager, request, _ = _manager(tmp_path)
    with (
        patch("scMM.application.tasks.subprocess.Popen", return_value=Mock(pid=4321)),
        patch("scMM.application.tasks._worker_is_running", return_value=True),
    ):
        manager.submit(request)
        with pytest.raises(TaskBusyError, match="still running"):
            manager.submit(request)


def test_task_manager_reads_bounded_log_and_reconciles_dead_worker(tmp_path: Path) -> None:
    manager, request, _ = _manager(tmp_path)
    with patch("scMM.application.tasks.subprocess.Popen", return_value=Mock(pid=4321)):
        task = manager.submit(request)
    Path(task.log_path).write_text("0123456789", encoding="utf-8")

    with patch("scMM.application.tasks._worker_is_running", return_value=False):
        reconciled = manager.get(task.task_id)

    assert reconciled.status == "failed"
    assert "without recording" in reconciled.error
    assert manager.read_log(task.task_id, max_bytes=4) == "6789"


def test_worker_saves_result_and_reproducibility_manifest(tmp_path: Path) -> None:
    manager, request, output = _manager(tmp_path)
    plan = manager.planner.preflight(request)
    task = manager._create_task(request, plan)

    class FakeDataset:
        def __init__(self):
            self.data = pd.DataFrame([[1.0]], columns=[100.0])
            self.peak_meta = pd.DataFrame(index=[0])
            self.file_meta = {"name": "sample", "ref_mz": 100.0}

        def get_name(self):
            return self.file_meta["name"]

        def save(self, root, *, overwrite=False):
            result = Path(root) / self.file_meta["name"]
            result.mkdir(exist_ok=overwrite)
            return result

    with patch(
        "scMM.application.worker.CyESIData.load_from_file", return_value=FakeDataset()
    ) as loader:
        result = run_request(task.request_path, task.state_path)

    assert result == output / "sample"
    loader.assert_called_once()
    manifest = json.loads((result / "scmm-manifest.json").read_text())
    assert manifest["task_id"] == task.task_id
    assert manifest["parameters"]["ref_mz"] == 100
    assert (result / "quality-summary.json").is_file()
    assert ProcessingTask.from_json(task.state_path).status == "succeeded"


def test_update_task_preserves_unmodified_fields(tmp_path: Path) -> None:
    manager, request, _ = _manager(tmp_path)
    task = manager._create_task(request, manager.planner.preflight(request))

    updated = update_task(task.state_path, status="failed", error="boom")

    assert updated.task_id == task.task_id
    assert updated.error == "boom"
