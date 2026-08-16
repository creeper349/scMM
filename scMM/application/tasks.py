"""Detached single-machine processing tasks with persistent state and logs."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .processing import ProcessingPlan, ProcessingPlanner, ProcessingRequest

TaskStatus = Literal["queued", "running", "succeeded", "failed"]
_TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


def utc_now() -> str:
    """Return a stable UTC timestamp for task metadata."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ProcessingTask:
    """Persistent task state shared by the server and worker process."""

    task_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    request_path: str
    state_path: str
    log_path: str
    input_path: str
    result_path: str
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @classmethod
    def from_json(cls, path: str | Path):
        """Load task state written by the manager or worker."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def write(self) -> None:
        """Atomically replace the on-disk state document."""
        _write_json_atomic(Path(self.state_path), asdict(self))


class TaskBusyError(RuntimeError):
    """Raised when the laboratory worker already has an active task."""


class ProcessingTaskManager:
    """Submit and inspect one detached processing task at a time."""

    def __init__(self, planner: ProcessingPlanner, state_root: str | Path) -> None:
        self.planner = planner
        self.state_root = Path(state_root).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.state_root / ".manager.lock"

    def submit(self, request: ProcessingRequest) -> ProcessingTask:
        """Preflight and launch a worker detached from the Panel session."""
        plan = self.planner.preflight(request)
        with self._locked():
            active = self.active()
            if active is not None:
                raise TaskBusyError(f"Task {active.task_id} is still {active.status}")
            task = self._create_task(request, plan)
            gate_path = Path(task.state_path).with_name("start.ready")
            try:
                with Path(task.log_path).open("ab", buffering=0) as log_handle:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "scMM.application.worker",
                            "--request",
                            task.request_path,
                            "--state",
                            task.state_path,
                            "--start-gate",
                            str(gate_path),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        cwd=str(self.state_root),
                        start_new_session=True,
                        close_fds=True,
                    )
            except Exception as exc:
                failed = replace(
                    task,
                    status="failed",
                    updated_at=utc_now(),
                    finished_at=utc_now(),
                    error=f"Could not start worker: {exc}",
                )
                failed.write()
                raise
            running = replace(
                task,
                status="running",
                updated_at=utc_now(),
                started_at=utc_now(),
                pid=process.pid,
            )
            running.write()
            gate_path.touch()
            return running

    def get(self, task_id: str) -> ProcessingTask:
        """Return reconciled state for one task ID."""
        if not task_id or Path(task_id).name != task_id:
            raise ValueError(f"Invalid task ID: {task_id!r}")
        path = self.state_root / task_id / "state.json"
        if not path.is_file():
            raise KeyError(f"Unknown task: {task_id}")
        task = ProcessingTask.from_json(path)
        if (
            task.status not in _TERMINAL_STATUSES
            and task.pid
            and not _worker_is_running(task.pid, task.state_path)
        ):
            task = replace(
                task,
                status="failed",
                updated_at=utc_now(),
                finished_at=utc_now(),
                error=task.error or "Worker exited without recording a terminal state",
            )
            task.write()
        return task

    def list(self) -> tuple[ProcessingTask, ...]:
        """Return known tasks, newest first."""
        tasks = []
        for path in self.state_root.glob("*/state.json"):
            try:
                tasks.append(self.get(path.parent.name))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(tasks, key=lambda task: task.created_at, reverse=True))

    def active(self) -> ProcessingTask | None:
        """Return the active task, if the worker is occupied."""
        return next(
            (task for task in self.list() if task.status not in _TERMINAL_STATUSES),
            None,
        )

    def read_log(self, task_id: str, *, max_bytes: int = 100_000) -> str:
        """Read the tail of a task log without loading an unbounded file."""
        task = self.get(task_id)
        path = Path(task.log_path)
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    def _create_task(self, request: ProcessingRequest, plan: ProcessingPlan) -> ProcessingTask:
        task_id = uuid.uuid4().hex
        task_dir = self.state_root / task_id
        task_dir.mkdir()
        request_path = task_dir / "request.json"
        state_path = task_dir / "state.json"
        log_path = task_dir / "worker.log"
        payload = {
            "task_id": task_id,
            "input_path": str(plan.input_path),
            "output_root": str(plan.output_root),
            "result_path": str(plan.result_path),
            "result_name": plan.result_path.name,
            "overwrite": request.overwrite,
            "parameters": asdict(request.parameters),
            "warnings": list(plan.warnings),
        }
        _write_json_atomic(request_path, payload)
        now = utc_now()
        task = ProcessingTask(
            task_id=task_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request_path=str(request_path),
            state_path=str(state_path),
            log_path=str(log_path),
            input_path=str(plan.input_path),
            result_path=str(plan.result_path),
        )
        task.write()
        return task

    @contextmanager
    def _locked(self):
        with self._lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_request(path: str | Path) -> dict[str, Any]:
    """Read one worker request document."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def update_task(path: str | Path, **updates: Any) -> ProcessingTask:
    """Atomically update state from the worker process."""
    task = ProcessingTask.from_json(path)
    updated = replace(task, updated_at=utc_now(), **updates)
    updated.write()
    return updated


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _worker_is_running(pid: int, state_path: str) -> bool:
    command_path = Path("/proc") / str(pid) / "cmdline"
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    except PermissionError:
        command = ""
    if command:
        return "scMM.application.worker" in command and state_path in command
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "ProcessingTask",
    "ProcessingTaskManager",
    "TaskBusyError",
    "TaskStatus",
    "load_request",
    "update_task",
    "utc_now",
]
