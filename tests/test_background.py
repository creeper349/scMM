from threading import Event
from types import SimpleNamespace

import pytest

from scMM.background import BackgroundJob
from scMM.webapp import _execute, _poll_background


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Terminal:
    current = 0

    def __init__(self):
        self.events = []

    def start(self, title):
        self.events.append(title)

    def update(self, fraction, message):
        self.events.append((fraction, message))

    def render(self):
        pass

    def finish(self, message):
        self.events.append(message)

    def fail(self, error, traceback_text=None):
        self.events.append((error, traceback_text))


def test_page_interruption_keeps_worker_and_receives_result_once():
    class PageSwitch(BaseException):
        pass

    class InterruptedTerminal(Terminal):
        def update(self, fraction, message):
            raise PageSwitch()

    release = Event()
    received = []
    state = State(background_job=None, completed_steps=set())
    ui = SimpleNamespace(session_state=state)

    def calculate(progress):
        progress(.2, "step 1/2")
        if not release.wait(5):
            raise TimeoutError("Test did not release the background worker")
        progress(1, "step 2/2")
        return 42

    try:
        with pytest.raises(PageSwitch):
            _execute(ui, InterruptedTerminal(), "test", calculate, "analysis", on_success=received.append)
        job = state.background_job
        assert job.thread.is_alive()
        assert not received
        release.set()
        assert job.done.wait(5)
        terminal = Terminal()
        _poll_background(ui, terminal)
        _poll_background(ui, terminal)
        assert received == [42]
        assert state.completed_steps == {"analysis"}
        assert state.background_job is None
    finally:
        release.set()


def test_background_error_is_collected_after_page_change():
    def fail(progress):
        raise ValueError("invalid input")

    received, errors = [], []
    job = BackgroundJob(fail, on_success=received.append, completed_step="data", task_index=0).start()
    assert job.done.wait(5)
    state = State(background_job=job, completed_steps=set())
    terminal = Terminal()
    _poll_background(SimpleNamespace(session_state=state, error=errors.append), terminal)
    assert not received and not state.completed_steps
    assert errors and "invalid input" in errors[0]
    assert "ValueError: invalid input" in terminal.events[0][1]
    assert state.background_job is None


def test_rerun_during_completion_retries_without_losing_result():
    class PageSwitch(BaseException):
        pass

    attempts, received = [], []

    def receive(result):
        attempts.append(result)
        if len(attempts) == 1:
            raise PageSwitch()
        received.append(result)

    job = BackgroundJob(lambda progress: 42, on_success=receive, task_index=0).start()
    assert job.done.wait(5)
    state = State(background_job=job, completed_steps=set())
    ui = SimpleNamespace(session_state=state)
    with pytest.raises(PageSwitch):
        _poll_background(ui, Terminal())
    assert state.background_job is job and not job.applied

    class InterruptedFinish(Terminal):
        def finish(self, message):
            raise PageSwitch()

    with pytest.raises(PageSwitch):
        _poll_background(ui, InterruptedFinish())
    assert received == [42]
    _poll_background(ui, Terminal())
    assert received == [42]
    assert state.background_job is None


def test_navigation_monitor_resumes_completed_task():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    release = Event()
    received = []

    def calculate(progress):
        if not release.wait(10):
            raise TimeoutError("Test did not release worker")
        return "complete"

    job = BackgroundJob(calculate, on_success=received.append, task_index=0).start()
    app = AppTest.from_file(Path(__file__).parents[1] / "scMM" / "webapp.py", default_timeout=30)
    app.session_state.background_job = job
    app.session_state.tasks = [dict(step="test", status="running", progress=0, message="running", substeps=[])]
    try:
        app.run()
        assert not app.exception
        assert any(header.value == "任务正在运行" for header in app.header)
        assert job.thread.is_alive()
        release.set()
        assert job.done.wait(5)
        app.run()
        assert not app.exception
        assert received == ["complete"]
        assert any(header.value == "数据导入与 CyESI 峰提取" for header in app.header)
    finally:
        release.set()
