"""Session-owned work that continues across Streamlit page reruns."""

from queue import Queue
from threading import Event, Thread
import traceback


class BackgroundJob:
    def __init__(self, function, *, on_success=None, completed_step=None, task_index=None, title="任务"):
        self.events = Queue()
        self.done = Event()
        self.result = None
        self.error = None
        self.traceback = ""
        self.on_success = on_success
        self.completed_step = completed_step
        self.task_index = task_index
        self.title = title
        self.applied = False
        self.result_applied = False
        self.thread = Thread(target=self._run, args=(function,), daemon=True, name="scmm-analysis")

    def start(self):
        self.thread.start()
        return self

    def _run(self, function):
        try:
            self.result = function(lambda fraction, message: self.events.put((float(fraction), str(message))))
        except BaseException as exc:
            self.error = exc
            self.traceback = traceback.format_exc()
        finally:
            self.done.set()
