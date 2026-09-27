from __future__ import annotations

import threading
import time

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.tasks.base import Task, TaskContext, TaskResult
from rokbot.vision.screen_state import ScreenStateDetector


class BotController:
    """Single-owner executor for game tasks.

    Tasks are intentionally serialized: one task owns the emulator at a time,
    which prevents simultaneous ADB taps from different workers.
    """

    def __init__(
        self,
        adb: ADBClient | None = None,
        screen: Screen | None = None,
        state_detector: ScreenStateDetector | None = None,
    ) -> None:
        self.adb = adb or ADBClient()
        self.screen = screen or Screen(self.adb)
        self.state_detector = state_detector or ScreenStateDetector()
        self._lock = threading.Lock()
        self._running = False
        self._last_result: TaskResult | None = None
        self._last_started = 0.0
        self._last_finished = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_result(self) -> TaskResult | None:
        return self._last_result

    def run(self, task: Task) -> TaskResult:
        if not self._lock.acquire(blocking=False):
            return TaskResult(False, getattr(task, "name", task.__class__.__name__), "Another task is already running")

        self._running = True
        self._last_started = time.time()
        name = getattr(task, "name", task.__class__.__name__)
        try:
            self.adb.select_first_device()
            ctx = TaskContext(self.adb, self.screen, self.state_detector)
            result = task.run(ctx)
            self._last_result = result
            return result
        except Exception as exc:
            result = TaskResult(False, name, f"Task failed: {exc}")
            self._last_result = result
            return result
        finally:
            self._last_finished = time.time()
            self._running = False
            self._lock.release()

    def status(self) -> dict:
        result = self._last_result
        return {
            "running": self._running,
            "last_started": self._last_started or None,
            "last_finished": self._last_finished or None,
            "last_result": None if result is None else {
                "ok": result.ok,
                "task": result.task,
                "message": result.message,
                "state": result.state,
                "data": result.data,
            },
        }
