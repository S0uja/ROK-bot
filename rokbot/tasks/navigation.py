from __future__ import annotations

import time

from rokbot.tasks.base import Task, TaskContext, TaskResult
from rokbot.vision.screen_state import ScreenState
from rokbot.vision.controls import ROKControls


class EnsureCityTask:
    name = "ensure_city"

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self.controls = ROKControls()

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()
        if state.state == ScreenState.CITY:
            return TaskResult(True, self.name, "Already in city", state.state.value, {"confidence": state.confidence})

        if state.state != ScreenState.MAP:
            return TaskResult(False, self.name, "Cannot navigate: screen is not CITY or MAP", state.state.value, {"confidence": state.confidence})

        # Android BACK returns from the world map to the city in the current UI.
        ctx.adb.shell("input", "keyevent", "KEYCODE_BACK")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            state = ctx.refresh()
            if state.state == ScreenState.CITY:
                return TaskResult(True, self.name, "Returned to city", state.state.value, {"confidence": state.confidence})
            time.sleep(0.25)

        return TaskResult(False, self.name, "City did not appear after BACK", state.state.value, {"confidence": state.confidence})


class OpenMapTask:
    name = "open_map"

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self.controls = ROKControls()

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()
        if state.state == ScreenState.MAP:
            return TaskResult(True, self.name, "Already on map", state.state.value, {"confidence": state.confidence})
        if state.state != ScreenState.CITY:
            return TaskResult(False, self.name, "Cannot open map from UNKNOWN", state.state.value, {"confidence": state.confidence})

        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point("left.map", w, h)
        ctx.adb.tap(x, y)

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            state = ctx.refresh()
            if state.state == ScreenState.MAP:
                return TaskResult(True, self.name, "Map opened", state.state.value, {"confidence": state.confidence, "tap": [x, y]})
            time.sleep(0.25)

        return TaskResult(False, self.name, "Map did not appear after tap", state.state.value, {"confidence": state.confidence, "tap": [x, y]})
