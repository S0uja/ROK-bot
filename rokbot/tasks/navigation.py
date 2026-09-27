from __future__ import annotations

import time

from rokbot.tasks.base import Task, TaskContext, TaskResult
from rokbot.vision.screen_state import ScreenState
from rokbot.vision.controls import ROKControls


class OpenMapTask:
    name = "open_map"

    def __init__(self, timeout: float = 6.0, map_confidence_threshold: float = 0.70) -> None:
        self.timeout = timeout
        self.map_confidence_threshold = map_confidence_threshold
        self.controls = ROKControls()

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()

        if state.state == ScreenState.MAP and state.confidence >= self.map_confidence_threshold:
            return TaskResult(
                True,
                self.name,
                "Already on map",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        if state.state != ScreenState.CITY:
            return TaskResult(
                False,
                self.name,
                "Cannot open map: screen is not CITY",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        # CITY <-> MAP is the same physical toggle in the lower-left corner.
        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point("left.city_map_toggle", w, h)
        ctx.adb.tap(x, y)

        deadline = time.monotonic() + self.timeout
        last_state = state
        while time.monotonic() < deadline:
            time.sleep(0.25)
            last_state = ctx.refresh()
            if last_state.state == ScreenState.MAP and last_state.confidence >= 0.45:
                return TaskResult(
                    True,
                    self.name,
                    "Map opened",
                    last_state.state.value,
                    {
                        "confidence": last_state.confidence,
                        "tap": [x, y],
                        "details": last_state.details,
                    },
                )

        return TaskResult(
            False,
            self.name,
            "Map did not appear after city/map toggle",
            last_state.state.value,
            {
                "confidence": last_state.confidence,
                "tap": [x, y],
                "details": last_state.details,
            },
        )


class EnsureMapTask:
    """Make the emulator end up on the world map before map-only tasks."""

    name = "ensure_map"

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self.open_map = OpenMapTask(timeout=timeout)

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()

        if state.state == ScreenState.MAP:
            return TaskResult(
                True,
                self.name,
                "Already on map",
                state.state.value,
                {
                    "confidence": state.confidence,
                    "details": state.details,
                },
            )

        if state.state == ScreenState.CITY:
            result = self.open_map.run(ctx)
            return TaskResult(
                result.ok,
                self.name,
                "Opened map automatically" if result.ok else result.message,
                result.state,
                {
                    "from": "CITY",
                    "navigation": result.data,
                },
            )

        return TaskResult(
            False,
            self.name,
            "Cannot navigate to map: screen is UNKNOWN",
            state.state.value,
            {
                "confidence": state.confidence,
                "details": state.details,
            },
        )


class EnsureCityTask:
    name = "ensure_city"

    def __init__(self, timeout: float = 6.0, city_confidence_threshold: float = 0.70) -> None:
        self.timeout = timeout
        self.city_confidence_threshold = city_confidence_threshold
        self.controls = ROKControls()

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()

        # Only skip the toggle when CITY is strong enough. Weak CITY results
        # can be false positives because the map and city share HUD elements.
        if state.state == ScreenState.CITY and state.confidence >= self.city_confidence_threshold:
            return TaskResult(
                True,
                self.name,
                "Already in city",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        if state.state not in (ScreenState.MAP, ScreenState.CITY):
            return TaskResult(
                False,
                self.name,
                "Cannot navigate: screen is UNKNOWN",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point("left.city_map_toggle", w, h)
        ctx.adb.tap(x, y)

        deadline = time.monotonic() + self.timeout
        last_state = state
        while time.monotonic() < deadline:
            time.sleep(0.25)
            last_state = ctx.refresh()
            if last_state.state == ScreenState.CITY and last_state.confidence >= 0.45:
                return TaskResult(
                    True,
                    self.name,
                    "Returned to city",
                    last_state.state.value,
                    {
                        "confidence": last_state.confidence,
                        "tap": [x, y],
                        "details": last_state.details,
                    },
                )

        return TaskResult(
            False,
            self.name,
            "City did not appear after city/map toggle",
            last_state.state.value,
            {
                "confidence": last_state.confidence,
                "tap": [x, y],
                "details": last_state.details,
            },
        )
