from __future__ import annotations

import time

from rokbot.tasks.base import Task, TaskContext, TaskResult
from rokbot.vision.screen_state import ScreenState
from rokbot.vision.controls import ROKControls


class EnsureCityTask:
    name = "ensure_city"

    def __init__(self, timeout: float = 6.0, city_confidence_threshold: float = 0.70) -> None:
        self.timeout = timeout
        self.city_confidence_threshold = city_confidence_threshold
        self.controls = ROKControls()

    def run(self, ctx: TaskContext) -> TaskResult:
        state = ctx.refresh()

        # Do not trust a weak CITY classification. The map and city share a
        # number of HUD elements, so a low-confidence CITY result can be a
        # false positive. Only skip the physical toggle when CITY is strong.
        if state.state == ScreenState.CITY and state.confidence >= self.city_confidence_threshold:
            return TaskResult(
                True,
                self.name,
                "Already in city",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        # The user-visible RoK control is a single CITY <-> MAP toggle in the
        # lower-left corner. If the detector is uncertain, use the control
        # rather than claiming that we are already in the city.
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
        while time.monotonic() < deadline:
            state = ctx.refresh()
            if state.state == ScreenState.CITY and state.confidence >= 0.45:
                return TaskResult(
                    True,
                    self.name,
                    "Returned to city",
                    state.state.value,
                    {"confidence": state.confidence, "tap": [x, y], "details": state.details},
                )
            time.sleep(0.25)

        return TaskResult(
            False,
            self.name,
            "City did not appear after city/map toggle",
            state.state.value,
            {"confidence": state.confidence, "tap": [x, y], "details": state.details},
        )


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
                "Cannot open map from UNKNOWN",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        # Same physical button toggles CITY <-> MAP.
        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point("left.city_map_toggle", w, h)
        ctx.adb.tap(x, y)

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            state = ctx.refresh()
            if state.state == ScreenState.MAP and state.confidence >= 0.45:
                return TaskResult(
                    True,
                    self.name,
                    "Map opened",
                    state.state.value,
                    {"confidence": state.confidence, "tap": [x, y], "details": state.details},
                )
            time.sleep(0.25)

        return TaskResult(
            False,
            self.name,
            "Map did not appear after city/map toggle",
            state.state.value,
            {"confidence": state.confidence, "tap": [x, y], "details": state.details},
        )
