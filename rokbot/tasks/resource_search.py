from __future__ import annotations

import time

from rokbot.tasks.base import TaskContext, TaskResult
from rokbot.tasks.navigation import EnsureMapTask
from rokbot.vision.controls import ROKControls
from rokbot.vision.screen_state import ScreenState


class SearchResourceTask:
    """Search a resource, collect it, and send a new troop march.

    The task owns the full prerequisite flow: if the game is in CITY it first
    opens the world map, then performs the resource-search sequence.
    """

    name = "search_resource"

    RESOURCES = {"food", "wood", "stone", "gold"}

    def __init__(self, resource: str = "food", level: int = 3, timeout: float = 6.0) -> None:
        self.resource = resource
        self.level = max(1, min(10, int(level)))
        self.timeout = timeout
        self.controls = ROKControls()
        self.ensure_map = EnsureMapTask(timeout=timeout)

    def _tap(self, ctx: TaskContext, name: str, pause: float = 0.25) -> tuple[int, int]:
        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point(name, w, h)
        ctx.adb.tap(x, y)
        if pause:
            time.sleep(pause)
        return x, y

    def _tap_search_button(self, ctx: TaskContext) -> tuple[int, int]:
        """Detect the orange SEARCH button in a fresh dialog frame."""
        import cv2
        import numpy as np

        image = ctx.screen.capture_cv()
        ctx.image = image
        h, w = image.shape[:2]

        # The SEARCH button is in the lower-middle of the resource dialog.
        # A narrow ROI prevents map markers and HUD icons from being selected.
        x1, x2 = int(w * 0.28), int(w * 0.46)
        y1, y2 = int(h * 0.57), int(h * 0.69)
        roi = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([3, 90, 100], dtype=np.uint8),
            np.array([28, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(300.0, w * h * 0.00002):
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < w * 0.04 or bw < bh * 1.8:
                continue
            candidates.append((area, bx, by, bw, bh))

        if candidates:
            _, bx, by, bw, bh = max(candidates, key=lambda item: item[0])
            x = x1 + bx + bw // 2
            y = y1 + by + bh // 2
        else:
            # Calibrated fallback if the button color is temporarily obscured.
            x, y = self.controls.point("resource_search.search", w, h)

        ctx.adb.tap(x, y)
        time.sleep(0.8)
        return x, y

    def run(self, ctx: TaskContext) -> TaskResult:
        resource = self.resource.lower().strip()
        if resource not in self.RESOURCES:
            return TaskResult(
                False,
                self.name,
                f"Unknown resource: {resource}",
                data={"resource": resource},
            )

        # Map is a prerequisite of resource search, so navigation is part of
        # this task rather than a separate manual action.
        navigation = self.ensure_map.run(ctx)
        if not navigation.ok:
            return TaskResult(
                False,
                self.name,
                f"Cannot start resource search: {navigation.message}",
                navigation.state,
                {"navigation": navigation.data},
            )

        state = ctx.refresh()
        if state.state != ScreenState.MAP:
            return TaskResult(
                False,
                self.name,
                "Map was not confirmed after navigation",
                state.state.value,
                {
                    "confidence": state.confidence,
                    "details": state.details,
                    "navigation": navigation.data,
                },
            )

        taps = {}
        taps["resource_search"] = self._tap(ctx, "left.resource_search")

        # Choose the resource first, then reset the remembered level to 1 and
        # raise it to the requested level.
        taps["resource"] = self._tap(ctx, f"resource_search.{resource}")

        for _ in range(10):
            taps["level_minus"] = self._tap(
                ctx, "resource_search.level_minus", pause=0.08
            )

        for _ in range(self.level - 1):
            taps["level_plus"] = self._tap(
                ctx, "resource_search.level_plus", pause=0.08
            )

        taps["search"] = self._tap_search_button(ctx)
        taps["collect"] = self._tap(ctx, "resource_search.collect", pause=0.8)
        taps["new_troops"] = self._tap(ctx, "resource_search.new_troops", pause=0.8)
        taps["march"] = self._tap(ctx, "resource_search.march", pause=0.8)

        return TaskResult(
            True,
            self.name,
            f"Resource found and collection started: {resource} level {self.level}",
            "MAP",
            {
                "resource": resource,
                "level": self.level,
                "navigation": navigation.data,
                "taps": taps,
            },
        )
