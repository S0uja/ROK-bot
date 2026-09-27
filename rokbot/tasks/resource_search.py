from __future__ import annotations

import time

from rokbot.tasks.base import TaskContext, TaskResult
from rokbot.vision.controls import ROKControls


class SearchResourceTask:
    """Search a resource, collect it, and send a new troop march.

    This is deliberately coordinate-driven: the RoK search dialog is a fixed
    UI flow on the calibrated emulator, so there is no need to detect map
    resource objects with CV.
    """

    name = "search_resource"

    RESOURCES = {"food", "wood", "stone", "gold"}

    def __init__(self, resource: str = "food", level: int = 3, timeout: float = 3.0) -> None:
        self.resource = resource
        self.level = max(1, min(10, int(level)))
        self.timeout = timeout
        self.controls = ROKControls()

    def _tap(self, ctx: TaskContext, name: str, pause: float = 0.25) -> tuple[int, int]:
        image = ctx.image
        h, w = image.shape[:2]
        x, y = self.controls.point(name, w, h)
        ctx.adb.tap(x, y)
        if pause:
            time.sleep(pause)
        return x, y

    def run(self, ctx: TaskContext) -> TaskResult:
        resource = self.resource.lower().strip()
        if resource not in self.RESOURCES:
            return TaskResult(False, self.name, f"Unknown resource: {resource}", data={"resource": resource})

        state = ctx.refresh()
        # Search is only available from the world map. We intentionally do not
        # navigate automatically here; navigation remains a separate task.
        if state.state.value != "MAP":
            return TaskResult(
                False,
                self.name,
                "Resource search requires MAP",
                state.state.value,
                {"confidence": state.confidence, "details": state.details},
            )

        taps = {}
        taps["resource_search"] = self._tap(ctx, "left.resource_search")

        # Follow the exact visible UI flow: choose the resource first,
        # then set the requested level, then press SEARCH.
        taps["resource"] = self._tap(ctx, f"resource_search.{resource}")

        # The dialog opens with a remembered/default level. Reset to level 1
        # with the minus control, then move to the requested level.
        for _ in range(10):
            taps["level_minus"] = self._tap(ctx, "resource_search.level_minus", pause=0.08)
        for _ in range(self.level - 1):
            taps["level_plus"] = self._tap(ctx, "resource_search.level_plus", pause=0.08)

        taps["search"] = self._tap(ctx, "resource_search.search", pause=0.8)
        taps["collect"] = self._tap(ctx, "resource_search.collect", pause=0.8)
        taps["new_troops"] = self._tap(ctx, "resource_search.new_troops", pause=0.8)
        taps["march"] = self._tap(ctx, "resource_search.march", pause=0.8)

        return TaskResult(
            True,
            self.name,
            f"Resource found and collection started: {resource} level {self.level}",
            "MAP",
            {"resource": resource, "level": self.level, "taps": taps},
        )
