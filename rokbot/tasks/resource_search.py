from __future__ import annotations

import time
from pathlib import Path

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
        w, h = ctx.adb.display_size()
        x, y = self.controls.point(name, w, h)
        ctx.adb.tap(x, y)
        print(f"[RESOURCE] tap {name}: ({x}, {y}) on {w}x{h}")
        if pause:
            time.sleep(pause)
        return x, y

    def _debug_frame(self, ctx: TaskContext, filename: str) -> None:
        path = ctx.screen.save(Path("logs") / filename)
        print(f"[RESOURCE] screenshot: {path}")


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

        self._debug_frame(ctx, "resource_before_search.png")
        taps["search"] = self._tap(ctx, "resource_search.search", pause=0.8)
        self._debug_frame(ctx, "resource_after_search.png")
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
