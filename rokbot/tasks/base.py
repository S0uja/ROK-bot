from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.screen_state import ScreenState, ScreenStateDetector, StateResult


@dataclass
class TaskContext:
    adb: ADBClient
    screen: Screen
    state_detector: ScreenStateDetector
    image: np.ndarray | None = None
    state: StateResult | None = None
    data: dict = field(default_factory=dict)

    def refresh(self) -> StateResult:
        self.image = self.screen.capture_cv()
        self.state = self.state_detector.detect(self.image)
        return self.state

    def require_state(self, expected: ScreenState) -> bool:
        result = self.refresh()
        return result.state == expected


@dataclass(frozen=True)
class TaskResult:
    ok: bool
    task: str
    message: str
    state: str | None = None
    data: dict = field(default_factory=dict)


class Task(Protocol):
    name: str

    def run(self, ctx: TaskContext) -> TaskResult:
        ...
