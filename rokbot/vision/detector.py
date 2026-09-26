from __future__ import annotations

import re

from rokbot.core.adb import ADBClient


class ScreenDetector:
    """Lightweight screen/state detector.

    The first version intentionally uses Android's window information rather
    than fragile pixel coordinates. Visual detectors can be added later.
    """

    def __init__(self, adb: ADBClient) -> None:
        self.adb = adb

    def current_package(self) -> str | None:
        output = self.adb.shell("dumpsys", "window", "windows")
        patterns = (
            r"mCurrentFocus=Window\{[^ ]+ u\d+ ([^/ ]+)/",
            r"mFocusedApp=ActivityRecord\{[^ ]+ u\d+ ([^/ ]+)/",
        )

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)

        return None

    def state(self) -> str:
        package = self.current_package()

        if not package:
            return "UNKNOWN"

        if package in {"com.android.launcher", "com.android.launcher3"}:
            return "LAUNCHER"

        return "APP"
