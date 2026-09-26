from __future__ import annotations

import os
import re

from rokbot.core.adb import ADBClient


class ScreenDetector:
    """Detect the foreground Android application and coarse game state."""

    ROK_PACKAGE_HINTS = ("lilith", "roc")

    def __init__(self, adb: ADBClient) -> None:
        self.adb = adb
        self.rok_package = os.getenv("ROK_PACKAGE")

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

    def is_rok(self, package: str | None = None) -> bool:
        package = package or self.current_package()
        if not package:
            return False

        if self.rok_package:
            return package == self.rok_package

        package_lower = package.lower()
        return any(hint in package_lower for hint in self.ROK_PACKAGE_HINTS)

    def state(self) -> str:
        package = self.current_package()

        if not package:
            return "UNKNOWN"

        if package in {"com.android.launcher", "com.android.launcher3"}:
            return "LAUNCHER"

        if self.is_rok(package):
            return "ROK"

        return "APP"
