from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from rokbot.core.config import load_settings


WINDOWS_ADB_CANDIDATES = (
    Path(r"C:\LDPlayer\LDPlayer9\adb.exe"),
    Path(r"C:\LDPlayer\LDPlayer4\adb.exe"),
)


def find_adb() -> str:
    settings = load_settings()
    configured_setting = settings.adb_executable
    if configured_setting and configured_setting.lower() not in {"auto", "adb"} and Path(configured_setting).is_file():
        return configured_setting
    """Find the ADB binary used by the emulator.

    ROK-bot prefers an explicit ROK_ADB_PATH, then LDPlayer's bundled ADB,
    and only then falls back to ADB available on PATH.
    """
    configured = os.getenv("ROK_ADB_PATH")
    if configured and Path(configured).is_file():
        return configured

    for candidate in WINDOWS_ADB_CANDIDATES:
        if candidate.is_file():
            return str(candidate)

    path_adb = shutil.which("adb")
    if path_adb:
        return path_adb

    raise FileNotFoundError(
        "ADB was not found. Set ROK_ADB_PATH or install/configure an Android emulator."
    )


class ADBClient:
    def __init__(self, executable: str | None = None, device: str | None = None) -> None:
        settings = load_settings()
        self.executable = executable or find_adb()
        configured_device = settings.adb_device
        self.device = device or (configured_device if configured_device.lower() != "auto" else None)

    def _cmd(self, *args: str) -> list[str]:
        cmd = [self.executable]
        if self.device:
            cmd += ["-s", self.device]
        cmd += list(args)
        return cmd

    def devices(self) -> list[str]:
        result = subprocess.run(
            [self.executable, "devices"],
            capture_output=True,
            text=True,
            check=True,
        )
        devices: list[str] = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def select_first_device(self) -> str:
        if self.device and self.device in self.devices():
            return self.device
        devices = self.devices()
        if not devices:
            raise RuntimeError(
                f"No Android devices found by {self.executable}. "
                "Start the emulator and enable ADB debugging."
            )
        self.device = devices[0]
        return self.device

    def display_size(self) -> tuple[int, int]:
        """Return Android/LDPlayer's configured display size via wm size."""
        output = self.shell("wm", "size")
        # Physical size: 3440x1440
        import re
        match = re.search(r"(?:Physical size|Override size):\s*(\d+)x(\d+)", output)
        if not match:
            match = re.search(r"(\d+)x(\d+)", output)
        if not match:
            raise RuntimeError(f"Unable to parse emulator display size: {output!r}")
        return int(match.group(1)), int(match.group(2))

    def shell(self, *args: str) -> str:
        result = subprocess.run(
            self._cmd("shell", *args),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(x), str(y))

    def dump_ui_hierarchy(self) -> str:
        """Return Android UI Automator hierarchy for the current screen."""
        remote = "/sdcard/window.xml"
        self.shell("uiautomator", "dump", remote)
        return self.shell("cat", remote)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        self.shell(
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def screenshot(self, output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            self._cmd("exec-out", "screencap", "-p"),
            capture_output=True,
            check=True,
        )
        output.write_bytes(result.stdout)
        return output
