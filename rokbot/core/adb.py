from __future__ import annotations

import subprocess
from pathlib import Path


class ADBClient:
    def __init__(self, executable: str = "adb", device: str | None = None) -> None:
        self.executable = executable
        self.device = device

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

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
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
