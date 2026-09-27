from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
from PIL import Image

from rokbot.core.adb import ADBClient


class Screen:
    """High-level screen capture helper for the connected Android device."""

    def __init__(self, adb: ADBClient) -> None:
        self.adb = adb
        self._capture_lock = threading.Lock()
        self._last_bytes: bytes | None = None
        self._last_capture_time = 0.0
        self._last_capture_ms = 0.0

    def capture_bytes(self) -> bytes:
        """Capture one fresh frame from LDPlayer, serializing ADB screencap calls."""
        with self._capture_lock:
            started = time.perf_counter()
            result = subprocess.run(
                self.adb._cmd("exec-out", "screencap", "-p"),
                capture_output=True,
                check=True,
                timeout=15,
            )
            data = result.stdout
            self._last_capture_ms = (time.perf_counter() - started) * 1000.0
            if not data:
                raise RuntimeError("ADB screencap returned empty output")
            self._last_bytes = data
            self._last_capture_time = time.monotonic()
            return data

    def cached_bytes(self, max_age: float = 5.0) -> bytes:
        """Return the latest successful frame, refreshing only when necessary."""
        if self._last_bytes is not None and (time.monotonic() - self._last_capture_time) <= max_age:
            return self._last_bytes
        return self.capture_bytes()

    def capture(self) -> Image.Image:
        return Image.open(BytesIO(self.capture_bytes())).convert("RGB")

    def capture_cv(self) -> np.ndarray:
        """Return the current screen as an OpenCV BGR image."""
        rgb = np.asarray(self.capture())
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def save(self, output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.capture_bytes())
        return output

    def size(self) -> tuple[int, int]:
        image = self.capture()
        return image.width, image.height

    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Image.Image:
        image = self.capture()
        return image.crop((x, y, x + width, y + height))
