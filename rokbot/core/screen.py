from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rokbot.core.adb import ADBClient


class Screen:
    """High-level screen capture helper for the connected Android device."""

    def __init__(self, adb: ADBClient) -> None:
        self.adb = adb

    def capture_bytes(self) -> bytes:
        result = __import__("subprocess").run(
            self.adb._cmd("exec-out", "screencap", "-p"),
            capture_output=True,
            check=True,
        )
        return result.stdout

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
