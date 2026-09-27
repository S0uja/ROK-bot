from __future__ import annotations

from io import BytesIO
import struct
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
        self._last_png_decode_ms = 0.0
        self._last_image: np.ndarray | None = None
        self._last_image_time = 0.0
        self._raw_capture_supported: bool | None = None

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

    def _capture_raw_cv(self) -> np.ndarray:
        """Capture Android's uncompressed screencap framebuffer."""
        started = time.perf_counter()
        result = subprocess.run(
            self.adb._cmd("exec-out", "screencap"),
            capture_output=True,
            check=True,
            timeout=15,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        data = result.stdout
        if len(data) < 16:
            raise RuntimeError("ADB raw screencap returned an invalid header")

        width, height, pixel_format, _colorspace = struct.unpack_from("<IIII", data, 0)
        if width <= 0 or height <= 0 or width > 10000 or height > 10000:
            raise RuntimeError(f"Invalid raw screencap dimensions: {width}x{height}")

        bpp = {1: 4, 2: 4, 3: 3, 4: 2, 5: 4}.get(pixel_format)
        if bpp is None:
            raise RuntimeError(f"Unsupported raw screencap pixel format: {pixel_format}")

        expected = width * height * bpp
        payload = data[16:16 + expected]
        if len(payload) != expected:
            raise RuntimeError(
                f"Raw screencap payload is incomplete: {len(payload)} != {expected}"
            )

        flat = np.frombuffer(payload, dtype=np.uint8)
        if pixel_format in (1, 2):
            rgba = flat.reshape((height, width, 4))
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        elif pixel_format == 5:
            bgra = flat.reshape((height, width, 4))
            image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        elif pixel_format == 3:
            rgb = flat.reshape((height, width, 3))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            rgb565 = flat.reshape((height, width, 2))
            image = cv2.cvtColor(rgb565, cv2.COLOR_BGR5652BGR)

        self._last_capture_ms = elapsed_ms
        self._last_png_decode_ms = 0.0
        self._raw_capture_supported = True
        return image

    def capture(self) -> Image.Image:
        image = self.capture_cv()
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def capture_cv(self) -> np.ndarray:
        """Return the current screen as an OpenCV BGR image."""
        with self._capture_lock:
            try:
                image = self._capture_raw_cv()
            except Exception:
                if self._raw_capture_supported is True:
                    raise
                raw = self.capture_bytes()
                decode_started = time.perf_counter()
                image = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                self._last_png_decode_ms = (
                    time.perf_counter() - decode_started
                ) * 1000.0
                if image is None:
                    raise RuntimeError("Could not decode ADB screenshot PNG")
                self._raw_capture_supported = False

            self._last_image = image
            self._last_image_time = time.monotonic()
            return image

    def cached_cv(self, max_age: float = 5.0) -> np.ndarray:
        if (
            self._last_image is not None
            and (time.monotonic() - self._last_image_time) <= max_age
        ):
            return self._last_image
        return self.capture_cv()

    def cached_bytes(self, max_age: float = 5.0) -> bytes:
        """Return the latest frame as PNG without another ADB capture."""
        if (
            self._last_image is not None
            and (time.monotonic() - self._last_image_time) <= max_age
        ):
            ok, encoded = cv2.imencode(".png", self._last_image)
            if not ok:
                raise RuntimeError("OpenCV could not encode cached screenshot")
            return encoded.tobytes()
        return self.capture_bytes()

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
