from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import tempfile

import cv2

from rokbot.core.adb import ADBClient
from rokbot.core.config import load_settings
from rokbot.core.screen import Screen
from rokbot.vision.bottom_navigation import BottomNavigationDetector



def save_png(image, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode debug image.")

    data = encoded.tobytes()
    temp_path: Path | None = None

    try:
        fd, temp_name = tempfile.mkstemp(
            prefix="bottom_navigation_",
            suffix=".tmp",
            dir=str(output.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.write_bytes(data)
        os.replace(temp_path, output)
        return output
    except (OSError, PermissionError):
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        fallback = output.with_name(
            f"{output.stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
        )
        fallback.write_bytes(data)
        return fallback


def main() -> None:
    adb = ADBClient()
    device = adb.select_first_device()
    screen = Screen(adb)
    settings = load_settings()
    image = screen.capture_cv()

    detector = BottomNavigationDetector()
    detections = detector.detect(image)

    print(f"Device: {device}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
    print("Bottom navigation")
    print("=================")

    for item in detections:
        print(
            f"{item.name:10s} "
            f"confidence={item.confidence:.3f} "
            f"center={item.center} "
            f"box={item.box}"
        )

    debug = image.copy()

    for item in detections:
        x1, y1, x2, y2 = item.box
        color = (0, 255, 0) if item.confidence >= 0.45 else (0, 165, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 4)
        cv2.circle(debug, item.center, 8, color, -1)

        label = f"{item.name} {item.confidence:.2f}"
        cv2.putText(
            debug,
            label,
            (x1 + 8, max(y1 + 28, y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    saved = save_png(debug, settings.screenshot_dir / "debug" / "bottom_navigation.png")
    print(f"Debug image: {saved.resolve()}")
    print(f"File size: {saved.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
