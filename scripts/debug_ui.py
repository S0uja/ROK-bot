from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import tempfile

import cv2

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.ui_regions import UIRegions


REGION_COLORS = [
    (0, 255, 255),
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (147, 20, 255),
    (0, 0, 255),
]


def save_png(image, output: Path) -> Path:
    """Save PNG robustly on Windows, including when the previous file is open."""
    output.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("OpenCV could not encode the debug image as PNG.")

    data = encoded.tobytes()

    # Write to a temporary file in the same directory, then atomically replace
    # the target. If Windows has the old target locked, fall back to a timestamp.
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix="rok_ui_regions_",
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
    regions = UIRegions()

    image = screen.capture_cv()
    height, width = image.shape[:2]
    boxes = regions.boxes(width, height)

    for index, (name, (x1, y1, x2, y2)) in enumerate(boxes.items()):
        color = REGION_COLORS[index % len(REGION_COLORS)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

        (text_width, text_height), baseline = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            2,
        )
        label_y = max(y1, text_height + baseline + 4)
        label_x2 = min(width, x1 + text_width + 16)

        cv2.rectangle(
            image,
            (x1, label_y - text_height - baseline - 8),
            (label_x2, label_y),
            color,
            -1,
        )
        cv2.putText(
            image,
            name,
            (x1 + 8, label_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output = (
        Path(__file__).resolve().parents[1]
        / "screenshots"
        / "debug"
        / "rok_ui_regions.png"
    )

    saved = save_png(image, output)

    print(f"Device: {device}")
    print(f"Resolution: {width}x{height}")
    print("Saved: True")
    print(f"Debug overlay: {saved.resolve()}")
    print(f"File size: {saved.stat().st_size:,} bytes")

    print("Regions:")
    for name, box in boxes.items():
        print(f"  {name}: {box}")


if __name__ == "__main__":
    main()
