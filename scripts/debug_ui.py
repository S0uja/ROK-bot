from __future__ import annotations

from pathlib import Path

import cv2

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.ui_regions import UIRegions


# BGR colors chosen to make adjacent regions easy to distinguish.
REGION_COLORS = [
    (0, 255, 255),    # yellow
    (255, 0, 0),      # blue
    (0, 165, 255),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (255, 255, 0),    # cyan
    (147, 20, 255),   # pink
    (0, 0, 255),      # red
]


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
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use imencode + write_bytes instead of cv2.imwrite. This is more reliable
    # on Windows and lets us verify that the PNG bytes were actually produced.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("OpenCV could not encode the debug image as PNG.")

    output.write_bytes(encoded.tobytes())

    print(f"Device: {device}")
    print(f"Resolution: {width}x{height}")
    print(f"Saved: True")
    print(f"Debug overlay: {output.resolve()}")
    print(f"File size: {output.stat().st_size:,} bytes")

    print("Regions:")
    for name, box in boxes.items():
        print(f"  {name}: {box}")


if __name__ == "__main__":
    main()
