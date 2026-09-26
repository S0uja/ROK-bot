from __future__ import annotations

from pathlib import Path

import cv2

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.bottom_navigation import BottomNavigationDetector


OUTPUT = Path(__file__).resolve().parents[1] / "screenshots" / "debug" / "bottom_navigation.png"


def main() -> None:
    adb = ADBClient()
    device = adb.select_first_device()
    screen = Screen(adb)
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", debug)
    if not ok:
        raise RuntimeError("Could not encode debug image.")

    OUTPUT.write_bytes(encoded.tobytes())
    print(f"Debug image: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
