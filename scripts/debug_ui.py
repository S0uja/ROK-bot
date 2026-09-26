from __future__ import annotations

from pathlib import Path

import cv2

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.ui_regions import UIRegions


def main() -> None:
    adb = ADBClient()
    device = adb.select_first_device()

    screen = Screen(adb)
    regions = UIRegions()

    image = screen.capture_cv()
    height, width = image.shape[:2]

    for name, (x1, y1, x2, y2) in regions.boxes(width, height).items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            image,
            name,
            (x1 + 8, min(y1 + 28, height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output = Path("screenshots") / "debug" / "rok_ui_regions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)

    print(f"Device: {device}")
    print(f"Resolution: {width}x{height}")
    print(f"Debug overlay: {output}")
    print("Regions:")
    for name, box in regions.boxes(width, height).items():
        print(f"  {name}: {box}")


if __name__ == "__main__":
    main()
