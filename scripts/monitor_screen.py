from __future__ import annotations

import time
from pathlib import Path

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.detector import ScreenDetector


def main() -> None:
    adb = ADBClient()
    device = adb.select_first_device()

    screen = Screen(adb)
    detector = ScreenDetector(adb)

    print(f"ADB: {adb.executable}")
    print(f"Device: {device}")
    print("Monitoring screen. Press Ctrl+C to stop.\n")

    try:
        while True:
            width, height = screen.size()
            package = detector.current_package() or "unknown"
            state = detector.state()

            print(
                f"state={state:<9} "
                f"package={package:<45} "
                f"resolution={width}x{height}"
            )

            screen.save(Path("screenshots") / "current.png")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
