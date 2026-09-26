from __future__ import annotations

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.controls import ROKControls


def main() -> None:
    adb = ADBClient()
    device = adb.select_first_device()
    screen = Screen(adb)
    controls = ROKControls()

    width, height = screen.size()

    print(f"Device: {device}")
    print(f"Resolution: {width}x{height}")
    print()

    for name in controls.names():
        x, y = controls.point(name, width, height)
        print(f"{name:<20} -> ({x}, {y})")


if __name__ == "__main__":
    main()
