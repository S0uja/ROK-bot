from pathlib import Path

from rokbot.core.adb import ADBClient


def main() -> None:
    adb = ADBClient()

    devices = adb.devices()
    print(f"Devices: {devices}")

    if not devices:
        print("No Android device/emulator detected.")
        return

    adb.device = devices[0]
    path = adb.screenshot(Path("screenshots/test.png"))
    print(f"Screenshot saved to: {path}")


if __name__ == "__main__":
    main()
