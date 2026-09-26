from __future__ import annotations

from rokbot.vision.screen_state import detect_current_screen


def main() -> None:
    result = detect_current_screen()

    print("RoK screen detection")
    print("====================")
    print(f"State:      {result.state.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print()
    print("Evidence:")
    for name, score in result.details.items():
        print(f"  {name:24s} {score:.3f}")


if __name__ == "__main__":
    main()
