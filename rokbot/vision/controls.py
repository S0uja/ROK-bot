from __future__ import annotations

from pathlib import Path

import yaml


class ROKControls:
    """Semantic click targets calibrated in normalized screen coordinates."""

    def __init__(self, config_path: str | Path = "config/rok_controls.yaml") -> None:
        path = Path(config_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.controls: dict[str, dict[str, float]] = {}

        for group, controls in data.get("controls", {}).items():
            for name, point in controls.items():
                self.controls[f"{group}.{name}"] = point

    def point(self, name: str, width: int, height: int) -> tuple[int, int]:
        point = self.controls[name]
        return round(point["x"] * width), round(point["y"] * height)

    def names(self) -> list[str]:
        return sorted(self.controls)
