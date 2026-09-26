from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Region:
    name: str
    x: float
    y: float
    w: float
    h: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = round(self.x * width)
        y1 = round(self.y * height)
        x2 = round((self.x + self.w) * width)
        y2 = round((self.y + self.h) * height)
        return x1, y1, x2, y2


class UIRegions:
    def __init__(self, config_path: str | Path = "config/rok_ui.yaml") -> None:
        self.config_path = Path(config_path)
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.regions = {
            name: Region(name=name, **values)
            for name, values in data.get("regions", {}).items()
        }

    def get(self, name: str) -> Region:
        return self.regions[name]

    def boxes(self, width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
        return {
            name: region.pixels(width, height)
            for name, region in self.regions.items()
        }
