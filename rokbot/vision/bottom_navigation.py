from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rokbot.vision.controls import ROKControls
from rokbot.vision.ui_regions import UIRegions


@dataclass(frozen=True)
class ElementDetection:
    name: str
    center: tuple[int, int]
    box: tuple[int, int, int, int]
    confidence: float


class BottomNavigationDetector:
    """Detect the six bottom buttons around calibrated screen centers."""

    NAMES = ("campaign", "items", "alliance", "commander", "mail", "menu")

    def __init__(
        self,
        regions: UIRegions | None = None,
        controls: ROKControls | None = None,
    ) -> None:
        self.regions = regions or UIRegions()
        self.controls = controls or ROKControls()

    @staticmethod
    def _score(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        edge_score = float(np.mean(edges > 0))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_score = float(np.mean(hsv[:, :, 1] > 65))

        score = edge_score * 5.0 + sat_score * 0.35
        return float(max(0.0, min(1.0, score)))

    def detect(self, image: np.ndarray) -> list[ElementDetection]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.regions.get("bottom_navigation").pixels(width, height)

        # The navigation panel has an empty area on the left, so six equal
        # slots are incorrect. Centers come from rok_controls.yaml.
        button_w = max(90, round((y2 - y1) * 0.78))
        button_h = max(90, round((y2 - y1) * 0.78))

        results: list[ElementDetection] = []

        for name in self.NAMES:
            cx, cy = self.controls.point(f"bottom.{name}", width, height)

            bx1 = max(x1, cx - button_w // 2)
            bx2 = min(x2, cx + button_w // 2)
            by1 = max(y1, cy - button_h // 2)
            by2 = min(y2, cy + button_h // 2)

            crop = image[by1:by2, bx1:bx2]
            confidence = self._score(crop)

            results.append(
                ElementDetection(
                    name=name,
                    center=(cx, cy),
                    box=(bx1, by1, bx2, by2),
                    confidence=round(confidence, 3),
                )
            )

        return results


def detect_bottom_navigation(image: np.ndarray) -> list[ElementDetection]:
    return BottomNavigationDetector().detect(image)
