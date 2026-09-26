from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from rokbot.vision.ui_regions import UIRegions


@dataclass(frozen=True)
class ElementDetection:
    name: str
    center: tuple[int, int]
    box: tuple[int, int, int, int]
    confidence: float


class BottomNavigationDetector:
    """Detect/score the six bottom navigation slots.

    The slots are defined relative to the calibrated bottom_navigation region.
    This avoids hard-coded 3440x1440 coordinates and keeps detection resolution
    independent. Confidence is a visual-presence score, not a semantic OCR claim.
    """

    def __init__(self, regions: UIRegions | None = None) -> None:
        self.regions = regions or UIRegions()
        self.names = (
            "campaign",
            "items",
            "alliance",
            "commander",
            "mail",
            "menu",
        )

    @staticmethod
    def _score(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        edge_score = float(np.mean(edges > 0))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_score = float(np.mean(hsv[:, :, 1] > 65))

        # Icons/text produce local edges; colored UI elements add saturation.
        score = edge_score * 5.0 + sat_score * 0.35
        return float(max(0.0, min(1.0, score)))

    def detect(self, image: np.ndarray) -> list[ElementDetection]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.regions.get("bottom_navigation").pixels(width, height)

        region_w = x2 - x1
        region_h = y2 - y1

        # Six equal slots across the calibrated navigation bar.
        results: list[ElementDetection] = []
        slot_w = region_w / 6.0

        for index, name in enumerate(self.names):
            sx1 = round(x1 + index * slot_w)
            sx2 = round(x1 + (index + 1) * slot_w)

            # Ignore the extreme top/bottom edges where panel borders can
            # otherwise inflate the visual score.
            pad_x = max(4, round((sx2 - sx1) * 0.12))
            pad_top = max(4, round(region_h * 0.08))
            pad_bottom = max(4, round(region_h * 0.10))

            bx1 = sx1 + pad_x
            bx2 = sx2 - pad_x
            by1 = y1 + pad_top
            by2 = y2 - pad_bottom

            crop = image[by1:by2, bx1:bx2]
            confidence = self._score(crop)

            results.append(
                ElementDetection(
                    name=name,
                    center=((bx1 + bx2) // 2, (by1 + by2) // 2),
                    box=(bx1, by1, bx2, by2),
                    confidence=round(confidence, 3),
                )
            )

        return results


def detect_bottom_navigation(image: np.ndarray) -> list[ElementDetection]:
    return BottomNavigationDetector().detect(image)
