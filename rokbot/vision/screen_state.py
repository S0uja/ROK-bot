from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.ui_regions import UIRegions


class ScreenState(str, Enum):
    CITY = "CITY"
    MAP = "MAP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StateResult:
    state: ScreenState
    confidence: float
    details: dict[str, float]


class ScreenStateDetector:
    """Initial RoK city/map detector using visual geometry.

    This intentionally uses conservative heuristics rather than OCR/ML.
    It is a first safety layer before automated actions are added.
    """

    def __init__(self, regions: UIRegions | None = None) -> None:
        self.regions = regions or UIRegions()

    @staticmethod
    def _edge_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.mean(edges > 0))

    @staticmethod
    def _color_score(image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return float(np.mean(saturation > 70))

    def detect(self, image: np.ndarray) -> StateResult:
        height, width = image.shape[:2]
        boxes = self.regions.boxes(width, height)

        # A city screen normally contains the right character panel and
        # bottom navigation simultaneously. These are strong structural cues.
        character = self._crop(image, boxes["right_character"])
        bottom = self._crop(image, boxes["bottom_navigation"])
        chat = self._crop(image, boxes["chat"])
        city = self._crop(image, boxes["city_area"])

        character_score = min(1.0, self._color_score(character) * 2.0)
        bottom_score = min(1.0, self._edge_score(bottom) * 5.0)
        chat_score = min(1.0, self._edge_score(chat) * 5.0)
        city_score = min(1.0, self._color_score(city) * 1.5)

        city_confidence = (
            0.38 * character_score
            + 0.27 * bottom_score
            + 0.20 * chat_score
            + 0.15 * city_score
        )

        # MAP is intentionally only reported when city evidence is weak.
        # A later detector can add a dedicated map template/icon detector.
        if city_confidence >= 0.45:
            state = ScreenState.CITY
            confidence = city_confidence
        else:
            state = ScreenState.UNKNOWN
            confidence = max(0.0, 1.0 - city_confidence)

        return StateResult(
            state=state,
            confidence=round(float(confidence), 3),
            details={
                "character": round(character_score, 3),
                "bottom_navigation": round(bottom_score, 3),
                "chat": round(chat_score, 3),
                "city_area": round(city_score, 3),
            },
        )

    @staticmethod
    def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]


def detect_current_screen() -> StateResult:
    adb = ADBClient()
    adb.select_first_device()
    screen = Screen(adb)
    detector = ScreenStateDetector()
    return detector.detect(screen.capture_cv())
