from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import cv2
import numpy as np

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
    """Conservative CITY/MAP/UNKNOWN detector for the current RoK layout.

    MAP detection uses the coordinate strip shown by RoK on the world map
    (e.g. '#4188 X:570 Y:33') when OCR is available, plus visual evidence.
    CITY detection uses the city character/navigation structure and city-area
    texture. No action should be triggered from UNKNOWN.
    """

    MAP_COORDINATE_RE = re.compile(
        r"(?:X\s*[:=]?\s*\d{1,5}).{0,20}(?:Y\s*[:=]?\s*\d{1,5})",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, regions: UIRegions | None = None) -> None:
        self.regions = regions or UIRegions()

    @staticmethod
    def _edge_score(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.mean(edges > 0))

    @staticmethod
    def _color_score(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1] > 70))

    @staticmethod
    def _green_score(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # RoK world-map terrain is predominantly green/yellow-green.
        mask = (
            (hsv[:, :, 0] >= 25)
            & (hsv[:, :, 0] <= 75)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 65)
        )
        return float(np.mean(mask))

    @staticmethod
    def _ocr_map_coordinates(image: np.ndarray) -> float:
        """Return 1.0 when the world-map X/Y coordinate text is detected.

        OCR is optional: if Tesseract is unavailable, this feature contributes
        zero rather than making screen detection fail.
        """
        try:
            import pytesseract
        except Exception:
            return 0.0

        height, width = image.shape[:2]

        # Coordinate strip is immediately right of the profile in the current
        # 3440x1440 layout. Keep OCR narrowly scoped to reduce false positives.
        x1, x2 = round(width * 0.145), round(width * 0.36)
        y1, y2 = 0, round(height * 0.09)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return 0.0

        scale = 2
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]

        try:
            text = pytesseract.image_to_string(
                gray,
                config="--psm 7",
            )
        except Exception:
            return 0.0

        normalized = text.replace("×", "X").replace(":", ":")
        return 1.0 if ScreenStateDetector.MAP_COORDINATE_RE.search(normalized) else 0.0

    def detect(self, image: np.ndarray) -> StateResult:
        height, width = image.shape[:2]
        boxes = self.regions.boxes(width, height)

        character = self._crop(image, boxes["right_character"])
        bottom = self._crop(image, boxes["bottom_navigation"])
        city = self._crop(image, boxes["city_area"])
        tasks = self._crop(image, boxes["left_tasks"])
        resources = self._crop(image, boxes["resources"])

        character_score = min(1.0, self._color_score(character) * 2.0)
        bottom_score = min(1.0, self._edge_score(bottom) * 5.0)
        city_green = min(1.0, self._green_score(city) * 1.5)
        city_edges = min(1.0, self._edge_score(city) * 6.0)
        task_edges = min(1.0, self._edge_score(tasks) * 5.0)
        resource_edges = min(1.0, self._edge_score(resources) * 8.0)
        resource_color = min(1.0, self._color_score(resources) * 1.8)

        # The calibrated city HUD is a strong local signal. It is cheap and
        # does not require OCR: the resource strip contains five colored,
        # separated counters in CITY and remains stable while the scene moves.
        city_hud = min(
            1.0,
            0.45 * resource_edges
            + 0.30 * resource_color
            + 0.15 * task_edges
            + 0.10 * character_score,
        )

        map_ocr = self._ocr_map_coordinates(image)

        # World-map terrain is broad and green; city view has much denser
        # building edges. OCR is the strongest signal when available.
        map_visual = min(
            1.0,
            city_green * 0.55 + max(0.0, 1.0 - city_edges) * 0.45,
        )
        map_confidence = max(
            map_ocr,
            0.35 * map_visual + 0.20 * (1.0 - character_score) + 0.10 * (1.0 - bottom_score),
        )

        city_confidence = (
            0.35 * character_score
            + 0.25 * bottom_score
            + 0.15 * city_edges
            + 0.15 * city_hud
            + 0.10 * (1.0 - map_ocr)
        )

        # Do not let a weak visual score block navigation when the map
        # coordinate strip is absent and the calibrated city HUD is present.
        # This is intentionally below the MAP threshold: MAP still requires
        # strong coordinate evidence or a clearly map-like visual.
        strong_city_layout = (
            map_ocr == 0.0
            and character_score >= 0.45
            and city_hud >= 0.20
        )

        if map_confidence >= 0.75 and map_confidence > city_confidence:
            state = ScreenState.MAP
            confidence = map_confidence
        elif strong_city_layout or city_confidence >= 0.45:
            state = ScreenState.CITY
            confidence = max(city_confidence, 0.45)
        else:
            state = ScreenState.UNKNOWN
            confidence = max(map_confidence, city_confidence)

        return StateResult(
            state=state,
            confidence=round(float(min(1.0, confidence)), 3),
            details={
                "map_coordinates_ocr": round(map_ocr, 3),
                "map_visual": round(map_visual, 3),
                "character": round(character_score, 3),
                "bottom_navigation": round(bottom_score, 3),
                "city_edges": round(city_edges, 3),
                "city_green": round(city_green, 3),
                "task_edges": round(task_edges, 3),
                "resource_edges": round(resource_edges, 3),
                "resource_color": round(resource_color, 3),
                "city_hud": round(city_hud, 3),
            },
        )

    @staticmethod
    def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]


def detect_current_screen() -> StateResult:
    from rokbot.core.adb import ADBClient
    from rokbot.core.screen import Screen

    adb = ADBClient()
    adb.select_first_device()
    return ScreenStateDetector().detect(Screen(adb).capture_cv())
