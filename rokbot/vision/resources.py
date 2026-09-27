from __future__ import annotations
from dataclasses import dataclass
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from rokbot.vision.ui_regions import UIRegions

RESOURCE_NAMES=("food","wood","stone","gold","gems")

@dataclass(frozen=True)
class ResourceDetection:
    values: dict[str,str]
    raw: str
    candidates: list[dict]
    available: bool
    error: str|None
    boxes: dict[str, tuple[int,int,int,int]] = None

class ResourceDetector:
    """Detect resources only inside the calibrated resources UI region."""
    def __init__(self,regions:UIRegions|None=None)->None:
        self.regions=regions or UIRegions()
        self._configure_tesseract()
        self._stable_values: dict[str, str] = {}
        self._pending_values: dict[str, str] = {}
        self._pending_counts: dict[str, int] = {}
        self._stable_boxes: dict[str, tuple[int,int,int,int]] = {}

    @staticmethod
    def _configure_tesseract():
        for p in (Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),Path.home()/"AppData/Local/Programs/Tesseract-OCR/tesseract.exe"):
            if p.exists():
                pytesseract.pytesseract.tesseract_cmd=str(p); return str(p)
        found=shutil.which("tesseract")
        if found: pytesseract.pytesseract.tesseract_cmd=found
        return found

    @staticmethod
    def _token(raw:str)->str|None:
        token=raw.strip().replace(" ","").replace("%","K")
        token=re.sub(r"[^0-9.,KMBTkmbt]","",token)
        if not re.fullmatch(r"\d[\d.,]*[KMBT]?",token,re.I): return None
        if len(re.sub(r"[^0-9]","",token))<2: return None
        return token

    def _assign_candidates(self, candidates: list[dict], width: float) -> dict[str, str]:
        """Assign OCR tokens to calibrated resource anchors."""
        anchors = self.regions.anchors("resources")
        if not anchors:
            return {}

        values: dict[str, str] = {}
        used: set[int] = set()

        for name in RESOURCE_NAMES:
            anchor = anchors.get(name)
            if anchor is None:
                continue

            best_index = None
            best_distance = float("inf")
            for index, item in enumerate(candidates):
                if index in used:
                    continue
                distance = abs((item["x"] / max(width, 1.0)) - anchor)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None and best_distance <= 0.12:
                values[name] = candidates[best_index]["value"]
                used.add(best_index)

        return values

    @staticmethod
    def _extract_tokens(data: dict, x1: int, scale: int) -> list[dict]:
        """Extract confident numeric tokens and merge split thousands groups."""
        tokens: list[dict] = []

        for i, raw in enumerate(data.get("text", [])):
            token = ResourceDetector._token(raw)
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0

            # Very low-confidence garbage is much more likely to be a
            # fragment from an icon/background than the resource counter.
            if conf < 30:
                continue

            left = float(data["left"][i]) / scale
            top = float(data["top"][i]) / scale
            right = (float(data["left"][i]) + float(data["width"][i])) / scale
            bottom = (float(data["top"][i]) + float(data["height"][i])) / scale

            tokens.append({
                "value": token,
                "confidence": round(conf, 1),
                "box": (
                    round(x1 + left),
                    round(top),
                    round(x1 + right),
                    round(bottom),
                ),
            })

        tokens.sort(key=lambda item: (item["box"][1], item["box"][0]))

        # RoK can render a resource counter with a thousands separator
        # (for example "92 658"). Tesseract may return that as two tokens
        # ("92" + "658"). Merge adjacent numeric fragments before selecting
        # the result, otherwise the OCR can incorrectly expose only "658".
        merged: list[dict] = []
        for item in tokens:
            if not merged:
                merged.append(item)
                continue

            prev = merged[-1]
            px1, py1, px2, py2 = prev["box"]
            x1i, y1i, x2i, y2i = item["box"]

            prev_h = max(1, py2 - py1)
            item_h = max(1, y2i - y1i)
            same_line = abs(((py1 + py2) / 2) - ((y1i + y2i) / 2)) <= max(prev_h, item_h) * 0.45
            gap = x1i - px2
            close = gap <= max(8.0, min(prev_h, item_h) * 0.75)

            prev_digits = re.sub(r"[^0-9]", "", prev["value"])
            item_digits = re.sub(r"[^0-9]", "", item["value"])
            can_merge = (
                same_line
                and close
                and prev_digits
                and item_digits
                and len(prev_digits) <= 3
                and len(item_digits) <= 3
                and len(prev_digits) + len(item_digits) >= 4
            )

            if can_merge:
                merged[-1] = {
                    "value": prev["value"] + item["value"],
                    "confidence": round(min(prev["confidence"], item["confidence"]), 1),
                    "box": (
                        min(px1, x1i),
                        min(py1, y1i),
                        max(px2, x2i),
                        max(py2, y2i),
                    ),
                }
            else:
                merged.append(item)

        return merged

    def _read_anchor(self, crop: np.ndarray, anchor: float, scale: int = 3) -> dict | None:
        """OCR one calibrated resource slot, using the minimum needed passes."""
        h, w = crop.shape[:2]
        center = int(anchor * w)
        half = max(70, int(w * 0.085))
        x1 = max(0, center - half)
        x2 = min(w, center + half)
        slot = crop[:, x1:x2]
        slot = cv2.resize(slot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)

        variants = [
            gray,
            cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1],
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 7
            ),
        ]
        best = None

        def run(variant: np.ndarray, psm: int):
            try:
                return pytesseract.image_to_data(
                    variant,
                    config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.,KMBT",
                    output_type=pytesseract.Output.DICT,
                    timeout=1.0,
                )
            except RuntimeError:
                return None

        for variant in variants:
            data = run(variant, 7)
            if not data:
                continue

            parsed = self._extract_tokens(data, x1, scale)
            if parsed:
                candidate = max(parsed, key=lambda item: item["confidence"])
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
                # A merged multi-token number is more trustworthy than a
                # single short fragment, so return it immediately.
                if any(len(re.sub(r"[^0-9]", "", item["value"])) >= 4 for item in parsed):
                    return max(
                        parsed,
                        key=lambda item: (
                            len(re.sub(r"[^0-9]", "", item["value"])) >= 4,
                            item["confidence"],
                        ),
                    )

            if best is not None and best["confidence"] >= 45:
                return best

        data = run(gray, 11)
        if data:
            parsed = self._extract_tokens(data, x1, scale)
            if parsed:
                return max(
                    parsed,
                    key=lambda item: (
                        len(re.sub(r"[^0-9]", "", item["value"])) >= 4,
                        item["confidence"],
                    ),
                )
        return best

    def detect(self, image: np.ndarray) -> ResourceDetection:
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = self.regions.get("resources").pixels(w, h)
            crop = image[y1:y2, x1:x2]
            anchors = self.regions.anchors("resources")

            detected: dict[str, str] = {}
            boxes: dict[str, tuple[int, int, int, int]] = {}
            candidates: list[dict] = []

            for name in RESOURCE_NAMES:
                anchor = anchors.get(name)
                if anchor is None:
                    continue
                result = self._read_anchor(crop, anchor)
                if result is None:
                    continue
                detected[name] = result["value"]
                box = tuple(round(v) for v in result["box"])
                boxes[name] = box
                candidates.append({
                    "value": result["value"],
                    "x": round((box[0] + box[2]) / 2, 1),
                    "y": round((box[1] + box[3]) / 2, 1),
                    "confidence": result["confidence"],
                })

            values = dict(self._stable_values)
            for name, candidate in detected.items():
                if candidate == self._stable_values.get(name):
                    if name in boxes:
                        self._stable_boxes[name] = boxes[name]
                    self._pending_values.pop(name, None)
                    self._pending_counts.pop(name, None)
                    continue
                if candidate == self._pending_values.get(name):
                    self._pending_counts[name] = self._pending_counts.get(name, 0) + 1
                else:
                    self._pending_values[name] = candidate
                    self._pending_counts[name] = 1
                if self._pending_counts[name] >= 2:
                    self._stable_values[name] = candidate
                    if name in boxes:
                        self._stable_boxes[name] = boxes[name]
                    values[name] = candidate
                    self._pending_values.pop(name, None)
                    self._pending_counts.pop(name, None)

            return ResourceDetection(
                values,
                " ".join(x["value"] for x in candidates),
                candidates,
                True,
                None,
                self._stable_boxes,
            )

        except Exception as exc:
            return ResourceDetection({}, "", [], False, str(exc))
