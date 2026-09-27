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
    """Detect resources from the calibrated HUD."""
    def __init__(self,regions:UIRegions|None=None)->None:
        self.regions=regions or UIRegions()
        self._rapidocr = None
        self._rapidocr_error: str | None = None
        self._configure_tesseract()
        self._stable_values: dict[str, str] = {}
        self._pending_values: dict[str, str] = {}
        self._pending_counts: dict[str, int] = {}
        self._stable_boxes: dict[str, tuple[int,int,int,int]] = {}
        self._startup_scan_done = False

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

    @staticmethod
    def _extract_tokens(data: dict, x1: int, scale: int) -> list[dict]:
        tokens: list[dict] = []
        for i, raw in enumerate(data.get("text", [])):
            token = ResourceDetector._token(raw)
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            digits = len(re.sub(r"[^0-9]", "", token))
            if conf < 30 and digits < 4:
                continue
            if conf < 18:
                continue

            left = float(data["left"][i]) / scale
            top = float(data["top"][i]) / scale
            right = (float(data["left"][i]) + float(data["width"][i])) / scale
            bottom = (float(data["top"][i]) + float(data["height"][i])) / scale
            tokens.append({
                "value": token,
                "confidence": round(conf, 1),
                "box": (
                    round(x1 + left), round(top),
                    round(x1 + right), round(bottom),
                ),
            })

        tokens.sort(key=lambda item: (item["box"][1], item["box"][0]))

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
            close = gap <= max(10.0, min(prev_h, item_h) * 0.9)
            prev_digits = re.sub(r"[^0-9]", "", prev["value"])
            item_digits = re.sub(r"[^0-9]", "", item["value"])
            can_merge = (
                same_line and close and prev_digits and item_digits
                and len(prev_digits) <= 3 and len(item_digits) <= 3
                and len(prev_digits) + len(item_digits) >= 4
            )
            if can_merge:
                merged[-1] = {
                    "value": prev["value"] + item["value"],
                    "confidence": round(min(prev["confidence"], item["confidence"]), 1),
                    "box": (
                        min(px1, x1i), min(py1, y1i),
                        max(px2, x2i), max(py2, y2i),
                    ),
                }
            else:
                merged.append(item)

        return merged


    def _get_rapidocr(self):
        if self._rapidocr is not None:
            return self._rapidocr
        if self._rapidocr_error is not None:
            return None
        try:
            from rapidocr import RapidOCR
            self._rapidocr = RapidOCR(
                text_score=0.20,
                min_height=10,
                width_height_ratio=-1,
                max_side_len=3000,
                min_side_len=10,
                det_thresh=0.20,
                det_box_thresh=0.20,
            )
            return self._rapidocr
        except Exception as exc:
            self._rapidocr_error = str(exc)
            return None

    @staticmethod
    def _merge_ocr_tokens(tokens: list[dict]) -> list[dict]:
        tokens.sort(key=lambda item: (item["box"][1], item["box"][0]))
        merged: list[dict] = []
        for item in tokens:
            if not merged:
                merged.append(item)
                continue
            prev = merged[-1]
            px1, py1, px2, py2 = prev["box"]
            x1i, y1i, x2i, y2i = item["box"]
            ph = max(1, py2 - py1)
            ih = max(1, y2i - y1i)
            same_line = abs(((py1 + py2) / 2) - ((y1i + y2i) / 2)) <= max(ph, ih) * 0.55
            gap = x1i - px2
            close = gap <= max(14.0, min(ph, ih) * 1.2)
            pd = re.sub(r"[^0-9]", "", prev["value"])
            idg = re.sub(r"[^0-9]", "", item["value"])
            if same_line and close and pd and idg and len(pd) <= 3 and len(idg) <= 3 and len(pd) + len(idg) >= 4:
                merged[-1] = {
                    "value": prev["value"] + item["value"],
                    "confidence": round(min(prev["confidence"], item["confidence"]), 1),
                    "box": (min(px1, x1i), min(py1, y1i), max(px2, x2i), max(py2, y2i)),
                }
            else:
                merged.append(item)
        return merged

    @staticmethod
    def _rapid_box(box) -> tuple[int,int,int,int] | None:
        try:
            points = np.asarray(box, dtype=float).reshape(-1, 2)
            if points.shape[0] < 4:
                return None
            return (
                round(float(points[:, 0].min())),
                round(float(points[:, 1].min())),
                round(float(points[:, 0].max())),
                round(float(points[:, 1].max())),
            )
        except Exception:
            return None

    def _scan_rapidocr(self, crop: np.ndarray) -> list[dict]:
        engine = self._get_rapidocr()
        if engine is None:
            return []
        try:
            result = engine(crop, use_cls=False)
            txts = getattr(result, "txts", None)
            boxes = getattr(result, "boxes", None)
            scores = getattr(result, "scores", None)
            if txts is None or boxes is None:
                return []

            candidates: list[dict] = []
            for i, raw in enumerate(txts):
                token = self._token(str(raw))
                if not token:
                    continue
                try:
                    confidence = float(scores[i])
                except (TypeError, ValueError, IndexError):
                    confidence = 0.0
                if confidence < 0.20:
                    continue
                box = self._rapid_box(boxes[i])
                if box is None:
                    continue
                candidates.append({
                    "value": token,
                    "confidence": round(confidence * 100.0, 1),
                    "box": box,
                })
            return self._merge_ocr_tokens(candidates)
        except Exception as exc:
            self._rapidocr_error = str(exc)
            return []

    def _detect_candidates(self, crop: np.ndarray) -> tuple[dict[str, dict], list[dict]]:
        # One RapidOCR pass over the complete HUD. Missing counters are
        # handled later by the existing targeted Tesseract fallback.
        candidates = self._scan_rapidocr(crop)
        if candidates:
            return self._assign_candidates(candidates, crop.shape[1]), candidates
        return self._startup_scan(crop)

    def _assign_candidates(self, candidates: list[dict], width: float) -> dict[str, dict]:
        """Assign one OCR token to each calibrated resource anchor."""
        anchors = self.regions.anchors("resources")
        assigned: dict[str, dict] = {}
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
                distance = abs((item["box"][0] + item["box"][2]) / 2 / max(width, 1.0) - anchor)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None and best_distance <= 0.12:
                assigned[name] = candidates[best_index]
                used.add(best_index)

        return assigned

    def _run_ocr(self, variant: np.ndarray, psm: int):
        try:
            return pytesseract.image_to_data(
                variant,
                config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.,KMBT",
                output_type=pytesseract.Output.DICT,
                timeout=1.0,
            )
        except RuntimeError:
            return None

    def _startup_scan(self, crop: np.ndarray) -> tuple[dict[str, dict], list[dict]]:
        """Scan the whole resource HUD once after process startup."""
        scale = 3
        scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        variants = [
            gray,
            cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)[1],
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 7
            ),
        ]

        best_candidates: list[dict] = []
        best_score = -1

        for variant in variants:
            data = self._run_ocr(variant, 7)
            if not data:
                continue
            parsed = self._extract_tokens(data, 0, scale)
            if len(parsed) > len(best_candidates):
                best_candidates = parsed
                best_score = len(parsed)
            elif len(parsed) == len(best_candidates) and parsed:
                score = sum(max(0.0, x["confidence"]) for x in parsed)
                if score > sum(max(0.0, x["confidence"]) for x in best_candidates):
                    best_candidates = parsed

        # Full-HUD fallback. This is only done once at startup.
        if len(best_candidates) < 5:
            for psm in (6, 11):
                data = self._run_ocr(gray, psm)
                if not data:
                    continue
                parsed = self._extract_tokens(data, 0, scale)
                if len(parsed) > len(best_candidates):
                    best_candidates = parsed

        assigned = self._assign_candidates(best_candidates, crop.shape[1])
        return assigned, best_candidates

    def _read_anchor(self, crop: np.ndarray, anchor: float, scale: int = 3) -> dict | None:
        h, w = crop.shape[:2]
        center = int(anchor * w)
        half = max(82, int(w * 0.105))
        x1 = max(0, center - half)
        x2 = min(w, center + half)

        slot = crop[:, x1:x2]
        slot = cv2.resize(slot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)

        variants = [
            gray,
            cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1],
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 7
            ),
        ]

        best = None

        def choose(parsed: list[dict]):
            if not parsed:
                return None
            return max(
                parsed,
                key=lambda item: (
                    len(re.sub(r"[^0-9]", "", item["value"])) >= 4,
                    len(re.sub(r"[^0-9]", "", item["value"])),
                    item["confidence"],
                ),
            )

        for variant in variants:
            data = self._run_ocr(variant, 7)
            if not data:
                continue
            parsed = self._extract_tokens(data, x1, scale)
            candidate = choose(parsed)
            if candidate is None:
                continue

            if best is None or (
                len(re.sub(r"[^0-9]", "", candidate["value"])) >= 4
                and len(re.sub(r"[^0-9]", "", best["value"])) < 4
            ) or (
                len(re.sub(r"[^0-9]", "", candidate["value"])) == len(re.sub(r"[^0-9]", "", best["value"]))
                and candidate["confidence"] > best["confidence"]
            ):
                best = candidate

            if len(re.sub(r"[^0-9]", "", candidate["value"])) >= 4 and candidate["confidence"] >= 35:
                return candidate

        for psm in (6, 11, 13):
            data = self._run_ocr(gray, psm)
            if not data:
                continue
            parsed = self._extract_tokens(data, x1, scale)
            candidate = choose(parsed)
            if candidate is None:
                continue
            if best is None or (
                len(re.sub(r"[^0-9]", "", candidate["value"])) > len(re.sub(r"[^0-9]", "", best["value"]))
            ) or (
                len(re.sub(r"[^0-9]", "", candidate["value"])) == len(re.sub(r"[^0-9]", "", best["value"]))
                and candidate["confidence"] > best["confidence"]
            ):
                best = candidate
            if len(re.sub(r"[^0-9]", "", candidate["value"])) >= 4:
                return candidate

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

            # First frame after dashboard/process startup: scan the entire
            # resource HUD together so we establish all five counters from
            # one coherent screenshot. Subsequent refreshes use the faster
            # per-resource calibrated slots.
            if not self._startup_scan_done:
                startup_assigned, startup_candidates = self._startup_scan(crop)
                self._startup_scan_done = True
                for name, result in startup_assigned.items():
                    detected[name] = result["value"]
                    boxes[name] = tuple(round(v) for v in result["box"])
                    candidates.append({
                        "value": result["value"],
                        "x": round((result["box"][0] + result["box"][2]) / 2, 1),
                        "y": round((result["box"][1] + result["box"][3]) / 2, 1),
                        "confidence": result["confidence"],
                    })

                # Seed stable values immediately from the coherent startup
                # scan. This avoids waiting two refreshes to populate the HUD.
                for name, value in detected.items():
                    self._stable_values[name] = value
                    if name in boxes:
                        self._stable_boxes[name] = boxes[name]

            # Normal fast path after startup: read each calibrated slot.
            if self._startup_scan_done and not detected:
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

            # If startup found at least one resource, also refresh missing
            # resources individually so a single weak OCR token doesn't make
            # the rest of the HUD disappear.
            if self._startup_scan_done:
                for name in RESOURCE_NAMES:
                    if name in detected:
                        continue
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
