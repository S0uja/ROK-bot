from __future__ import annotations
from dataclasses import dataclass
import re
import hashlib
import shutil
from pathlib import Path
import time
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
        self._last_resource_signature: bytes | None = None

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
            # Resource coordinates are calibrated, so text detection is unnecessary.
            # RapidOCR can run recognition-only and accepts a batch of image crops.
            self._rapidocr = RapidOCR(params={
                "Global.text_score": 0.20,
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.use_rec": True,
                "Global.max_side_len": 1000,
                "Global.min_side_len": 10,
                "Global.return_word_box": False,
            })
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
        """Recognize each calibrated resource counter independently.

        Recognition-only RapidOCR expects an image containing a text line.
        The HUD gives us exact horizontal positions, so each resource gets a
        small dedicated crop. Keeping the calls independent also preserves
        the resource -> OCR result mapping and avoids relying on detector
        boxes, which are unavailable in recognition-only mode.
        """
        started = time.perf_counter()
        engine = self._get_rapidocr()
        if engine is None:
            return []

        try:
            h, w = crop.shape[:2]
            anchors = self.regions.anchors("resources")
            candidates: list[dict] = []

            half = max(95, int(w * 0.070))
            y1 = max(0, int(h * 0.08))
            y2 = min(h, int(h * 0.78))

            for name in RESOURCE_NAMES:
                anchor = anchors.get(name)
                if anchor is None:
                    continue

                center = int(anchor * w)
                x1 = max(0, center - half)
                x2 = min(w, center + half)
                slot = crop[y1:y2, x1:x2]
                if slot.size == 0:
                    continue

                # Recognition-only works on a single text-line image. A
                # modest upscale improves the tiny HUD digits without
                # bringing back full-HUD text detection.
                slot = cv2.resize(
                    slot,
                    None,
                    fx=2.5,
                    fy=2.5,
                    interpolation=cv2.INTER_CUBIC,
                )

                result = engine(
                    slot,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
                txts = getattr(result, "txts", None)
                scores = getattr(result, "scores", None)
                if not txts:
                    continue

                for index, raw in enumerate(txts):
                    token = self._token(str(raw))
                    if not token:
                        continue

                    try:
                        confidence = float(scores[index])
                    except (TypeError, ValueError, IndexError):
                        confidence = 0.0

                    if confidence < 0.20:
                        continue

                    candidates.append({
                        "value": token,
                        "confidence": round(confidence * 100.0, 1),
                        "box": (x1, y1, x2, y2),
                        "resource": name,
                    })

                    # One numeric counter per calibrated slot is enough.
                    break

            self._last_rapidocr_ms = (time.perf_counter() - started) * 1000.0
            return candidates
        except Exception as exc:
            self._last_rapidocr_ms = (time.perf_counter() - started) * 1000.0
            self._rapidocr_error = str(exc)
            return []

    def _detect_candidates(self, crop: np.ndarray) -> tuple[dict[str, dict], list[dict]]:
        candidates = self._scan_rapidocr(crop)

        # Tesseract is a startup-only recovery path. Once the first scan has
        # completed, a temporary RapidOCR miss must never re-run Tesseract.
        if candidates:
            assigned = self._assign_candidates(candidates, crop.shape[1])
            if self._startup_scan_done or len(assigned) >= len(RESOURCE_NAMES):
                return assigned, candidates

        if not self._startup_scan_done:
            startup_assigned, startup_candidates = self._startup_scan(crop)
            return startup_assigned, startup_candidates

        return self._assign_candidates(candidates, crop.shape[1]), candidates

    def _assign_candidates(self, candidates: list[dict], width: float) -> dict[str, dict]:
        """Assign one OCR token to each calibrated resource anchor."""
        anchors = self.regions.anchors("resources")
        assigned: dict[str, dict] = {}
        used: set[int] = set()

        for name in RESOURCE_NAMES:
            explicit = [
                index for index, item in enumerate(candidates)
                if item.get("resource") == name and index not in used
            ]
            if explicit:
                assigned[name] = candidates[explicit[0]]
                used.add(explicit[0])
                continue

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
        started = time.perf_counter()
        try:
            result = pytesseract.image_to_data(
                variant,
                config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.,KMBT",
                output_type=pytesseract.Output.DICT,
                timeout=1.0,
            )
            self._last_tesseract_ms += (time.perf_counter() - started) * 1000.0
            return result
        except RuntimeError:
            self._last_tesseract_ms += (time.perf_counter() - started) * 1000.0
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
        detect_started = time.perf_counter()
        self._last_tesseract_ms = 0.0
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = self.regions.get("resources").pixels(w, h)
            crop = image[y1:y2, x1:x2]

            # The resource HUD is mostly static. Hash a tiny normalized
            # grayscale version so OCR is only executed when the HUD itself
            # changes. This keeps the normal polling loop cheap while still
            # detecting a resource counter as soon as its pixels change.
            signature_image = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                (96, 8),
                interpolation=cv2.INTER_AREA,
            )
            resource_signature = hashlib.blake2b(
                signature_image.tobytes(),
                digest_size=8,
            ).digest()

            if (
                self._last_resource_signature == resource_signature
                and self._stable_values
            ):
                self._last_rapidocr_ms = 0.0
                self._last_tesseract_ms = 0.0
                self._last_detect_ms = (time.perf_counter() - detect_started) * 1000.0
                return ResourceDetection(
                    dict(self._stable_values),
                    " ".join(self._stable_values.get(name, "") for name in RESOURCE_NAMES),
                    [],
                    True,
                    None,
                    self._stable_boxes,
                )

            self._last_resource_signature = resource_signature

            detected: dict[str, str] = {}
            boxes: dict[str, tuple[int, int, int, int]] = {}
            candidates: list[dict] = []

            # One RapidOCR pass over the complete HUD.
            assigned, scanned = self._detect_candidates(crop)
            self._startup_scan_done = True

            assigned_ids = {id(result): name for name, result in assigned.items()}

            # Expose every raw OCR detection in the dashboard diagnostics.
            for result in scanned:
                item = {
                    "value": result["value"],
                    "x": round((result["box"][0] + result["box"][2]) / 2, 1),
                    "y": round((result["box"][1] + result["box"][3]) / 2, 1),
                    "confidence": result["confidence"],
                    "box": [int(v) for v in result["box"]],
                    "engine": "rapidocr",
                }
                resource = assigned_ids.get(id(result))
                if resource:
                    item["resource"] = resource
                candidates.append(item)

            for name, result in assigned.items():
                detected[name] = result["value"]
                boxes[name] = tuple(round(v) for v in result["box"])

            # Seed the first coherent full-HUD scan immediately.
            if not self._stable_values:
                for name, value in detected.items():
                    self._stable_values[name] = value
                    if name in boxes:
                        self._stable_boxes[name] = boxes[name]

            # Tesseract is used only by _startup_scan().

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

            error = None
            if self._rapidocr is None and self._rapidocr_error:
                error = f"RapidOCR unavailable: {self._rapidocr_error}"

            self._last_detect_ms = (time.perf_counter() - detect_started) * 1000.0
            return ResourceDetection(
                values,
                " ".join(x["value"] for x in candidates),
                candidates,
                True,
                error,
                self._stable_boxes,
            )

        except Exception as exc:
            return ResourceDetection({}, "", [], False, str(exc))
