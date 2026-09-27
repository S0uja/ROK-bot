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

class ResourceDetector:
    """Detect resources only inside the calibrated resources UI region."""
    def __init__(self,regions:UIRegions|None=None)->None:
        self.regions=regions or UIRegions()
        self._configure_tesseract()
        self._stable_values: dict[str, str] = {}
        self._pending_values: dict[str, str] = {}
        self._pending_counts: dict[str, int] = {}
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

            # Prevent a random OCR token from being assigned to a resource
            # when it is far away from the calibrated slot.
            if best_index is not None and best_distance <= 0.12:
                values[name] = candidates[best_index]["value"]
                used.add(best_index)

        return values

    def _read_anchor(self, crop: np.ndarray, anchor: float, scale: int = 4) -> dict | None:
        """OCR one resource slot around its calibrated horizontal anchor."""
        h, w = crop.shape[:2]
        center = int(anchor * w)
        half = max(70, int(w * 0.085))
        x1 = max(0, center - half)
        x2 = min(w, center + half)
        slot = crop[:, x1:x2]
        slot = cv2.resize(slot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
        variants = [gray, cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1],
                    cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1],
                    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 7)]
        best = None
        for variant in variants:
            data = pytesseract.image_to_data(
                variant, config="--psm 7 -c tessedit_char_whitelist=0123456789.,KMBT",
                output_type=pytesseract.Output.DICT)
            for i, raw in enumerate(data.get("text", [])):
                token = self._token(raw)
                if not token:
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    conf = -1.0
                candidate = {"value": token, "confidence": round(conf, 1)}
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
        return best

    def detect(self,image:np.ndarray)->ResourceDetection:
        try:
            h,w=image.shape[:2]
            x1,y1,x2,y2=self.regions.get("resources").pixels(w,h)
            crop=image[y1:y2,x1:x2]
            scale=3
            crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            variants = [
                ("gray", gray, "--psm 11"),
                ("line", gray, "--psm 7"),
                ("threshold", cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1], "--psm 7"),
                ("adaptive", cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 8
                ), "--psm 7"),
            ]

            found=[]
            for _, variant, psm in variants:
                data=pytesseract.image_to_data(
                    variant,
                    config=f"{psm} -c tessedit_char_whitelist=0123456789.,KMBT",
                    output_type=pytesseract.Output.DICT,
                )
                for i,raw in enumerate(data.get("text",[])):
                    token=self._token(raw)
                    if not token: continue
                    try:
                        conf=float(data["conf"][i])
                    except (TypeError, ValueError):
                        conf=-1.0
                    cx=(data["left"][i]+data["width"][i]/2)/scale
                    cy=(data["top"][i]+data["height"][i]/2)/scale
                    found.append({
                        "value": token,
                        "x": round(cx,1),
                        "y": round(cy,1),
                        "confidence": round(conf,1),
                    })

            # Keep the strongest OCR result for approximately the same position.
            deduped=[]
            for item in sorted(found, key=lambda x:(x["x"], -x["confidence"])):
                duplicate=False
                for existing in deduped:
                    if (
                        abs(item["x"] - existing["x"]) <= 18
                        and abs(item["y"] - existing["y"]) <= 12
                    ):
                        duplicate=True
                        if item["confidence"] > existing["confidence"]:
                            existing.update(item)
                        break
                if not duplicate:
                    deduped.append(item)
            found=sorted(deduped, key=lambda x:x["x"])

            # RoK may render large values with a visual space, e.g. "88 058"
            # or "4 329". Merge nearby numeric OCR fragments on the same row.
            merged=[]
            for item in found:
                if merged:
                    prev=merged[-1]
                    gap=item["x"] - prev["x"]
                    same_row=abs(item["y"] - prev["y"]) <= 18
                    if same_row and gap <= 55 and len(prev["value"]) <= 3:
                        prev["value"] += item["value"]
                        prev["x"] = round((prev["x"] + item["x"]) / 2, 1)
                        prev["confidence"] = min(prev["confidence"], item["confidence"])
                        continue
                merged.append(item)
            found=merged
            detected = {}
            anchors = self.regions.anchors("resources")
            for name in RESOURCE_NAMES:
                anchor = anchors.get(name)
                if anchor is None:
                    continue
                result = self._read_anchor(crop, anchor)
                if result is not None:
                    detected[name] = result["value"]

            # OCR can fluctuate from frame to frame. Require two consecutive
            # observations before accepting a changed value, while retaining
            # the last good value during a transient OCR miss.
            values = dict(self._stable_values)
            for name, candidate in detected.items():
                if candidate == self._stable_values.get(name):
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
                    values[name] = candidate
                    self._pending_values.pop(name, None)
                    self._pending_counts.pop(name, None)

            return ResourceDetection(
                values,
                " ".join(x["value"] for x in found),
                found,
                True,
                None,
            )
        except Exception as exc:
            return ResourceDetection({}, "", [], False, str(exc))
