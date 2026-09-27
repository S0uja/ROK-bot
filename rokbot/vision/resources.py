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
        if len(re.sub(r"[^0-9]","",token))<3: return None
        return token
    def _assign_candidates(self, candidates: list[dict]) -> dict[str, str]:
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
                distance = abs((item["x"] / max(self._resource_width, 1.0)) - anchor)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            # Prevent a random OCR token from being assigned to a resource
            # when it is far away from the calibrated slot.
            if best_index is not None and best_distance <= 0.12:
                values[name] = candidates[best_index]["value"]
                used.add(best_index)

        return values

    def detect(self,image:np.ndarray)->ResourceDetection:
        try:
            h,w=image.shape[:2]
            x1,y1,x2,y2=self.regions.get("resources").pixels(w,h)
            crop=image[y1:y2,x1:x2]
            scale=3
            crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            data=pytesseract.image_to_data(gray,config="--psm 11 -c tessedit_char_whitelist=0123456789.,KMBT",output_type=pytesseract.Output.DICT)
            found=[]
            for i,raw in enumerate(data.get("text",[])):
                token=self._token(raw)
                if not token: continue
                conf=float(data["conf"][i])
                cx=(data["left"][i]+data["width"][i]/2)/scale
                cy=(data["top"][i]+data["height"][i]/2)/scale
                found.append({"value":token,"x":round(cx,1),"y":round(cy,1),"confidence":round(conf,1)})
            found.sort(key=lambda x:x["x"])
            self._resource_width = crop.shape[1] / scale
            values = self._assign_candidates(found)
            return ResourceDetection(
                values,
                " ".join(x["value"] for x in found),
                found,
                True,
                None,
            )
        except Exception as exc:
            return ResourceDetection({}, "", [], False, str(exc))
