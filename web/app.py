from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.screen_state import ScreenStateDetector
from rokbot.vision.ui_regions import UIRegions

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"

app = FastAPI(title="RoK Bot Dashboard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

adb = ADBClient()
screen = Screen(adb)
regions = UIRegions()

RESOURCE_NAMES = ["food", "wood", "stone", "gold", "gems"]

def _configure_tesseract() -> str | None:
    """Find Tesseract on Windows without requiring it to be on PATH."""
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for path in candidates:
        if path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return str(path)
    try:
        import shutil
        found = shutil.which("tesseract")
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            return found
    except Exception:
        pass
    return None


TESSERACT_PATH = _configure_tesseract()



def _device():
    try:
        return adb.select_first_device()
    except Exception:
        return None


def _resource_ocr(image: np.ndarray) -> dict:
    """Read each RoK resource counter from its own small crop."""
    height, width = image.shape[:2]

    # Centers measured from the actual 3440x1440 RoK layout.
    anchors = {
        "food": 0.660,
        "wood": 0.736,
        "stone": 0.831,
        "gold": 0.894,
        "gems": 0.962,
    }

    values = {}
    candidates = []
    raw_parts = []

    try:
        for name, center in anchors.items():
            # Keep the crop tight so neighbouring counters cannot steal a value.
            x1 = max(0, round(width * (center - 0.040)))
            x2 = min(width, round(width * (center + 0.040)))
            y1 = 0
            y2 = round(height * 0.085)
            crop = image[y1:y2, x1:x2]

            # Upscale improves recognition of the small resource text.
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            data = pytesseract.image_to_data(
                gray,
                config="--psm 7 -c tessedit_char_whitelist=0123456789.,KMBT",
                output_type=pytesseract.Output.DICT,
            )

            found = []
            for i, raw in enumerate(data.get("text", [])):
                token = raw.strip().replace(" ", "").replace("%", "K")
                token = re.sub(r"[^0-9.,KMBTkmbt]", "", token)
                if not re.fullmatch(r"\d[\d.,]*[KMBT]?", token, flags=re.I):
                    continue
                if len(re.sub(r"[^0-9]", "", token)) < 3:
                    continue
                conf = float(data["conf"][i])
                found.append((conf, token))

            if found:
                found.sort(key=lambda item: item[0], reverse=True)
                token = found[0][1]
                values[name] = token
                candidates.append({"resource": name, "value": token, "confidence": found[0][0]})
                raw_parts.append(token)

        return {
            "values": values,
            "raw": " ".join(raw_parts),
            "available": True,
            "error": None,
            "candidates": candidates,
            "detected_count": len(values),
            "expected_count": len(RESOURCE_NAMES),
        }
    except Exception as exc:
        return {
            "values": values,
            "raw": " ".join(raw_parts),
            "available": False,
            "error": str(exc),
            "candidates": candidates,
            "detected_count": len(values),
            "expected_count": len(RESOURCE_NAMES),
        }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    device = _device()
    try:
        image = screen.capture_cv()
        h, w = image.shape[:2]
        state = ScreenStateDetector().detect(image)
        ocr = _resource_ocr(image)
        return {
            "ok": True,
            "timestamp": time.time(),
            "device": device,
            "package": "com.lilithgame.roc.gp",
            "resolution": {"width": w, "height": h},
            "screen": {
                "state": state.state.value,
                "confidence": round(state.confidence, 3),
                "evidence": state.details,
            },
            "resources": ocr,
        }
    except Exception as exc:
        return {
            "ok": False,
            "timestamp": time.time(),
            "device": device,
            "error": str(exc),
        }


@app.get("/api/screenshot")
def screenshot():
    data = screen.capture_bytes()
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
