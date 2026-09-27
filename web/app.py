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


def _device():
    try:
        return adb.select_first_device()
    except Exception:
        return None


def _resource_ocr(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    crop = image[0:int(h * 0.075), int(w * 0.55):w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    try:
        text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=0123456789.,KMBT+",
        ).strip()
    except Exception as exc:
        # OCR is optional. A missing Tesseract executable must not make the
        # whole dashboard appear offline.
        return {
            "values": {},
            "raw": "",
            "available": False,
            "error": str(exc),
        }

    values = re.findall(r"\d[\d,.]*\s*[KMBT]?", text, flags=re.I)
    parsed = {}
    for i, value in enumerate(values[:5]):
        parsed[RESOURCE_NAMES[i]] = value.replace(" ", "")

    return {
        "values": parsed,
        "raw": text,
        "available": True,
        "error": None,
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
