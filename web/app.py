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
    """Read the five resource counters from the top-right RoK resource bar.

    RoK places an icon + number in each counter. Reading the whole bar with
    psm 7 is unreliable because the icons break the text line, so we use
    Tesseract word boxes and map numeric tokens to calibrated normalized
    X positions.
    """
    height, width = image.shape[:2]

    # The calibrated resources region starts around x=.56. Extend the height
    # slightly because the resource counters sit below the very top edge.
    x1 = round(width * 0.55)
    x2 = width
    y1 = 0
    y2 = round(height * 0.13)
    crop = image[y1:y2, x1:x2]

    anchors = {
        "food": 0.694,
        "wood": 0.781,
        "stone": 0.852,
        "gold": 0.932,
        "gems": 0.988,
    }

    try:
        data = pytesseract.image_to_data(
            crop,
            config="--psm 11 -c tessedit_char_whitelist=0123456789.,KMBT",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        return {
            "values": {},
            "raw": "",
            "available": False,
            "error": str(exc),
            "candidates": [],
            "detected_count": 0,
            "expected_count": len(RESOURCE_NAMES),
        }

    candidates = []
    raw_parts = []

    for i, raw in enumerate(data.get("text", [])):
        token = raw.strip()
        if not token:
            continue

        # OCR sometimes reads K as a visually similar character or drops it.
        token = token.replace("%", "K").replace(" ", "")
        match = re.fullmatch(r"\d[\d.,]*[KMBT]?", token, flags=re.I)
        if not match:
            continue

        # Avoid tiny OCR fragments; resource counters are normally at least
        # three digits in the current account.
        digits = re.sub(r"[^0-9]", "", token)
        if len(digits) < 3:
            continue

        center_x = (
            x1 + data["left"][i] + data["width"][i] / 2
        ) / width
        center_y = (
            data["top"][i] + data["height"][i] / 2
        ) / height

        if center_y > 0.11:
            continue

        confidence = float(data["conf"][i])
        candidates.append(
            {
                "value": token,
                "x": center_x,
                "confidence": confidence,
            }
        )
        raw_parts.append(token)

    # Assign each OCR token to the nearest known resource position.
    values = {}
    used = set()
    for name, anchor in anchors.items():
        best_index = None
        best_score = None

        for i, candidate in enumerate(candidates):
            if i in used:
                continue

            distance = abs(candidate["x"] - anchor)
            if distance > 0.065:
                continue

            # Position is more important than OCR confidence because RoK's
            # resource icons can reduce Tesseract's confidence substantially.
            score = distance * 100 - max(0.0, candidate["confidence"]) * 0.02
            if best_score is None or score < best_score:
                best_score = score
                best_index = i

        if best_index is not None:
            used.add(best_index)
            values[name] = candidates[best_index]["value"]

    return {
        "values": values,
        "raw": " ".join(raw_parts),
        "available": True,
        "error": None,
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
