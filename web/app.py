from __future__ import annotations

import time
from pathlib import Path

import cv2

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.detector import ScreenDetector
from rokbot.vision.resources import ResourceDetector
from rokbot.vision.screen_state import ScreenStateDetector
ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"

app = FastAPI(title="RoK Bot Dashboard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

adb = ADBClient()
screen = Screen(adb)
screen_detector = ScreenDetector(adb)
state_detector = ScreenStateDetector()
resource_detector = ResourceDetector()

def _device():
    try:
        return adb.select_first_device()
    except Exception:
        return None

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.get("/api/status")
def status():
    device = _device()
    try:
        image = screen.capture_cv()
        h, w = image.shape[:2]
        package = screen_detector.current_package()
        state = state_detector.detect(image)
        resources = resource_detector.detect(image)
        return {
            "ok": True,
            "timestamp": time.time(),
            "device": device,
            "package": package,
            "is_rok": screen_detector.is_rok(package),
            "resolution": {"width": w, "height": h},
            "screen": {"state": state.state.value, "confidence": round(state.confidence,3), "evidence": state.details},
            "resources": {
                "values": resources.values,
                "raw": resources.raw,
                "available": resources.available,
                "error": resources.error,
                "candidates": resources.candidates,
                "boxes": resources.boxes or {},
                "detected_count": len(resources.values),
                "expected_count": 5,
            },
        }
    except Exception as exc:
        return {"ok": False, "timestamp": time.time(), "device": device, "error": str(exc)}

@app.get("/api/screenshot")
def screenshot():
    image = screen.capture_cv()
    h, w = image.shape[:2]
    resources = resource_detector.detect(image)

    # Show exactly where the resource OCR is looking.
    region = resource_detector.regions.get("resources")
    rx1, ry1, rx2, ry2 = region.pixels(w, h)
    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
    cv2.putText(
        image, "RESOURCE OCR AREA", (rx1 + 10, max(28, ry1 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )

    anchors = resource_detector.regions.anchors("resources")
    for name in ("food", "wood", "stone", "gold", "gems"):
        anchor = anchors.get(name)
        if anchor is None:
            continue

        center_x = rx1 + round(anchor * region.pixels(w, h)[2] - region.pixels(w, h)[0])
        slot_half = max(70, round(region.pixels(w, h)[2] - region.pixels(w, h)[0]) * 0.085)
        sx1 = max(rx1, center_x - slot_half)
        sx2 = min(rx2, center_x + slot_half)
        cv2.rectangle(image, (int(sx1), ry1), (int(sx2), ry2), (255, 180, 0), 2)

        box = resources.boxes.get(name) if resources.boxes else None
        if box:
            bx1, by1, bx2, by2 = box
            bx1 += rx1
            bx2 += rx1
            by1 += ry1
            by2 += ry1
            cv2.rectangle(image, (bx1 - 5, by1 - 5), (bx2 + 5, by2 + 5), (0, 255, 0), 3)
            label = f"{name}: {resources.values.get(name, '—')}"
            cv2.putText(
                image, label, (bx1, max(24, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return Response(status_code=500)
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"Cache-Control":"no-store"},
    )

@app.get("/api/screenshot/raw")
def screenshot_raw():
    data = screen.capture_bytes()
    return Response(content=data, media_type="image/png", headers={"Cache-Control":"no-store"})
