from __future__ import annotations

import time
from pathlib import Path

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
                "detected_count": len(resources.values),
                "expected_count": 5,
            },
        }
    except Exception as exc:
        return {"ok": False, "timestamp": time.time(), "device": device, "error": str(exc)}

@app.get("/api/screenshot")
def screenshot():
    data = screen.capture_bytes()
    return Response(content=data, media_type="image/png", headers={"Cache-Control":"no-store"})
