from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import yaml

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

@app.get("/api/resources/calibration")
def resource_calibration():
    image = screen.capture_cv()
    h, w = image.shape[:2]
    region = resource_detector.regions.get("resources")
    rx1, ry1, rx2, ry2 = region.pixels(w, h)
    try:
        emulator_w, emulator_h = adb.display_size()
    except Exception:
        emulator_w, emulator_h = w, h
    return {
        "width": emulator_w, "height": emulator_h,
        "capture_width": w, "capture_height": h,
        "region_x": rx1, "region_y": ry1,
        "region_y_norm": region.y,
        "region_width": rx2 - rx1, "region_height": ry2 - ry1,
        "slot_width": max(70, round((rx2 - rx1) * 0.085)) * 2,
        "anchors": resource_detector.regions.anchors("resources"),
        "values": resource_detector.detect(image).values,
        "detected_boxes": {
            name: [
                box[0] + rx1, box[1] + ry1,
                box[2] + rx1, box[3] + ry1
            ]
            for name, box in (resource_detector.detect(image).boxes or {}).items()
        },
    }

@app.get("/api/resources/calibration/defaults")
def resource_calibration_defaults():
    image = screen.capture_cv()
    h, w = image.shape[:2]
    region = resource_detector.regions.get("resources")
    rx1, ry1, rx2, ry2 = region.pixels(w, h)
    try:
        emulator_w, emulator_h = adb.display_size()
    except Exception:
        emulator_w, emulator_h = w, h
    return {
        "width": emulator_w, "height": emulator_h,
        "capture_width": w, "capture_height": h,
        "region_x": rx1, "region_y": ry1,
        "region_y_norm": region.y,
        "region_width": rx2 - rx1, "region_height": ry2 - ry1,
        "slot_width": max(70, round((rx2 - rx1) * 0.085)) * 2,
        "anchors": {"food":0.260,"wood":0.405,"stone":0.570,"gold":0.725,"gems":0.870},
        "values": resource_detector.detect(image).values,
    }

@app.post("/api/resources/calibration")
def save_resource_calibration(payload: dict):
    anchors = payload.get("anchors")
    region_y_norm = payload.get("region_y_norm", resource_detector.regions.get("resources").y)
    allowed = {"food","wood","stone","gold","gems"}
    if not isinstance(anchors, dict) or set(anchors) != allowed:
        raise HTTPException(status_code=400, detail="invalid anchors")
    clean = {}
    for name in allowed:
        try:
            value = float(anchors[name])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"invalid anchor: {name}")
        if not 0 <= value <= 1.15:
            raise HTTPException(status_code=400, detail=f"anchor out of range: {name}")
        clean[name] = round(value, 6)

    try:
        region_y_norm = float(region_y_norm)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid region_y_norm")
    if not 0 <= region_y_norm <= 0.85:
        raise HTTPException(status_code=400, detail="region_y_norm out of range")

    config_path = ROOT / "config" / "rok_ui.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    resources_config = data.setdefault("regions", {}).setdefault("resources", {})
    resources_config["anchors"] = clean
    resources_config["y"] = round(region_y_norm, 6)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    resource_detector.regions = __import__("rokbot.vision.ui_regions", fromlist=["UIRegions"]).UIRegions()
    resource_detector._stable_values.clear()
    resource_detector._pending_values.clear()
    resource_detector._pending_counts.clear()
    resource_detector._stable_boxes.clear()
    return {"ok": True, "anchors": clean, "region_y_norm": region_y_norm}

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

def _png_response(data: bytes) -> Response:
    if not data or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=500, detail=f"Invalid PNG screenshot ({len(data)} bytes)")
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Content-Length": str(len(data)),
        },
    )

@app.get("/api/screenshot")
def screenshot():
    raw = screen.cached_bytes(max_age=5.0)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=500, detail="Cached screenshot could not be decoded")
    h, w = image.shape[:2]
    resources = resource_detector.detect(image)

    region = resource_detector.regions.get("resources")
    rx1, ry1, rx2, ry2 = region.pixels(w, h)
    anchors = resource_detector.regions.anchors("resources")
    region_w = rx2 - rx1
    for name in ("food", "wood", "stone", "gold", "gems"):
        anchor = anchors.get(name)
        if anchor is None:
            continue

        center_x = rx1 + round(anchor * region_w)
        slot_half = max(70, round(region_w * 0.085))
        sx1 = max(rx1, center_x - slot_half)
        sx2 = min(rx2, center_x + slot_half)
        cv2.rectangle(image, (int(sx1), ry1), (int(sx2), ry2), (0, 140, 255), 2)

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
        raise HTTPException(status_code=500, detail="OpenCV could not encode screenshot")
    return _png_response(encoded.tobytes())

@app.get("/api/screenshot/raw")
def screenshot_raw():
    return _png_response(screen.cached_bytes(max_age=5.0))

@app.get("/api/screenshot/diagnostics")
def screenshot_diagnostics():
    raw = screen.cached_bytes(max_age=5.0)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return {
        "bytes": len(raw),
        "png_signature": raw[:8].hex(),
        "decoded": image is not None,
        "shape": list(image.shape) if image is not None else None,
    }
