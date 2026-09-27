from __future__ import annotations

import re
from pathlib import Path

from rokbot.core.config import ROOT, load_settings
from rokbot.vision.controls import ROKControls
from rokbot.vision.ui_regions import UIRegions


PY_ROOT = ROOT / "rokbot"
SCRIPTS = ROOT / "scripts"


def check(label: str, condition: bool, detail: str) -> bool:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return condition


def main() -> int:
    failures = 0

    settings = load_settings()
    failures += not check(
        "settings",
        (ROOT / "config" / "settings.yaml").exists(),
        str(ROOT / "config" / "settings.yaml"),
    )
    failures += not check(
        "UI regions",
        (ROOT / "config" / "rok_ui.yaml").exists() and bool(UIRegions().regions),
        "rok_ui.yaml loads",
    )
    failures += not check(
        "controls",
        (ROOT / "config" / "rok_controls.yaml").exists() and bool(ROKControls().names()),
        "rok_controls.yaml loads",
    )
    failures += not check(
        "configured screenshot directory",
        settings.screenshot_dir.is_absolute(),
        str(settings.screenshot_dir),
    )

    py_files = list(PY_ROOT.rglob("*.py")) + list(SCRIPTS.rglob("*.py"))
    suspicious = []

    patterns = [
        (re.compile(r'com\.lilithgame\.roc\.gp'), "hardcoded RoK package"),
        (re.compile(r'Path\([^\n]*["\']screenshots'), "hardcoded screenshots path"),
        (re.compile(r'["\']screenshots/'), "hardcoded screenshots path"),
    ]

    for path_obj in py_files:
        try:
            source = path_obj.read_text(encoding="utf-8")
        except OSError:
            continue
        for pattern, reason in patterns:
            if pattern.search(source):
                suspicious.append(f"{path_obj.relative_to(ROOT)}: {reason}")

    failures += not check(
        "no duplicated runtime package/path constants",
        not suspicious,
        "; ".join(suspicious) if suspicious else "no suspicious matches",
    )

    resources = (PY_ROOT / "vision" / "resources.py").read_text(encoding="utf-8")
    failures += not check(
        "resource OCR uses calibrated region",
        'get("resources")' in resources and "UIRegions" in resources,
        "ResourceDetector reads the configured resources region",
    )
    failures += not check(
        "resource OCR has no absolute screen anchors",
        not any(token in resources for token in ('0.660', '0.736', '0.831', '0.894', '0.962')),
        "no legacy resource anchor constants",
    )

    print()
    print("Validation complete.")
    print("Result:", "PASS" if failures == 0 else f"FAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
