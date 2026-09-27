from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rokbot.core.adb import ADBClient


def main() -> int:
    adb = ADBClient()
    device = adb.select_first_device()
    width, height = adb.display_size()
    xml = adb.dump_ui_hierarchy()

    print(f"Device: {device}")
    print(f"Display: {width}x{height}")
    print(f"XML length: {len(xml)}")
    print()

    labels = ("ПОИСК", "SEARCH", "СБОР", "НА МАРШ", "НОВЫЕ ВОЙСКА", "Новые войска")
    found = []
    for label in labels:
        if label.lower() in xml.lower():
            found.append(label)

    print("Target text found:")
    if found:
        for label in found:
            print(f"  YES: {label}")
    else:
        print("  NONE")

    print()
    print("Relevant UI nodes:")
    for node in re.findall(r"<node\b[^>]*>", xml):
        lower = node.lower()
        if any(label.lower() in lower for label in labels):
            print(node)

    out = ROOT / "logs" / "ui_hierarchy.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print()
    print(f"Full hierarchy saved to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
