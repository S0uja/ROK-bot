from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

@dataclass(frozen=True)
class Settings:
    adb_executable: str = "auto"
    adb_device: str = "auto"
    screenshot_format: str = "png"
    loop_interval: float = 1.0
    screenshot_dir: Path = ROOT / "screenshots"
    log_dir: Path = ROOT / "logs"

def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else CONFIG_DIR / "settings.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    adb = data.get("adb", {})
    bot = data.get("bot", {})
    screenshot_dir = Path(bot.get("screenshot_dir", "screenshots"))
    log_dir = Path(bot.get("log_dir", "logs"))
    if not screenshot_dir.is_absolute():
        screenshot_dir = ROOT / screenshot_dir
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    return Settings(
        adb_executable=str(adb.get("executable", "auto")),
        adb_device=str(adb.get("device", "auto")),
        screenshot_format=str(adb.get("screenshot_format", "png")),
        loop_interval=float(bot.get("loop_interval", 1.0)),
        screenshot_dir=screenshot_dir,
        log_dir=log_dir,
    )
