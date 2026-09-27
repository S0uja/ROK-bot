from __future__ import annotations
from pathlib import Path
import yaml
from rokbot.core.config import ROOT, CONFIG_DIR

class ROKControls:
    def __init__(self,config_path:str|Path|None=None)->None:
        path=Path(config_path) if config_path else CONFIG_DIR/"rok_controls.yaml"
        if not path.is_absolute(): path=ROOT/path
        data=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.controls={}
        for group,items in data.get("controls",{}).items():
            for name,point in items.items(): self.controls[f"{group}.{name}"]=point
    def point(self,name,width,height):
        p=self.controls[name]; return round(p["x"]*width),round(p["y"]*height)
    def names(self): return sorted(self.controls)
