from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
from rokbot.core.config import ROOT, CONFIG_DIR

@dataclass(frozen=True)
class Region:
    name: str
    x: float
    y: float
    w: float
    h: float
    def pixels(self,width:int,height:int)->tuple[int,int,int,int]:
        return (round(self.x*width),round(self.y*height),round((self.x+self.w)*width),round((self.y+self.h)*height))

class UIRegions:
    def __init__(self,config_path:str|Path|None=None)->None:
        path=Path(config_path) if config_path else CONFIG_DIR/"rok_ui.yaml"
        if not path.is_absolute(): path=ROOT/path
        data=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.config_path=path
        self.regions={name:Region(name=name,**values) for name,values in data.get("regions",{}).items()}
    def get(self,name:str)->Region: return self.regions[name]
    def boxes(self,width:int,height:int)->dict[str,tuple[int,int,int,int]]:
        return {name:r.pixels(width,height) for name,r in self.regions.items()}
