from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import yaml
from PIL import Image, ImageTk

from rokbot.core.adb import ADBClient
from rokbot.core.screen import Screen
from rokbot.vision.ui_regions import UIRegions


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "rok_ui.yaml"

REGION_COLORS = [
    "#f5d90a",
    "#2684ff",
    "#ff8a00",
    "#e83cff",
    "#22c55e",
    "#00d4ff",
    "#ff4fa3",
    "#ff3333",
    "#9b59ff",
    "#14b8a6",
]


class UICalibrator:
    """Mouse-based editor for normalized RoK UI regions."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RoK UI Calibrator")
        self.root.geometry("1600x900")
        self.root.minsize(1100, 700)

        self.source_image = self._capture()
        self.source_h, self.source_w = self.source_image.shape[:2]

        self.regions = self._load_regions()
        self.names = list(self.regions.keys())
        self.selected_name = self.names[0] if self.names else ""

        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.tk_image: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._update_status()
        self.root.after(100, self._redraw)

    def _capture(self):
        adb = ADBClient()
        device = adb.select_first_device()
        print(f"Device: {device}")
        return Screen(adb).capture_cv()

    def _load_regions(self) -> dict[str, dict[str, float]]:
        if not CONFIG_PATH.exists():
            return {}

        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        regions = data.get("regions", {})
        return {
            name: {
                "x": float(values["x"]),
                "y": float(values["y"]),
                "w": float(values["w"]),
                "h": float(values["h"]),
            }
            for name, values in regions.items()
        }

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, background="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)

        side = ttk.Frame(main, width=300, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        ttk.Label(
            side,
            text="RoK UI CALIBRATOR",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            side,
            text="1. Выбери зону.\n"
                 "2. Зажми ЛКМ на скриншоте.\n"
                 "3. Протяни рамку.\n"
                 "4. Отпусти.\n"
                 "5. Save.",
            justify="left",
        ).pack(anchor="w", pady=(8, 14))

        self.listbox = tk.Listbox(side, height=16, exportselection=False)
        self.listbox.pack(fill="x")
        for name in self.names:
            self.listbox.insert(tk.END, name)
        if self.names:
            self.listbox.selection_set(0)
        self.listbox.bind("<<ListboxSelect>>", self._select_region)

        ttk.Button(side, text="📸 Новый скриншот", command=self._refresh).pack(
            fill="x", pady=(12, 4)
        )
        ttk.Button(side, text="🗑 Очистить выбранную", command=self._clear_selected).pack(
            fill="x", pady=4
        )
        ttk.Button(side, text="💾 SAVE → rok_ui.yaml", command=self._save).pack(
            fill="x", pady=(4, 12)
        )

        self.status = ttk.Label(side, justify="left")
        self.status.pack(anchor="w", fill="x")

        ttk.Label(
            side,
            text="Подсказка: красная пунктирная рамка — "
                 "зона, которую сейчас редактируешь.",
            wraplength=270,
            justify="left",
        ).pack(anchor="w", pady=(18, 0))

        self.canvas.focus_set()

    def _select_region(self, _event=None) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.selected_name = self.names[selection[0]]
            self._update_status()
            self._redraw()

    def _canvas_to_source(self, x: float, y: float) -> tuple[float, float]:
        sx = (x - self.offset_x) / self.scale
        sy = (y - self.offset_y) / self.scale
        sx = max(0.0, min(float(self.source_w), sx))
        sy = max(0.0, min(float(self.source_h), sy))
        return sx, sy

    def _mouse_down(self, event) -> None:
        self.drag_start = self._canvas_to_source(event.x, event.y)
        self.drag_current = self.drag_start
        self._redraw()

    def _mouse_drag(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_current = self._canvas_to_source(event.x, event.y)
        self._redraw()

    def _mouse_up(self, event) -> None:
        if self.drag_start is None:
            return

        end = self._canvas_to_source(event.x, event.y)
        start = self.drag_start
        self.drag_start = None
        self.drag_current = None

        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))

        # Ignore accidental clicks smaller than 3 source pixels.
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            self._redraw()
            return

        self.regions[self.selected_name] = {
            "x": x1 / self.source_w,
            "y": y1 / self.source_h,
            "w": (x2 - x1) / self.source_w,
            "h": (y2 - y1) / self.source_h,
        }

        self._update_status()
        self._redraw()

    def _clear_selected(self) -> None:
        if not self.selected_name:
            return
        self.regions[self.selected_name] = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._update_status()
        self._redraw()

    def _refresh(self) -> None:
        try:
            self.source_image = self._capture()
            self.source_h, self.source_w = self.source_image.shape[:2]
            self._redraw()
            self._update_status()
        except Exception as exc:
            messagebox.showerror("ADB / Screenshot error", str(exc))

    def _redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # Keep the complete emulator screenshot visible.
        self.scale = min(cw / self.source_w, ch / self.source_h)
        display_w = max(1, round(self.source_w * self.scale))
        display_h = max(1, round(self.source_h * self.scale))
        self.offset_x = (cw - display_w) / 2
        self.offset_y = (ch - display_h) / 2

        rgb = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.tk_image,
        )

        for index, name in enumerate(self.names):
            values = self.regions[name]
            x1 = self.offset_x + values["x"] * self.source_w * self.scale
            y1 = self.offset_y + values["y"] * self.source_h * self.scale
            x2 = x1 + values["w"] * self.source_w * self.scale
            y2 = y1 + values["h"] * self.source_h * self.scale

            selected = name == self.selected_name
            color = "#ff3333" if selected else REGION_COLORS[index % len(REGION_COLORS)]
            width = 4 if selected else 2
            dash = (8, 5) if selected else None

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=width,
                dash=dash,
            )
            self.canvas.create_rectangle(
                x1, y1, x1 + max(70, len(name) * 8 + 12), y1 + 22,
                fill=color,
                outline=color,
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 11,
                text=name,
                anchor="w",
                fill="black",
                font=("Segoe UI", 9, "bold"),
            )

        if self.drag_start and self.drag_current:
            x1s, y1s = self.drag_start
            x2s, y2s = self.drag_current
            x1 = self.offset_x + x1s * self.scale
            y1 = self.offset_y + y1s * self.scale
            x2 = self.offset_x + x2s * self.scale
            y2 = self.offset_y + y2s * self.scale
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="#ff3333",
                width=3,
                dash=(8, 5),
            )

    def _update_status(self) -> None:
        if not hasattr(self, "status"):
            return
        values = self.regions.get(self.selected_name)
        if not values:
            self.status.config(text="Нет зон.")
            return

        px = (
            round(values["x"] * self.source_w),
            round(values["y"] * self.source_h),
            round((values["x"] + values["w"]) * self.source_w),
            round((values["y"] + values["h"]) * self.source_h),
        )

        self.status.config(
            text=(
                f"Selected: {self.selected_name}\n"
                f"Pixels: {px[0]}, {px[1]} → {px[2]}, {px[3]}\n"
                f"Size: {px[2]-px[0]} × {px[3]-px[1]}\n\n"
                f"Normalized:\n"
                f"x={values['x']:.4f}\n"
                f"y={values['y']:.4f}\n"
                f"w={values['w']:.4f}\n"
                f"h={values['h']:.4f}"
            )
        )

    def _save(self) -> None:
        data = {
            "regions": {
                name: {
                    "x": round(values["x"], 6),
                    "y": round(values["y"], 6),
                    "w": round(values["w"], 6),
                    "h": round(values["h"], 6),
                }
                for name, values in self.regions.items()
            }
        }

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        messagebox.showinfo(
            "Saved",
            f"Сохранено:\n{CONFIG_PATH}\n\n"
            "Теперь запусти debug_ui.py для проверки.",
        )


def main() -> None:
    root = tk.Tk()
    UICalibrator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
