# timeline_widgets.py
#
# Blender-dopesheet-style timeline drawing helpers, shared by the annotator
# and the video splitter so both timelines look and behave the same way.

import tkinter as tk

from .theme import BG_PANEL, FG, FG_MUTED, ACCENT, PLAYHEAD

MARKER_SIZE = 5
RULER_HEIGHT = 16
LABEL_HEIGHT = 12  # space reserved above the playhead for its frame number


class BlenderSlider:
    """
    A horizontal filled-bar slider in the style of Blender's: the current
    value is shown as a proportionally-filled accent bar with the label and
    value overlaid as text, rather than a native OS groove-and-handle.

    Drop-in compatible with the handful of tk.Scale methods this codebase
    actually uses (get/set/pack/winfo_height), so it can swap in without
    touching call sites elsewhere.
    """

    def __init__(self, parent, from_, to, label, length=300, height=24, initial=None):
        self.from_ = from_
        self.to = to
        self.label = label
        self._value = initial if initial is not None else from_
        self.canvas = tk.Canvas(
            parent, width=length, height=height, bg=BG_PANEL, highlightthickness=0
        )
        self.canvas.bind("<Button-1>", self._on_seek)
        self.canvas.bind("<B1-Motion>", self._on_seek)
        self.canvas.bind("<Configure>", lambda event: self._redraw())
        self._redraw()

    def _on_seek(self, event):
        # Button-1 bit in the event state mask; guards against any stray
        # <B1-Motion> delivered without an actual held button.
        if event.type == tk.EventType.Motion and not (event.state & 0x100):
            return
        w = self.canvas.winfo_width() or 1
        frac = min(max(event.x / w, 0.0), 1.0)
        self.set(self.from_ + frac * (self.to - self.from_))

    def _redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or int(self.canvas["width"])
        h = self.canvas.winfo_height() or int(self.canvas["height"])
        span = max(self.to - self.from_, 1e-9)
        frac = min(max((self._value - self.from_) / span, 0.0), 1.0)
        fill_x = int(frac * w)
        self.canvas.create_rectangle(0, 0, w, h, fill=BG_PANEL, outline="")
        if fill_x > 0:
            self.canvas.create_rectangle(0, 0, fill_x, h, fill=ACCENT, outline="")
        text_color = "#1a1a1a" if frac > 0.12 else FG
        self.canvas.create_text(
            6, h / 2, text=f"{self.label}: {int(self._value)}",
            fill=text_color, anchor="w", font=("TkDefaultFont", 9, "bold"),
        )

    def get(self):
        return self._value

    def set(self, value):
        self._value = min(max(round(value), self.from_), self.to)
        self._redraw()

    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)

    def winfo_height(self):
        return self.canvas.winfo_height()


def draw_diamond_marker(canvas, x, y, fill, size=MARKER_SIZE, outline="#1a1a1a"):
    """Draws a Blender-keyframe-style diamond centered at (x, y)."""
    canvas.create_polygon(
        x, y - size,
        x + size, y,
        x, y + size,
        x - size, y,
        fill=fill, outline=outline, width=1,
    )


def draw_playhead(canvas, x, top, bottom, frame=None, color=PLAYHEAD, flag_size=5):
    """
    Draws a vertical playhead line with a small triangular flag at top, and
    (if frame is given) its frame number just above the flag - space for it
    must already be reserved by starting `top` at least LABEL_HEIGHT down
    from the canvas's own top edge.
    """
    canvas.create_line(x, top, x, bottom, fill=color, width=1)
    canvas.create_polygon(
        x - flag_size, top,
        x + flag_size, top,
        x, top + flag_size * 1.6,
        fill=color, outline="",
    )
    if frame is not None:
        canvas.create_text(
            x, max(top - 2, 2), text=str(frame), fill=color,
            anchor="s", font=("TkDefaultFont", 8, "bold"),
        )


def _nice_interval(total_frames: int, target_ticks: int) -> int:
    """Rounds a raw tick spacing up to a clean 1/2/5 x 10^n step."""
    raw = max(total_frames // max(target_ticks, 1), 1)
    magnitude = 10 ** max(len(str(raw)) - 1, 0)
    for base in (1, 2, 5, 10):
        interval = base * magnitude
        if interval >= raw:
            return interval
    return raw


def draw_frame_ruler(canvas, canvas_width, canvas_height, total_frames, frame_to_x):
    """Draws tick marks + frame-number labels along the bottom of the canvas."""
    if total_frames <= 0:
        return
    target_ticks = max(canvas_width // 90, 2)
    interval = _nice_interval(total_frames, target_ticks)
    ruler_top = canvas_height - RULER_HEIGHT
    canvas.create_line(0, ruler_top, canvas_width, ruler_top, fill="#3a3a3a", width=1)

    frame = 0
    while frame <= total_frames:
        x = frame_to_x(frame, canvas_width)
        canvas.create_line(x, ruler_top, x, ruler_top + 4, fill=FG_MUTED, width=1)
        canvas.create_text(
            x, canvas_height, text=str(frame), fill=FG_MUTED,
            anchor="s", font=("TkDefaultFont", 7),
        )
        frame += interval
