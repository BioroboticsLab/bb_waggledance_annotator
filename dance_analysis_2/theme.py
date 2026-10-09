# theme.py
#
# Shared Blender-inspired dark color palette for the annotation and
# splitting tools, so both stay visually consistent and in sync.

BG = "#2b2b2b"         # main window/panel background
BG_PANEL = "#1b1b1b"   # darker recessed panel (timeline strips)
FG = "#e5e5e5"         # primary text
FG_MUTED = "#9a9a9a"   # secondary/muted text (status/hint labels)

ACCENT = "#ff8c1a"        # primary accent (orange) - active state, primary actions
ACCENT_BLUE = "#4a90d9"   # secondary accent (blue) - pairs with orange
ACCENT_RED = "#e74c3c"    # warning/alert, used sparingly

PLAYHEAD = "#f2f2f2"      # near-white, for the current-position line on timelines


def hex_to_bgr(hex_color: str):
    """Converts a '#rrggbb' string to a (B, G, R) tuple for OpenCV drawing."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


ACCENT_BGR = hex_to_bgr(ACCENT)
ACCENT_BLUE_BGR = hex_to_bgr(ACCENT_BLUE)
ACCENT_RED_BGR = hex_to_bgr(ACCENT_RED)
