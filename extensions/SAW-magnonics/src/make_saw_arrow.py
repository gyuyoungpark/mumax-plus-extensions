"""Standalone SAW propagation arrow icon (vector format).

A sinusoidal fluctuation that ends in an arrowhead pointing in the +x
direction, drawn in the Okabe-Ito blue (RGB 0, 114, 178). The arrowhead
sits directly at the wave terminus (no straight extension).

Outputs:
  figures/fig_saw_arrow.pdf
  figures/fig_saw_arrow.eps
  figures/fig_saw_arrow.svg
  figures/fig_saw_arrow.png  (for quick preview)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Okabe-Ito blue
COLOR = (0 / 255.0, 114 / 255.0, 178 / 255.0)  # = "#0072B2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Geometry
N_PEAKS = 4         # number of sine peaks before the arrowhead
WAVE_LEN = 2.0      # wavelength
AMP = 0.4           # peak amplitude of sine
LW = 2.4            # line width
ARROW_W = 0.55      # arrowhead width along x  (forward extent from peak)
ARROW_H = 0.35      # arrowhead full height (top to bottom)

# Last sine peak occurs at x = WAVE_LEN/4 + (N_PEAKS - 1) * WAVE_LEN
x_end = WAVE_LEN / 4 + (N_PEAKS - 1) * WAVE_LEN
y_end = AMP   # at peak

# Figure size
fig_w_in = 5.0
fig_h_in = fig_w_in * (2 * AMP + 0.6) / (x_end + ARROW_W + 0.4)

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

# Sinusoidal SAW fluctuation, terminating exactly at the last peak so the
# tangent is horizontal there (slope = 0).
xs = np.linspace(0.0, x_end, 1200)
ys = AMP * np.sin(2 * np.pi * xs / WAVE_LEN)
ax.plot(xs, ys, color=COLOR, lw=LW,
        solid_capstyle="round", solid_joinstyle="round")

# Arrowhead polygon at the wave terminus, pointing in +x.
# Back edge is centered on the peak (x_end, y_end); tip extends to
# (x_end + ARROW_W, y_end).
triangle = Polygon(
    [
        (x_end, y_end + ARROW_H / 2),
        (x_end + ARROW_W, y_end),
        (x_end, y_end - ARROW_H / 2),
    ],
    closed=True,
    facecolor=COLOR,
    edgecolor="none",
)
ax.add_patch(triangle)

# Layout
ax.set_xlim(-0.2, x_end + ARROW_W + 0.3)
ax.set_ylim(-(AMP + 0.3), AMP + ARROW_H / 2 + 0.2)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save in vector formats + preview PNG
out_base = os.path.join(FIG_DIR, "fig_saw_arrow")
for ext in ("pdf", "eps", "svg"):
    fig.savefig(out_base + "." + ext, format=ext,
                bbox_inches="tight", pad_inches=0.02,
                transparent=True)
fig.savefig(out_base + ".png", format="png",
            bbox_inches="tight", pad_inches=0.02,
            dpi=600, transparent=True)

plt.close(fig)
print(f"Saved: {out_base}.{{pdf,eps,svg,png}}")
print(f"Color: RGB(0, 114, 178) = #0072B2")
print(f"Geometry: {N_PEAKS} peaks, wavelength {WAVE_LEN}, terminal arrowhead at x={x_end}")
