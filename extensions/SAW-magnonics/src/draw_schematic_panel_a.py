"""Draft PRL Fig. 1(a): longitudinal Rayleigh-SAW geometry."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

try:
    from plot_style import apply_style, BLUE, VERMILION, BLACK
except Exception:
    BLUE = "#0072B2"
    VERMILION = "#D55E00"
    BLACK = "#000000"

    def apply_style():
        plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "paper", "prl")
os.makedirs(OUT_DIR, exist_ok=True)


def arrow(ax, xy0, xy1, color=BLACK, lw=1.2, ms=10, **kwargs):
    patch = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle="-|>",
        color=color,
        lw=lw,
        mutation_scale=ms,
        shrinkA=0,
        shrinkB=0,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def curved_arrow(ax, center, width, height, theta1, theta2, color, lw=1.2):
    ax.add_patch(Arc(center, width, height, angle=0, theta1=theta1,
                     theta2=theta2, color=color, lw=lw))
    t = np.deg2rad(theta2)
    x = center[0] + 0.5 * width * np.cos(t)
    y = center[1] + 0.5 * height * np.sin(t)
    tangent = np.array([-np.sin(t), np.cos(t)])
    start = np.array([x, y]) - 0.055 * tangent
    end = np.array([x, y])
    arrow(ax, start, end, color=color, lw=lw, ms=8)


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Substrate and ferromagnetic film.
    ax.add_patch(Rectangle((0.08, 0.13), 0.84, 0.28, facecolor="#F2F2F2",
                           edgecolor=BLACK, lw=0.8))
    ax.add_patch(Rectangle((0.08, 0.40), 0.84, 0.055, facecolor="#D9D9D9",
                           edgecolor=BLACK, lw=0.8))
    ax.text(0.88, 0.425, "FM", fontsize=8, ha="center", va="center")

    # Light hatching to communicate the elastic substrate without clutter.
    for x0 in np.linspace(0.10, 0.86, 9):
        ax.plot([x0, x0 + 0.12], [0.13, 0.41], color="#D0D0D0",
                lw=0.55, solid_capstyle="butt", zorder=0)

    # Rayleigh surface wave.
    xs = np.linspace(0.12, 0.88, 300)
    ys = 0.65 + 0.035 * np.sin(2 * np.pi * 3.2 * (xs - xs.min()) / (xs.max() - xs.min()))
    ax.plot(xs, ys, color=BLUE, lw=1.5)
    ax.text(0.13, 0.71, "Rayleigh SAW", color=BLUE, fontsize=8,
            ha="left", va="bottom")

    # Propagation and longitudinal geometry arrows.
    arrow(ax, (0.25, 0.85), (0.82, 0.85), color=BLACK, lw=1.4, ms=11)
    ax.text(0.54, 0.88, r"$\mathbf{k}_\mathrm{SAW}$", fontsize=9,
            ha="center", va="bottom")
    arrow(ax, (0.18, 0.465), (0.74, 0.465), color=BLACK, lw=1.3, ms=10)
    ax.text(0.47, 0.49, r"$\mathbf{m}_0 \parallel \mathbf{B}_0$",
            fontsize=8, ha="center", va="bottom")

    # Longitudinal strain markers in the film/substrate.
    for x in np.linspace(0.20, 0.78, 5):
        arrow(ax, (x - 0.055, 0.315), (x + 0.035, 0.315),
              color=VERMILION, lw=1.4, ms=7)
        arrow(ax, (x + 0.055, 0.265), (x - 0.035, 0.265),
              color=VERMILION, lw=1.4, ms=7)
    ax.text(0.80, 0.335, r"$\varepsilon_{xx}$", color=VERMILION,
            fontsize=9, ha="left", va="center")

    # Enlarged Rayleigh particle ellipse and lattice rotation.
    ax.add_patch(Ellipse((0.28, 0.58), 0.11, 0.075, facecolor="none",
                         edgecolor=BLUE, lw=1.0, alpha=0.8))
    curved_arrow(ax, (0.28, 0.58), 0.13, 0.09, 35, 305, BLUE, lw=1.2)
    ax.text(0.18, 0.58, r"$\Omega_y$", color=BLUE, fontsize=9,
            ha="right", va="center")

    # Coordinate axes, deliberately small.
    arrow(ax, (0.08, 0.05), (0.16, 0.05), color=BLACK, lw=0.8, ms=7)
    arrow(ax, (0.08, 0.05), (0.08, 0.13), color=BLACK, lw=0.8, ms=7)
    ax.text(0.17, 0.05, r"$x$", fontsize=7, ha="left", va="center")
    ax.text(0.08, 0.145, r"$z$", fontsize=7, ha="center", va="bottom")

    ax.text(0.50, 0.02, "longitudinal geometry",
            fontsize=8, ha="center", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig1a_geometry_draft.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.025,
                    facecolor="white", transparent=False)
    plt.close(fig)
    print(os.path.join(OUT_DIR, "fig1a_geometry_draft.png"))


if __name__ == "__main__":
    main()
