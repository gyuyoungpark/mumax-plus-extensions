"""Detection-protocol schematic figure (Fig.~7 of the PRApplied draft).

Minimal in-figure annotations: the explanatory captions for the four
predicted S(k, f_K) outputs are kept in the LaTeX caption, not on the
figure itself.

Panels:
  (a) Cross-section of the ferromagnet/LiNbO3 device with opposing
      IDT_+/IDT_- transducers and a micro-BLS objective above the strip.
  (b1..b4) Predicted S(k, f_K) cartoons for the four protocol controls.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prapplied")
OUT_NAME = "fig7_protocol.pdf"

# Helvetica typography (matches Figs. 4-6)
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["axes.linewidth"] = 0.6

# Palette
ORANGE = "#D55E00"
BLUE = "#0072B2"
GREY = "#666666"
LiNbO_color = "#FFD580"
FM_color = "#9E9E9E"
GREEN = "#009E73"


def gauss(x, mu, sig, A=1.0):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def main():
    fig = plt.figure(figsize=(7.2, 3.0))

    # ===== Panel (a): device cross-section =====
    ax = fig.add_axes([0.04, 0.10, 0.40, 0.82])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    sub = patches.Rectangle((0.4, 0.4), 9.2, 1.6,
                            facecolor=LiNbO_color, edgecolor="k", lw=0.8)
    ax.add_patch(sub)
    ax.text(5.0, 1.2, r"LiNbO$_3$", ha="center", va="center", fontsize=9)

    fm = patches.Rectangle((2.5, 2.0), 5.0, 0.45,
                           facecolor=FM_color, edgecolor="k", lw=0.8)
    ax.add_patch(fm)
    ax.text(5.0, 2.78, "FM film", ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(5.6, 2.22), xytext=(4.4, 2.22),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.text(5.0, 2.46, r"$\mathbf{m}_0$", ha="center", va="bottom",
            fontsize=8)

    # IDT+
    for x0 in np.linspace(0.65, 1.85, 5):
        ax.plot([x0, x0], [2.0, 2.35], color=BLUE, lw=1.5)
    ax.text(1.25, 1.6, r"IDT$_+$", ha="center", color=BLUE, fontsize=9,
            fontweight="bold")

    # IDT-
    for x0 in np.linspace(8.15, 9.35, 5):
        ax.plot([x0, x0], [2.0, 2.35], color=ORANGE, lw=1.5)
    ax.text(8.75, 1.6, r"IDT$_-$", ha="center", color=ORANGE, fontsize=9,
            fontweight="bold")

    # SAW wave
    xs = np.linspace(2.0, 8.0, 400)
    saw = 0.10 * np.sin(2 * np.pi * (xs - 2.0) / 0.6)
    ax.plot(xs, 0.9 + saw, color=GREY, lw=0.7, alpha=0.6)
    ax.annotate("", xy=(7.6, 0.9), xytext=(2.4, 0.9),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))

    # micro-BLS objective
    obj_x, obj_y = 5.0, 4.2
    cone = patches.Polygon(
        [[obj_x - 0.55, obj_y + 0.4], [obj_x + 0.55, obj_y + 0.4],
         [obj_x + 0.18, obj_y - 0.4], [obj_x - 0.18, obj_y - 0.4]],
        facecolor="white", edgecolor="k", lw=0.9,
    )
    ax.add_patch(cone)
    ax.text(obj_x, obj_y + 0.85, r"$\mu$-BLS",
            ha="center", va="center", fontsize=9)
    ax.plot([obj_x - 0.05, obj_x - 0.05], [obj_y - 0.4, 2.45],
            color=GREEN, lw=0.8, alpha=0.7)
    ax.plot([obj_x + 0.05, obj_x + 0.05], [obj_y - 0.4, 2.45],
            color=GREEN, lw=0.8, alpha=0.7)

    # B0 indicator
    ax.annotate("", xy=(0.6, 5.2), xytext=(0.6, 4.6),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.text(0.85, 4.9, r"$\mathbf{B}_0$", fontsize=9, va="center")

    ax.text(0.0, 5.7, "(a)", fontsize=11, fontweight="bold",
            ha="left", va="top")

    # ===== Panels (b1)-(b4): predicted S(k, f_K) =====
    k = np.linspace(-2.5, 2.5, 401)
    sig_band = 0.35
    sig_single = 0.18

    panels = [
        ("(b1)", ORANGE, [+1.0]),
        ("(b2)", BLUE, [-1.0]),
        ("(b3)", GREY, [+1.0, -1.0]),
        ("(b4)", "#444444", [0.0]),
    ]

    left0, bottom0 = 0.50, 0.10
    width, height = 0.22, 0.34
    hgap, vgap = 0.03, 0.18
    positions = [
        (left0,                bottom0 + height + vgap),
        (left0 + width + hgap, bottom0 + height + vgap),
        (left0,                bottom0),
        (left0 + width + hgap, bottom0),
    ]

    for (tag, color, mus), (lx, ly) in zip(panels, positions):
        a = fig.add_axes([lx, ly, width, height])
        if len(mus) == 1 and mus[0] == 0.0:
            y = gauss(k, 0.0, sig_single, A=1.0)
        else:
            y = np.zeros_like(k)
            for mu in mus:
                y += gauss(k, mu, sig_band, A=1.0)
            y /= y.max()
        a.fill_between(k, 0, y, color=color, alpha=0.45)
        a.plot(k, y, color=color, lw=1.4)
        a.axvline(+1.0, color=ORANGE, ls="--", lw=0.5, alpha=0.45)
        a.axvline(-1.0, color=BLUE,   ls="--", lw=0.5, alpha=0.45)
        a.axvline( 0.0, color="k",    ls=":",  lw=0.5, alpha=0.45)
        a.set_xlim(-2.5, 2.5)
        a.set_ylim(0, 1.12)
        a.set_yticks([])
        a.set_xticks([-1, 0, 1])
        a.set_xticklabels([r"$-k_\mathrm{SAW}/2$", "$0$",
                           r"$+k_\mathrm{SAW}/2$"], fontsize=7)
        a.tick_params(axis="x", labelsize=7, pad=2)
        # Just the (bi) tag in the corner, no descriptive title
        a.text(0.04, 0.92, tag, transform=a.transAxes,
               ha="left", va="top", fontsize=10, fontweight="bold",
               color="#222222")
        if ly == bottom0:
            a.set_xlabel(r"$k$  at  $f=f_K$", fontsize=8)

    fig.savefig(os.path.join(FIGS, OUT_NAME), bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, OUT_NAME), bbox_inches="tight")
    print(f"saved {OUT_NAME}")


if __name__ == "__main__":
    main()
