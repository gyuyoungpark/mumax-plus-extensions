"""Fig 1(b) — Conventional Suhl vs. finite-momentum pairing comparison.

Two-panel schematic that contrasts the two regimes:

  Left  (Conventional Suhl):
    Uniform pump (k = 0) decays into a counter-propagating magnon pair
    with k_1 = -k_2.  Conservation: k_1 + k_2 = 0.

  Right (This work):
    Chiral SAW phonon (omega = 2 omega_K, k_SAW != 0) splits into two
    magnons that BOTH carry positive k_x. Conservation:
    k_1 + k_2 = k_SAW.  This is the central new physics of the paper.

Color scheme (Okabe-Ito): SAW phonon in BLUE (#0072B2) to match Fig 1(a);
conventional pump and ancillary text in GRAY; magnons / vertices in BLACK.

Outputs:
  figures/fig1b_splitting.{pdf,eps,svg,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# Colors
BLUE  = (0 / 255.0, 114 / 255.0, 178 / 255.0)   # #0072B2 — SAW phonon
GRAY  = (0.50, 0.50, 0.50)                      # conventional / dim
BLACK = (0.0, 0.0, 0.0)                         # magnons / vertices

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def arrow_polygon(xt, yt, angle_deg, head_len=0.42, head_w=0.32):
    """Return triangle vertices for an arrowhead with tip at (xt, yt)
    pointing along angle_deg (degrees, 0 = +x)."""
    a = angle_deg * np.pi / 180.0
    bx = xt - head_len * np.cos(a)
    by = yt - head_len * np.sin(a)
    lx = bx + (head_w / 2) * np.sin(a)
    ly = by - (head_w / 2) * np.cos(a)
    rx = bx - (head_w / 2) * np.sin(a)
    ry = by + (head_w / 2) * np.cos(a)
    return [(lx, ly), (xt, yt), (rx, ry)]


def draw_suhl(ax):
    """Left panel: conventional Suhl pumping (k_1 + k_2 = 0)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")

    vx, vy = 5.0, 3.5

    # Title
    ax.text(vx, 6.25, "Conventional Suhl",
            color=GRAY, fontsize=8.5, ha="center", va="center",
            fontfamily="sans-serif", fontstyle="italic")

    # Incoming uniform pump from above (vertical wavy line in gray)
    n_pts = 400
    ys_in = np.linspace(5.6, vy + 0.16, n_pts)
    wlen = 0.55
    amp = 0.13
    xs_in = vx + amp * np.sin(2 * np.pi * (5.6 - ys_in) / wlen)
    ax.plot(xs_in, ys_in, color=GRAY, lw=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=2)

    # Arrowhead at the vertex (pointing down)
    head_pts = [(vx - 0.22, vy + 0.46),
                (vx,         vy),
                (vx + 0.22, vy + 0.46)]
    ax.add_patch(Polygon(head_pts, closed=True,
                         facecolor=GRAY, edgecolor="none", zorder=3))

    # Pump labels
    ax.text(vx + 0.45, 4.85, r"pump",
            color=GRAY, fontsize=8.0, ha="left", va="center",
            fontfamily="sans-serif")
    ax.text(vx + 0.45, 4.40, r"$(\omega,\ \mathbf{k}=0)$",
            color=GRAY, fontsize=8.0, ha="left", va="center",
            fontfamily="sans-serif")

    # Vertex dot
    ax.add_patch(Circle((vx, vy), 0.14, facecolor=BLACK,
                        edgecolor="none", zorder=4))

    # Two outgoing magnons: opposite directions (+x and -x)
    mag_len = 2.4
    head_back = 0.42
    # Right
    rx_tip = vx + mag_len
    ax.plot([vx, rx_tip - head_back], [vy, vy],
            color=BLACK, lw=1.7, solid_capstyle="round", zorder=2)
    ax.add_patch(Polygon(arrow_polygon(rx_tip, vy, 0.0),
                         closed=True, facecolor=BLACK,
                         edgecolor="none", zorder=3))
    # Left
    lx_tip = vx - mag_len
    ax.plot([vx, lx_tip + head_back], [vy, vy],
            color=BLACK, lw=1.7, solid_capstyle="round", zorder=2)
    ax.add_patch(Polygon(arrow_polygon(lx_tip, vy, 180.0),
                         closed=True, facecolor=BLACK,
                         edgecolor="none", zorder=3))

    # Magnon labels at tips
    ax.text(rx_tip + 0.15, vy + 0.05, r"$\mathbf{k}_1$",
            color=BLACK, fontsize=9.0, ha="left", va="center",
            fontfamily="sans-serif")
    ax.text(lx_tip - 0.15, vy + 0.05, r"$\mathbf{k}_2$",
            color=BLACK, fontsize=9.0, ha="right", va="center",
            fontfamily="sans-serif")

    # Conservation rule
    ax.text(vx, 1.55, r"$\mathbf{k}_1 + \mathbf{k}_2 = 0$",
            color=GRAY, fontsize=10, ha="center", va="center",
            fontfamily="sans-serif")
    ax.text(vx, 0.75, r"zero-momentum pairs",
            color=GRAY, fontsize=7.5, ha="center", va="center",
            fontfamily="sans-serif", fontstyle="italic")


def draw_thiswork(ax):
    """Right panel: chiral SAW pumping (k_1 + k_2 = k_SAW)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")

    vx, vy = 3.6, 3.5

    # Title
    ax.text(5.0, 6.25, "This work: chiral SAW pumping",
            color=BLUE, fontsize=8.5, ha="center", va="center",
            fontfamily="sans-serif", fontstyle="italic")

    # Incoming SAW phonon (horizontal wavy line, BLUE)
    wlen = 1.20
    amp = 0.18
    phonon_x0 = 0.30
    xs = np.linspace(phonon_x0, vx, 500)
    ys = vy + amp * np.sin(2 * np.pi * (xs - phonon_x0) / wlen)
    ax.plot(xs, ys, color=BLUE, lw=2.2,
            solid_capstyle="round", solid_joinstyle="round", zorder=2)

    # Arrowhead at the vertex
    head_w_ph = 0.45
    head_h_ph = 0.32
    ax.add_patch(Polygon(
        [(vx - head_w_ph, vy + head_h_ph / 2),
         (vx,              vy),
         (vx - head_w_ph, vy - head_h_ph / 2)],
        closed=True, facecolor=BLUE, edgecolor="none", zorder=3))

    # SAW phonon labels
    ax.text(phonon_x0 + 0.05, vy + 0.62, r"SAW phonon",
            color=BLUE, fontsize=8.0, ha="left", va="bottom",
            fontfamily="sans-serif")
    ax.text(phonon_x0 + 0.05, vy - 0.65,
            r"$(2\omega_K,\ \mathbf{k}_\mathrm{SAW})$",
            color=BLUE, fontsize=8.0, ha="left", va="top",
            fontfamily="sans-serif")

    # Vertex dot
    ax.add_patch(Circle((vx, vy), 0.14, facecolor=BLACK,
                        edgecolor="none", zorder=4))

    # Two outgoing magnons: BOTH carry +x momentum (asymmetric angles)
    angles_deg = [+28.0, -10.0]
    mag_lens   = [2.7,  3.4]      # asymmetric lengths to evoke |k_1| != |k_2|
    head_back = 0.42
    rad = np.pi / 180.0

    tips = []
    for ang, mlen in zip(angles_deg, mag_lens):
        a = ang * rad
        x_tip = vx + mlen * np.cos(a)
        y_tip = vy + mlen * np.sin(a)
        x_shaft = vx + (mlen - head_back) * np.cos(a)
        y_shaft = vy + (mlen - head_back) * np.sin(a)
        ax.plot([vx, x_shaft], [vy, y_shaft],
                color=BLACK, lw=1.7,
                solid_capstyle="round", zorder=2)
        ax.add_patch(Polygon(arrow_polygon(x_tip, y_tip, ang),
                             closed=True, facecolor=BLACK,
                             edgecolor="none", zorder=3))
        tips.append((x_tip, y_tip))

    # Magnon labels
    ax.text(tips[0][0] + 0.15, tips[0][1] + 0.05, r"$\mathbf{k}_1$",
            color=BLACK, fontsize=9.0, ha="left", va="bottom",
            fontfamily="sans-serif")
    ax.text(tips[1][0] + 0.15, tips[1][1] - 0.05, r"$\mathbf{k}_2$",
            color=BLACK, fontsize=9.0, ha="left", va="top",
            fontfamily="sans-serif")

    # Conservation rule (highlighted)
    ax.text(5.0, 1.55,
            r"$\mathbf{k}_1 + \mathbf{k}_2 = \mathbf{k}_\mathrm{SAW}$",
            color=BLUE, fontsize=10, ha="center", va="center",
            fontfamily="sans-serif", fontweight="bold")
    ax.text(5.0, 0.75, r"finite-momentum pairs",
            color=BLUE, fontsize=7.5, ha="center", va="center",
            fontfamily="sans-serif", fontstyle="italic")


def main():
    fig_w_in = 6.8
    fig_h_in = 2.5
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(1, 2, wspace=0.05,
                          left=0.01, right=0.99,
                          top=0.99, bottom=0.01)
    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    draw_suhl(ax_left)
    draw_thiswork(ax_right)

    # Light dashed divider between panels
    divider = plt.Line2D([0.5, 0.5], [0.10, 0.90],
                         transform=fig.transFigure,
                         color=(0.82, 0.82, 0.82),
                         lw=0.5, linestyle='--', clip_on=False)
    fig.add_artist(divider)

    out_base = os.path.join(FIG_DIR, "fig1b_splitting")
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(out_base + "." + ext, format=ext,
                    bbox_inches="tight", pad_inches=0.02,
                    transparent=True)
    fig.savefig(out_base + ".png", format="png",
                bbox_inches="tight", pad_inches=0.02,
                dpi=600, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_base}.{{pdf,eps,svg,png}}")


if __name__ == "__main__":
    main()
