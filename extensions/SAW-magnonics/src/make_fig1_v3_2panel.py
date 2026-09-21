"""PRL Figure 1, version 3: two-panel concept-stamp schematic.

Design: maximum readability with minimum visual noise.

(a) Torque selection rule (real-space): a single horizontal m0 arrow
    with two effective fields drawn at the same origin: a large MEL
    field parallel to m0 (zero linear torque, marked with X and a
    Mathieu loop icon) and a small MR field perpendicular to m0 (sole
    linear driver, marked with checkmark). A small inset under the
    main diagram shows the SAW + substrate + FM cross-section so the
    reader knows where the fields come from.

(b) Frequency-domain channel map: two Lorentzian peaks at omega_K
    (MR direct, 1:1) and 2 omega_K (MEL parametric, 2:1). Both feed
    magnons at omega_K via colored arrows; the parametric channel
    carries a finite-momentum pair rule k1+k2 = k_SAW (highlighted)
    versus the Suhl k1+k2 = 0 (greyed comparison).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from plot_style import (apply_style, SINGLE_COL, DOUBLE_COL, CM_TO_INCH,
                        SKY_BLUE, VERMILION, BLACK)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
PAPER_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "prl")
os.makedirs(FIG_DIR, exist_ok=True)

GREY_LINE = '#5a5a5a'
GREY_FILL = '#ececec'
FILM_FILL = '#e8eef7'
GREY_TXT = '#707070'


# -----------------------------------------------------------------------------
def panel_a(ax):
    """Torque selection rule with small geometry inset."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # =========================================================
    # Main diagram (top 70% of panel): torque algebra at theta=0
    # =========================================================
    cx, cy = 0.50, 0.62  # origin of arrows

    # m0 (long black horizontal arrow)
    ax.annotate('', xy=(cx + 0.34, cy), xytext=(cx - 0.34, cy),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color=BLACK,
                                shrinkA=0, shrinkB=0))
    ax.text(cx + 0.36, cy, r'$\mathbf{m}_0$',
            ha='left', va='center', fontsize=10, color=BLACK)

    # MEL field (BIG red arrow, parallel to m0, drawn slightly above)
    mel_y = cy + 0.07
    ax.annotate('', xy=(cx + 0.30, mel_y), xytext=(cx - 0.30, mel_y),
                arrowprops=dict(arrowstyle='-|>', lw=2.8,
                                color=VERMILION, shrinkA=0, shrinkB=0))
    ax.text(cx - 0.32, mel_y + 0.005, r'$\mathbf{H}_\mathrm{mel}$',
            ha='right', va='center', fontsize=10, color=VERMILION,
            fontweight='bold')

    # "X" marker (no linear torque)
    ax.text(cx + 0.36, mel_y, r'$\boldsymbol{\times}$ no torque',
            ha='left', va='center', fontsize=8, color=VERMILION,
            style='italic')

    # MR field (small blue, perpendicular up from m0 origin)
    mr_x = cx
    ax.annotate('', xy=(mr_x, cy + 0.20), xytext=(mr_x, cy - 0.005),
                arrowprops=dict(arrowstyle='-|>', lw=1.2,
                                color=SKY_BLUE, shrinkA=0, shrinkB=0))
    ax.text(mr_x + 0.015, cy + 0.13, r'$\mathbf{H}_\mathrm{mr}$',
            ha='left', va='center', fontsize=10, color=SKY_BLUE,
            fontweight='bold')
    ax.text(mr_x + 0.015, cy + 0.21,
            r'$\checkmark$ linear torque',
            ha='left', va='bottom', fontsize=8, color=SKY_BLUE,
            style='italic')

    # Mathieu loop icon to the right (parametric pump indicator)
    loop_cx, loop_cy = cx + 0.30, mel_y + 0.10
    ax.add_patch(Arc((loop_cx, loop_cy), 0.10, 0.10,
                     theta1=40, theta2=320,
                     color=VERMILION, lw=1.0))
    ax.annotate('', xy=(loop_cx + 0.045, loop_cy - 0.025),
                xytext=(loop_cx + 0.050, loop_cy - 0.010),
                arrowprops=dict(arrowstyle='-|>', lw=0.8,
                                color=VERMILION, shrinkA=0, shrinkB=0))
    ax.text(loop_cx + 0.07, loop_cy, 'Mathieu\npump',
            ha='left', va='center', fontsize=6.8, color=VERMILION,
            style='italic')

    # Field-magnitude legend (small, near bottom of main diagram)
    ax.text(cx, cy - 0.15,
            r'$\,|\mathbf{H}_\mathrm{mel}|\!\gg\!|\mathbf{H}_\mathrm{mr}|$',
            ha='center', va='center', fontsize=7.5, color=GREY_TXT)

    # =========================================================
    # Bottom inset (~25% of panel): SAW + substrate + FM cross-section
    # =========================================================
    inset_left, inset_right = 0.10, 0.90
    sub_y0, sub_y1 = 0.02, 0.16
    film_y0, film_y1 = sub_y1, sub_y1 + 0.045

    # Substrate
    ax.add_patch(Rectangle((inset_left, sub_y0),
                           inset_right - inset_left, sub_y1 - sub_y0,
                           facecolor=GREY_FILL, edgecolor=GREY_LINE,
                           lw=0.4, hatch='////', alpha=0.8))
    # FM film
    ax.add_patch(Rectangle((inset_left, film_y0),
                           inset_right - inset_left, film_y1 - film_y0,
                           facecolor=FILM_FILL, edgecolor=BLACK, lw=0.5))
    ax.text(inset_right + 0.01, (film_y0 + film_y1) / 2, 'FM',
            ha='left', va='center', fontsize=6.5, color=BLACK)
    ax.text(inset_right + 0.01, (sub_y0 + sub_y1) / 2, 'piezo.',
            ha='left', va='center', fontsize=6.0, color=GREY_TXT)

    # SAW wave on top of film
    saw_y = film_y1 + 0.07
    xs = np.linspace(inset_left + 0.05, inset_right - 0.10, 200)
    ys = saw_y + 0.020 * np.sin(2 * np.pi * (xs - inset_left) / 0.16)
    ax.plot(xs, ys, color=BLACK, lw=0.8)
    ax.annotate('', xy=(inset_right - 0.05, saw_y),
                xytext=(inset_right - 0.10, saw_y),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color=BLACK,
                                shrinkA=0, shrinkB=0))
    ax.text(inset_left + 0.02, saw_y + 0.04,
            r'Rayleigh SAW $\mathbf{k}_\mathrm{SAW}$',
            ha='left', va='bottom', fontsize=6.5)

    # tiny m0 arrow inside film
    ax.annotate('', xy=(inset_left + 0.55, (film_y0 + film_y1) / 2),
                xytext=(inset_left + 0.30, (film_y0 + film_y1) / 2),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color=BLACK,
                                shrinkA=0, shrinkB=0))


# -----------------------------------------------------------------------------
def panel_b(ax):
    """Frequency-domain channel map: two peaks, two output rules."""
    ax.set_xlim(0.0, 3.4)
    ax.set_ylim(-0.85, 1.45)

    ax.set_yticks([])
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels([r'$\omega_K$', r'$2\omega_K$'])
    ax.tick_params(axis='x', length=3, pad=2)
    ax.set_ylabel(r'$|m_\perp|$ response', labelpad=2)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_visible(False)

    # x-axis label placed inline
    ax.text(3.35, -0.05, r'$\omega_\mathrm{SAW}$',
            ha='right', va='top', fontsize=8)

    omega = np.linspace(0.0, 3.4, 600)

    def lor(x, x0, w, A):
        return A * (w**2) / ((x - x0)**2 + w**2)

    direct = lor(omega, 1.0, 0.10, 0.55)
    para = lor(omega, 2.0, 0.10, 1.05)

    ax.fill_between(omega, 0, direct, color=SKY_BLUE, alpha=0.22, lw=0)
    ax.plot(omega, direct, color=SKY_BLUE, lw=1.2)
    ax.fill_between(omega, 0, para, color=VERMILION, alpha=0.22, lw=0)
    ax.plot(omega, para, color=VERMILION, lw=1.2)

    ax.text(1.0, 0.65,
            'MR direct\n(1 : 1)',
            ha='center', va='bottom', fontsize=7.5, color=SKY_BLUE)
    ax.text(2.0, 1.16,
            'MEL parametric\n(2 : 1)',
            ha='center', va='bottom', fontsize=7.5, color=VERMILION)

    # Output band
    out_y = -0.55
    ax.plot([0.4, 3.0], [out_y, out_y], color=BLACK, lw=0.7)
    ax.plot([1.0], [out_y], 'o', color=BLACK, markersize=4.5, zorder=5)
    ax.text(1.0, out_y - 0.13,
            r'magnon at $\omega_K$',
            ha='center', va='top', fontsize=7)

    # Direct (1 down-arrow, MR -> magnon)
    ax.annotate('', xy=(1.0, out_y + 0.06), xytext=(1.0, 0.04),
                arrowprops=dict(arrowstyle='-|>', lw=0.9,
                                color=SKY_BLUE, shrinkA=0, shrinkB=0))
    # Parametric (2 arrows from 2omega_K -> magnon, splitting into pair)
    ax.annotate('', xy=(1.07, out_y + 0.06), xytext=(2.0, 0.04),
                arrowprops=dict(arrowstyle='-|>', lw=0.9,
                                color=VERMILION, shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(0.93, out_y + 0.06), xytext=(2.0, 0.04),
                arrowprops=dict(arrowstyle='-|>', lw=0.9,
                                color=VERMILION, shrinkA=0, shrinkB=0))

    ax.text(2.55, out_y + 0.05,
            r'$\mathbf{k}_1\!+\!\mathbf{k}_2\!=\!\mathbf{k}_\mathrm{SAW}$',
            ha='left', va='bottom', fontsize=7.5, color=VERMILION,
            fontweight='bold')
    ax.text(2.55, out_y - 0.08,
            r'(Suhl: $\mathbf{k}_1\!+\!\mathbf{k}_2\!=\!0$)',
            ha='left', va='top', fontsize=6.5, color=GREY_TXT,
            style='italic')


# -----------------------------------------------------------------------------
def main():
    apply_style()
    fig_w = SINGLE_COL
    fig_h = 6.2 * CM_TO_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.20],
                          left=0.02, right=0.99,
                          bottom=0.06, top=0.96, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_a(ax_a)
    panel_b(ax_b)

    ax_a.text(-0.02, 1.00, '(a)', transform=ax_a.transAxes,
              fontsize=9, fontweight='bold',
              va='top', ha='left', fontfamily='serif')
    ax_b.text(-0.12, 1.00, '(b)', transform=ax_b.transAxes,
              fontsize=9, fontweight='bold',
              va='top', ha='left', fontfamily='serif')

    out_base = "fig_prl_schematic_v3"
    for d in [FIG_DIR, PAPER_DIR]:
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(d, f"{out_base}.{ext}"),
                        format=ext, dpi=600,
                        bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_base}.{{pdf,png}} to {FIG_DIR} and {PAPER_DIR}")


if __name__ == "__main__":
    main()
