"""PRL Figure 1 schematic — three-panel central physics summary.

(a) Cross-sectional geometry: piezoelectric substrate carrying a Rayleigh
    SAW, ferromagnetic film on top, equilibrium magnetization m0 along
    the SAW wavevector. Annotated with the longitudinal strain eps_xx
    and the elliptical-motion rotation pseudovector Omega_y.
(b) Torque algebra at theta = 0: H_mel parallel to m0 (zero cross
    product) versus H_mr perpendicular to m0 (finite torque, drives
    precession). Visualized on a Bloch-style sphere section.
(c) Frequency-domain selection rules: an MR drive at omega = omega_K
    excites the Kittel magnon directly; an MEL drive at omega = 2 omega_K
    parametrically generates magnon pairs at omega_K through the
    eps_xx * m_x^2 nonlinearity.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, Ellipse, Arc, Rectangle,
                                Circle)
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from plot_style import (apply_style, DOUBLE_COL, SKY_BLUE, VERMILION,
                        TEAL, BLACK)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
PAPER_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "prl")
ARXIV_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "arxiv")
PRB_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "prb_letter")
os.makedirs(FIG_DIR, exist_ok=True)

GREY_LINE = '#5a5a5a'
GREY_FILL = '#ececec'
FILM_FILL = '#e8eef7'
ANNOT = '#404040'


def panel_a_geometry(ax):
    """SAW geometry: substrate + film + strain + rotation pseudovector."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')

    x_left, x_right = 0.04, 0.96
    surf_y = -0.05
    film_top_y = 0.10

    # --- Substrate (piezoelectric) ---
    ax.add_patch(Rectangle((x_left, -0.46), x_right - x_left,
                           surf_y + 0.46,
                           facecolor=GREY_FILL, edgecolor=GREY_LINE,
                           lw=0.5, hatch='////', alpha=0.85))
    ax.text(x_left + 0.02, -0.40, 'piezo. substrate',
            fontsize=6, color=ANNOT, va='center')

    # --- Ferromagnetic film ---
    ax.add_patch(Rectangle((x_left, surf_y), x_right - x_left,
                           film_top_y - surf_y,
                           facecolor=FILM_FILL, edgecolor=BLACK, lw=0.6))
    ax.text(x_right - 0.02, (surf_y + film_top_y) / 2, 'FM',
            fontsize=6.5, color=BLACK, va='center', ha='right',
            style='italic')

    # --- Equilibrium magnetization arrows inside film ---
    for cx in np.linspace(0.16, 0.84, 5):
        cy = (surf_y + film_top_y) / 2
        ax.annotate('', xy=(cx + 0.045, cy), xytext=(cx - 0.045, cy),
                    arrowprops=dict(arrowstyle='-|>',
                                    color=BLACK, lw=0.8,
                                    mutation_scale=8))

    # --- Big magnetization label above the film ---
    ax.annotate('', xy=(0.78, 0.30), xytext=(0.22, 0.30),
                arrowprops=dict(arrowstyle='-|>', color=BLACK, lw=1.7,
                                mutation_scale=14))
    ax.text(0.50, 0.38,
            r'$\boldsymbol{m}_0\parallel\boldsymbol{k}_\mathrm{SAW}\parallel\boldsymbol{B}_0$',
            ha='center', va='bottom', fontsize=9, color=BLACK)

    # --- Elliptical particle motion (rotation Omega_y) on top edge ---
    ell_centers = [0.22, 0.50, 0.78]
    for cx in ell_centers:
        cy = film_top_y + 0.02
        ell = Ellipse((cx, cy + 0.05), width=0.055, height=0.13,
                      facecolor='none', edgecolor=SKY_BLUE, lw=1.2,
                      zorder=4)
        ax.add_patch(ell)
        # CCW rotation indicator
        arc = Arc((cx, cy + 0.05), 0.055, 0.13, theta1=200, theta2=330,
                  color=SKY_BLUE, lw=1.1)
        ax.add_patch(arc)
        ax.annotate('', xy=(cx + 0.026, cy + 0.022),
                    xytext=(cx + 0.020, cy + 0.005),
                    arrowprops=dict(arrowstyle='-|>', color=SKY_BLUE,
                                    lw=0.9, mutation_scale=7))
        ax.annotate('', xy=(cx - 0.010, cy + 0.115),
                    xytext=(cx - 0.022, cy + 0.095),
                    arrowprops=dict(arrowstyle='-|>', color=SKY_BLUE,
                                    lw=0.8, mutation_scale=6))

    ax.text(ell_centers[0] - 0.07, film_top_y + 0.21,
            r'$\Omega_y\,$ (rotation)', color=SKY_BLUE,
            fontsize=8, ha='left')

    # --- Longitudinal strain bracket inside substrate ---
    y_strain = -0.18
    for cx, sign in zip(ell_centers, [+1, -1, +1]):
        dx = 0.07 * sign
        ax.annotate('', xy=(cx + dx, y_strain),
                    xytext=(cx - dx, y_strain),
                    arrowprops=dict(arrowstyle='<|-|>', color=VERMILION,
                                    lw=1.0, mutation_scale=8))
    ax.text(ell_centers[2] + 0.1, y_strain,
            r'$\varepsilon_{xx}\,$ (strain)', color=VERMILION,
            fontsize=8, ha='left', va='center')

    # --- SAW propagation arrow (large, at top of figure) ---
    ax.annotate('', xy=(x_right, 0.48), xytext=(x_left, 0.48),
                arrowprops=dict(arrowstyle='-|>', color=ANNOT, lw=0.9,
                                mutation_scale=10))
    ax.text((x_left + x_right) / 2, 0.50, r'SAW',
            ha='center', va='bottom', fontsize=8, color=ANNOT)

    # --- Coordinate triad (bottom-left corner) ---
    cx0, cy0 = 0.0, -0.45
    ax.annotate('', xy=(cx0, cy0 + 0.10), xytext=(cx0, cy0),
                arrowprops=dict(arrowstyle='-|>', color=ANNOT, lw=0.7,
                                mutation_scale=7))
    ax.annotate('', xy=(cx0 + 0.07, cy0), xytext=(cx0, cy0),
                arrowprops=dict(arrowstyle='-|>', color=ANNOT, lw=0.7,
                                mutation_scale=7))
    ax.text(cx0 - 0.025, cy0 + 0.10, r'$z$', fontsize=7, color=ANNOT,
            va='center')
    ax.text(cx0 + 0.075, cy0 - 0.005, r'$x$', fontsize=7, color=ANNOT,
            va='center')


def panel_b_mechanism(ax):
    """Torque algebra: H_mel parallel m0 (zero) vs H_mr perp m0 (finite)."""
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')

    # Two side-by-side schematics: left = MEL, right = MR

    def draw_sphere(cx, cy, r=0.16, alpha_fill=0.05):
        """Draw a Bloch-sphere outline at (cx, cy)."""
        circle = Circle((cx, cy), r, facecolor=BLACK, alpha=alpha_fill,
                        edgecolor=ANNOT, lw=0.5, zorder=1)
        ax.add_patch(circle)
        # Equator (ellipse for perspective)
        eq = Ellipse((cx, cy), 2 * r, 0.45 * r, facecolor='none',
                     edgecolor=ANNOT, lw=0.4, ls=':', zorder=1)
        ax.add_patch(eq)

    # --- LEFT: MEL parallel m0 (zero torque) ---
    cx_l, cy_l = 0.27, 0.05
    draw_sphere(cx_l, cy_l)
    # m0 vector
    ax.annotate('', xy=(cx_l + 0.18, cy_l), xytext=(cx_l, cy_l),
                arrowprops=dict(arrowstyle='-|>', color=BLACK, lw=1.6,
                                mutation_scale=11))
    ax.text(cx_l + 0.20, cy_l - 0.005, r'$\boldsymbol{m}_0$',
            fontsize=9, va='center')
    # H_mel parallel to m0 (offset slightly above)
    ax.annotate('', xy=(cx_l + 0.20, cy_l + 0.06),
                xytext=(cx_l - 0.02, cy_l + 0.06),
                arrowprops=dict(arrowstyle='-|>', color=VERMILION,
                                lw=2.2, mutation_scale=14))
    ax.text(cx_l + 0.09, cy_l + 0.13,
            r'$\boldsymbol{H}_\mathrm{mel}\parallel\boldsymbol{m}_0$',
            color=VERMILION, fontsize=9, ha='center')
    # Result: zero torque
    ax.text(cx_l, cy_l - 0.30,
            r'$\boldsymbol{m}\!\times\!\boldsymbol{H}_\mathrm{mel}=\boldsymbol{0}$',
            ha='center', fontsize=10, color=BLACK)
    ax.text(cx_l, cy_l - 0.42, 'silent', ha='center', fontsize=8.5,
            color=VERMILION)

    # --- RIGHT: MR perp m0 (finite torque, precession) ---
    cx_r, cy_r = 0.73, 0.05
    draw_sphere(cx_r, cy_r)
    # m0 vector
    ax.annotate('', xy=(cx_r + 0.18, cy_r), xytext=(cx_r, cy_r),
                arrowprops=dict(arrowstyle='-|>', color=BLACK, lw=1.6,
                                mutation_scale=11))
    ax.text(cx_r + 0.20, cy_r - 0.005, r'$\boldsymbol{m}_0$',
            fontsize=9, va='center')
    # H_mr perpendicular (out of equator, vertical)
    ax.annotate('', xy=(cx_r, cy_r + 0.18), xytext=(cx_r, cy_r),
                arrowprops=dict(arrowstyle='-|>', color=SKY_BLUE,
                                lw=1.6, mutation_scale=11))
    ax.text(cx_r - 0.02, cy_r + 0.13,
            r'$\boldsymbol{H}_\mathrm{mr}$',
            color=SKY_BLUE, fontsize=9, ha='right', va='center')
    # Precession arrow (curved, in the plane perp to m0)
    arc = FancyArrowPatch((cx_r + 0.05, cy_r - 0.04),
                          (cx_r + 0.05, cy_r + 0.04),
                          connectionstyle="arc3,rad=-0.9",
                          arrowstyle='-|>', color=TEAL, lw=1.4,
                          mutation_scale=10)
    ax.add_patch(arc)
    ax.text(cx_r + 0.13, cy_r - 0.10,
            r'$\boldsymbol{\tau}=\boldsymbol{m}\!\times\!\boldsymbol{H}_\mathrm{mr}$',
            color=TEAL, fontsize=8, ha='left')
    # Result: nonzero torque
    ax.text(cx_r, cy_r - 0.30,
            r'$\boldsymbol{m}\!\times\!\boldsymbol{H}_\mathrm{mr}\neq\boldsymbol{0}$',
            ha='center', fontsize=10, color=BLACK)
    ax.text(cx_r, cy_r - 0.42, 'sole driver', ha='center', fontsize=8.5,
            color=SKY_BLUE)


def panel_c_frequency(ax):
    """Frequency selection rules:
       (top) MR direct: omega_SAW = omega_K -> magnon @ omega_K
       (bottom) MEL parametric: omega_SAW = 2 omega_K -> magnon pair @ omega_K
    """
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')

    # Magnon mode level at center
    y_K = 0.0
    ax.plot([0.10, 0.95], [y_K, y_K], color=BLACK, lw=1.0)
    ax.text(0.97, y_K, r'$\omega_K$', va='center', fontsize=9, color=BLACK)
    ax.text(0.10, y_K - 0.06, 'magnon mode', fontsize=7, color=ANNOT,
            ha='left', va='top')

    # ===== Direct (top half) =====
    y_pump_top = 0.36
    ax.plot([0.10, 0.95], [y_pump_top, y_pump_top],
            color=ANNOT, lw=0.4, ls=':')
    ax.text(0.10, y_pump_top + 0.04, r'$\omega_\mathrm{SAW}=\omega_K$',
            fontsize=8, color=BLACK, ha='left')
    # Single direct arrow
    ax.annotate('', xy=(0.42, y_K + 0.02),
                xytext=(0.42, y_pump_top - 0.02),
                arrowprops=dict(arrowstyle='-|>', color=SKY_BLUE,
                                lw=1.6, mutation_scale=11))
    ax.text(0.49, (y_K + y_pump_top) / 2 + 0.02, 'MR',
            color=SKY_BLUE, fontsize=8.5, va='center', fontweight='bold')

    # ===== Parametric (bottom half) =====
    y_pump_bot = -0.36
    ax.plot([0.10, 0.95], [y_pump_bot, y_pump_bot],
            color=ANNOT, lw=0.4, ls=':')
    ax.text(0.10, y_pump_bot - 0.06, r'$\omega_\mathrm{SAW}=2\omega_K$',
            fontsize=8, color=BLACK, ha='left', va='top')
    # Two arrows from pump line splitting into magnon pair on mode line
    pump_x = 0.72
    arr1 = FancyArrowPatch((pump_x, y_pump_bot + 0.02),
                           (pump_x - 0.10, y_K - 0.02),
                           arrowstyle='-|>', color=VERMILION, lw=1.6,
                           mutation_scale=11,
                           connectionstyle='arc3,rad=0.05')
    arr2 = FancyArrowPatch((pump_x, y_pump_bot + 0.02),
                           (pump_x + 0.10, y_K - 0.02),
                           arrowstyle='-|>', color=VERMILION, lw=1.6,
                           mutation_scale=11,
                           connectionstyle='arc3,rad=-0.05')
    ax.add_patch(arr1)
    ax.add_patch(arr2)
    ax.text(pump_x + 0.20, (y_K + y_pump_bot) / 2 - 0.01, 'MEL',
            color=VERMILION, fontsize=8.5, va='center',
            fontweight='bold')
    # Conservation labels above the magnon line (k1+k2 = k_SAW)
    ax.text(0.86, y_K + 0.10,
            r'$\omega_K\!+\!\omega_K=2\omega_K$',
            ha='center', fontsize=7, color=VERMILION)
    ax.text(0.86, y_K + 0.18,
            r'$k_1\!+\!k_2=k_\mathrm{SAW}$',
            ha='center', fontsize=7, color=VERMILION)

    # Labels for direct vs parametric (bottom)
    ax.text(0.42, y_pump_top - 0.10, 'direct',
            ha='center', fontsize=7.5, color=SKY_BLUE)
    ax.text(0.80, y_pump_bot - 0.11, 'parametric pair',
            ha='center', fontsize=7.5, color=VERMILION)


def main():
    apply_style()
    plt.rcParams['font.size'] = 8
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04,
                        wspace=0.04)

    panel_a_geometry(axes[0])
    panel_b_mechanism(axes[1])
    panel_c_frequency(axes[2])

    # Panel labels (a), (b), (c) — bold, top-left aligned
    for i, ax in enumerate(axes):
        ax.text(0.0, 1.04, f'({chr(97 + i)})',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                va='bottom', ha='left')

    out_paths = []
    for ext in ('pdf', 'png'):
        p = os.path.join(FIG_DIR, f'fig_prl_schematic.{ext}')
        fig.savefig(p, dpi=300)
        out_paths.append(p)

    import shutil
    src_pdf = os.path.join(FIG_DIR, 'fig_prl_schematic.pdf')
    for d in (PAPER_DIR, ARXIV_DIR, PRB_DIR):
        if os.path.isdir(d):
            shutil.copy2(src_pdf, os.path.join(d, 'fig_prl_schematic.pdf'))
    plt.close(fig)
    for p in out_paths:
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
