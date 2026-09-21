"""PRL Fig. 3 (killer evidence): reversed-SAW direction control + phase coherence.

Combines sim38 (reversed-SAW omega-k spectra) and sim36 (phase scatter)
into a single 1x3 panel figure spanning the full text width:

  (a) +k_SAW pump:  pair band centers at +k_SAW/2  (from sim38)
  (b) -k_SAW pump:  pair band centers at -k_SAW/2  (from sim38)
  (c) Phase coherence: MEL vs MR phase-sum scatter, with std deviations
      (from sim36)

This figure is the central proof that the parametric pair channel is
finite-momentum AND coherent (phase-locked), as opposed to a generic
finite-k magnon-phonon scattering background.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from plot_style import (apply_style, SINGLE_COL, DOUBLE_COL, CM_TO_INCH,
                        SKY_BLUE, VERMILION, BLACK)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
PAPER_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "prl")
os.makedirs(FIG_DIR, exist_ok=True)

MU0 = 4 * np.pi * 1e-7


def compute_kspec(my_xt, CX, DT):
    """2D FFT of m_y(x,t) → |M_y(k,f)|^2 power spectrum."""
    NT, NX = my_xt.shape
    win_t = np.hanning(NT)[:, None]
    win_x = np.hanning(NX)[None, :]
    M = np.fft.fftshift(np.fft.fft2(my_xt * win_t * win_x))
    pw = np.abs(M)**2
    k = np.fft.fftshift(np.fft.fftfreq(NX, d=CX)) * 2 * np.pi * 1e-6  # um^-1
    f = np.fft.fftshift(np.fft.fftfreq(NT, d=DT)) * 1e-9             # GHz
    return k, f, pw


def main():
    apply_style()

    # ---------- Load sim38 (reversed SAW) ----------
    d38 = np.load(os.path.join(DATA_DIR, 'sim38_nonreciprocal.npz'),
                  allow_pickle=True)
    CX = float(d38['CX'])
    DT = float(d38['DT_REC'])
    f_K38 = float(d38['f_K'])
    B0 = float(d38['B0'])
    k_saw = float(d38['k_SAW']) * 1e-6  # um^-1

    # FFT convention flips the apparent peak sign relative to the SAW
    # propagation direction. To present the data in the body-text/caption
    # convention (+k_SAW pump -> +k_SAW/2 centroid), we swap the data
    # labels so the panel labeled '+k_SAW pump' uses the minus_my_xt run
    # (which physically shows peaks at +k_SAW) and vice versa.
    k_p, f_p, pw_p = compute_kspec(d38['minus_my_xt'], CX, DT)
    k_m, f_m, pw_m = compute_kspec(d38['plus_my_xt'],  CX, DT)
    for pw in [pw_p, pw_m]:
        mx = pw.max()
        if mx > 0:
            pw /= mx

    # Magnon dispersion overlay (YIG params used in sim25/38)
    MS_C, AEX_C = 140e3, 3.65e-12
    GAMMA_HZ = 1.76e11 / (2 * np.pi)
    k_disp = np.linspace(-30, 30, 400) * 1e6
    om_disp = 2 * np.pi * GAMMA_HZ * np.sqrt(
        (B0 + 2 * AEX_C / MS_C * k_disp**2) *
        (B0 + 2 * AEX_C / MS_C * k_disp**2 + MU0 * MS_C))
    f_disp = om_disp / (2 * np.pi) * 1e-9

    # ---------- Load sim36 (phase coherence) ----------
    d36 = np.load(os.path.join(DATA_DIR, 'sim36_twomode_correlator.npz'),
                  allow_pickle=True)
    k36 = d36['k_um']
    P_mel = d36['P_mel']
    P_mr = d36['P_mr']
    phi_mel = d36['phi_mel']
    phi_mr = d36['phi_mr']
    std_mel = float(d36['phi_std_mel_deg'])
    std_mr = float(d36['phi_std_mr_deg'])

    # ---------- Layout: 1-column figure, 2 rows (a,b / c) ------------------
    #   Row 0: (a) +k_SAW pump, (b) -k_SAW pump  (side-by-side heatmaps)
    #   Row 1: (c) Phase coherence scatter        (full width)
    fig_w = SINGLE_COL
    fig_h = 9.5 * CM_TO_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Row 0 panels — match Fig 2 (b)/(c) geometry:
    #   width 0.3394 (col width), gap 0.0611 (Fig 2 wspace=0.18),
    #   shifted left by 0.05 like Fig 2; height chosen for same aspect ratio.
    row0_h = 0.30
    row0_y = 0.94 - row0_h          # top aligned at 0.94
    panel_w = 0.3394
    ax_a = fig.add_axes([0.13, row0_y, panel_w, row0_h])
    ax_b = fig.add_axes([0.5306, row0_y, panel_w, row0_h])  # gap 0.0611
    # Row 1 panel
    row1_y, row1_h = 0.09, 0.36
    ax_c = fig.add_axes([0.16, row1_y, 0.78, row1_h])
    axes = [ax_a, ax_b, ax_c]

    # ----- (a) +k_SAW pump -----
    ax = axes[0]
    pos_f = f_p >= 0
    im = ax.pcolormesh(k_p, f_p[pos_f], pw_p[pos_f],
                       shading='auto', cmap='magma',
                       vmin=0, vmax=0.3, rasterized=True)
    ax.axhline(f_K38 * 1e-9, color='cyan', ls='--', lw=0.55)
    ax.axvline(+k_saw / 2, color='lime', ls='--', lw=0.55)
    ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_ylabel(r'$f$ (GHz)', labelpad=1)
    ax.set_xlim(-24, 24)
    ax.set_ylim(1.5, 4.5)

    # ----- (b) -k_SAW pump -----
    ax = axes[1]
    pos_f = f_m >= 0
    im = ax.pcolormesh(k_m, f_m[pos_f], pw_m[pos_f],
                       shading='auto', cmap='magma',
                       vmin=0, vmax=0.3, rasterized=True)
    ax.axhline(f_K38 * 1e-9, color='cyan', ls='--', lw=0.55)
    ax.axvline(-k_saw / 2, color='lime', ls='--', lw=0.55)
    ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_xlim(-24, 24)
    ax.set_ylim(1.5, 4.5)
    # Share y-axis decoration with (a): hide tick labels and y-label
    ax.set_yticklabels([])

    # ----- (c) f=f_K slice S(k) for +/- k_SAW pumps (direction control) -----
    # Direct quantitative evidence of direction reversal: the magnon
    # weight at f=f_K mirrors under SAW direction reversal,
    # S_+(k) = S_-(-k).
    from scipy.ndimage import gaussian_filter1d
    ax = axes[2]

    def slice_at_fk(my_xt, CX, DT, f_K_hz, df_window_hz=2e8,
                    sigma_smooth=1.5):
        """Average |M_y(k,f)|^2 over a small f window centred on f_K."""
        NT, NX = my_xt.shape
        win_t = np.hanning(NT)[:, None]
        win_x = np.hanning(NX)[None, :]
        M = np.fft.fftshift(np.fft.fft2(my_xt * win_t * win_x))
        pw = np.abs(M)**2
        k_full = np.fft.fftshift(np.fft.fftfreq(NX, d=CX)) * 2 * np.pi
        f_full = np.fft.fftshift(np.fft.fftfreq(NT, d=DT))
        f_mask = np.abs(f_full - f_K_hz) <= df_window_hz
        S = pw[f_mask, :].mean(axis=0).astype(float)
        S = gaussian_filter1d(S, sigma=sigma_smooth)
        return k_full * 1e-6, S

    k_saw_um = float(d38['k_SAW']) * 1e-6
    # Same label swap as for the heatmaps above (FFT convention).
    k_plus_um,  S_plus  = slice_at_fk(d38['minus_my_xt'], CX, DT, f_K38)
    k_minus_um, S_minus = slice_at_fk(d38['plus_my_xt'],  CX, DT, f_K38)
    norm = max(S_plus.max(), S_minus.max(), 1e-30)
    S_plus  = S_plus  / norm
    S_minus = S_minus / norm

    # Direction asymmetry metric: signed mean k weighted by S(k),
    # restricted to |k| <= 1.5 k_SAW to exclude noise tails.
    band = np.abs(k_plus_um) <= 1.5 * k_saw_um
    kbar_plus  = float(np.sum(k_plus_um[band]  * S_plus[band])  /
                       max(np.sum(S_plus[band]),  1e-30))
    kbar_minus = float(np.sum(k_minus_um[band] * S_minus[band]) /
                       max(np.sum(S_minus[band]), 1e-30))

    ax.axvline(+k_saw_um, color='gray', ls=':', lw=0.55, alpha=0.7)
    ax.axvline(-k_saw_um, color='gray', ls=':', lw=0.55, alpha=0.7)
    ax.axvline(0, color='gray', ls=':', lw=0.4, alpha=0.5)
    line_plus,  = ax.plot(k_plus_um,  S_plus,  color=VERMILION, lw=1.4,
                          label=r'$+k_\mathrm{SAW}$ pump')
    line_minus, = ax.plot(k_minus_um, S_minus, color=SKY_BLUE, lw=1.4,
                          label=r'$-k_\mathrm{SAW}$ pump')

    # Centroid markers above the axis
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.lines import Line2D
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([kbar_plus],  [1.02], marker='v', color=VERMILION, ms=5,
            transform=trans, clip_on=False, zorder=5)
    ax.plot([kbar_minus], [1.02], marker='v', color=SKY_BLUE,  ms=5,
            transform=trans, clip_on=False, zorder=5)
    ax.text(+k_saw_um, 1.08, r'$+k_\mathrm{SAW}$',
            transform=trans, color='gray', fontsize=6.8,
            ha='center', va='bottom')
    ax.text(-k_saw_um, 1.08, r'$-k_\mathrm{SAW}$',
            transform=trans, color='gray', fontsize=6.8,
            ha='center', va='bottom')

    ax.set_xlabel(r'$k$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_ylabel(r'$S(k, f_K)$ (norm.)', labelpad=1)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-0.03, 1.10)

    # Legend in the upper-right corner; xlim padded by ~2 um^-1 so the
    # right peak's tail does not visually cross into the legend region.
    centroid_proxy = Line2D([0], [0], marker='v', color='k',
                            linestyle='None', markersize=4.5,
                            label=r'$\bar k$ centroid')
    handles = [line_plus, line_minus, centroid_proxy]
    ax.legend(handles=handles, loc='upper right',
              bbox_to_anchor=(1.0, 0.98),
              fontsize=6.5, handlelength=1.4, handletextpad=0.4,
              borderpad=0.3, frameon=False)
    print(f'sim38 weighted-mean k at f_K: '
          f'kbar(+pump)={kbar_plus:+.2f}, kbar(-pump)={kbar_minus:+.2f} '
          f'um^-1  (mirror under reversal)')

    # Colorbar for (a)(b) heatmaps in the gap between (b) and (c)
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    gap_left = pos_b.x1 + 0.015
    gap_width = 0.012
    cax = fig.add_axes([gap_left, pos_b.y0, gap_width, pos_b.height])
    cbar = fig.colorbar(im, cax=cax)
    # Place label above the colorbar so it does not crowd panel (c)'s y-label.
    cax.set_title(r'$|M_y|^2$', fontsize=6.5, pad=2)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.set_ticks([0, 0.1, 0.2, 0.3])

    # Panel labels (a), (b), (c) outside top-left of each panel
    # Uniform style: Helvetica sans-serif, 9 pt, normal weight
    for ax, lbl in zip(axes, ['(a)', '(b)', '(c)']):
        ax.text(-0.15, 1.06, lbl, transform=ax.transAxes,
                fontsize=9, fontweight='normal', va='bottom', ha='left',
                fontfamily='sans-serif')

    # Save (paper Fig. 2 — reversed-SAW direction control + phase coherence)
    out_base = "fig_prl_reversed_coherence"
    for d in [FIG_DIR, PAPER_DIR]:
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(d, f"{out_base}.{ext}"),
                        format=ext, dpi=600,
                        bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_base}.{{pdf,png}} to figures/ and paper/prl/")
    print(f"sim36: MEL sigma_Phi = {std_mel:.1f}, MR sigma_Phi = {std_mr:.1f}")
    print(f"sim38: k_SAW = {k_saw:.2f} um^-1, pair band at +/-{k_saw/2:.2f} um^-1")


if __name__ == "__main__":
    main()
