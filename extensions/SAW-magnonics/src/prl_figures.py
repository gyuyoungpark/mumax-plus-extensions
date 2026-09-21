"""PRL Figure Assembly: Parametric magnon generation in SAW-FMR.

3 figures, each 1x2 layout:

Figure 1 (premise): Silent giant
  (a) Coupling field hierarchy (field vs torque)
  (b) Channel decomposition time trace (sim10_micro)

Figure 2 (discovery): Parametric revival — MAIN FIGURE
  (a) Three-channel FMR spectra: MEL-only peaks at 2f_K
  (b) Peak frequency dispersion vs B0

Figure 3 (predictions): Testable signatures
  (a) Strain amplitude threshold (sim21)
  (b) Interference dip near theta_c (sim22)
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from plot_style import (apply_style, label_panels, axis_label,
                        SINGLE_COL, DOUBLE_COL, CM_TO_INCH,
                        SKY_BLUE, VERMILION, TEAL, YELLOW, PINK, ORANGE, BLUE, BLACK)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "prl")

GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7
MS = 140e3
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
EPS0 = 1e-4
F_SAW = 3.0e9
OMEGA_SAW = 2 * np.pi * F_SAW
FS = 10  # base fontsize


def kittel_freq(B0):
    return GAMMA / (2 * np.pi) * np.sqrt(np.maximum(B0 * (B0 + MU0 * MS), 0))


# ===========================================================================
# Figure 1: Silent giant (premise)
# ===========================================================================
def fig1_silent_giant():
    apply_style()
    fig, ax_only = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.82))
    fig.subplots_adjust(left=0.20, right=0.96, top=0.95, bottom=0.18)
    axes = [None, ax_only]  # only panel (b) is shown

    # Channel decomposition (sim10_micro) — single panel
    ax = axes[1]
    cache = os.path.join(DATA_DIR, "sim10_micro_validation.npz")
    if os.path.isfile(cache):
        d = dict(np.load(cache, allow_pickle=True))
        t_ns = d['full_times'] * 1e9
        ax.plot(t_ns, d['full_m_perp'], '-', color=BLACK, lw=0.8,
                label='MEL+MR')
        ax.plot(t_ns, d['mr_only_m_perp'], '-', color=SKY_BLUE, lw=0.8,
                label='MR only')
        ax.plot(t_ns, d['mel_only_m_perp'], '-', color=VERMILION, lw=0.8,
                label='MEL only', alpha=0.7)
    ax.set_xlabel(axis_label(r'$t$', 'ns'), fontsize=FS)
    ax.set_ylabel(axis_label(r'$|m_\perp|$'), fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 2)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    save_fig(fig, "fig_prl_silent_giant")


# ===========================================================================
# Figure 2: Parametric revival (MAIN)
# ===========================================================================
def fig2_parametric():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.5))
    fig.subplots_adjust(wspace=0.35)

    cache = os.path.join(DATA_DIR, "sim20_parametric_channels.npz")
    if not os.path.isfile(cache):
        print("  WARNING: sim20 data not found")
        plt.close(fig)
        return

    d = np.load(cache)
    f_ghz = d['f_saw_values'] * 1e-9
    B0_vals = d['B0_values']
    sf = d['spectra_full']
    sm = d['spectra_mr_only']
    se = d['spectra_mel_only']

    # (a) 3-channel spectra at B0=50 mT (index 2)
    ax = axes[0]
    bi = 2
    f_K = kittel_freq(B0_vals[bi]) * 1e-9
    norm = max(np.max(sf[bi]), 1e-10)

    ax.plot(f_ghz, sf[bi] / norm, '-', color=BLACK, lw=1.0,
            label='Full (MEL+MR)')
    ax.plot(f_ghz, sm[bi] / norm, '--', color=SKY_BLUE, lw=0.9,
            label='MR only')
    ax.plot(f_ghz, se[bi] / norm, ':', color=VERMILION, lw=1.2,
            label='MEL only')
    ax.axvline(f_K, color='gray', ls=':', lw=0.5, alpha=0.6)
    ax.axvline(2 * f_K, color=VERMILION, ls=':', lw=0.5, alpha=0.6)
    ax.set_xlabel(axis_label(r'$f_\mathrm{SAW}$', 'GHz'), fontsize=FS)
    ax.set_ylabel(r'$|m_\perp|$ (norm.)', fontsize=FS)
    ax.set_yticks([])
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 2)

    # (b) Peak frequency vs B0
    ax = axes[1]
    for spec_arr, label, color, marker in [
            (sf, 'Full', VERMILION, 'o'),
            (sm, 'MR only', SKY_BLUE, 's'),
            (se, 'MEL only', TEAL, '^')]:
        peaks_f, b0_valid = [], []
        for bi in range(len(B0_vals)):
            s = spec_arr[bi]
            if np.max(s) > 1e-6:
                pks, _ = find_peaks(s, height=0.3 * np.max(s))
                if len(pks):
                    peaks_f.append(f_ghz[pks[np.argmax(s[pks])]])
                    b0_valid.append(B0_vals[bi] * 1e3)
        if peaks_f:
            ax.plot(b0_valid, peaks_f, marker, color=color, ms=7,
                    mec='k', mew=0.3, label=label, zorder=5)

    B_fit = np.linspace(5, 150, 200) * 1e-3
    f_K_fit = np.array([kittel_freq(b) * 1e-9 for b in B_fit])
    ax.plot(B_fit * 1e3, f_K_fit, 'k--', lw=0.7, label=r'$\omega_K$')
    ax.plot(B_fit * 1e3, 2 * f_K_fit, '--', color=VERMILION, lw=0.7,
            alpha=0.5, label=r'$2\omega_K$')
    ax.set_xlabel(axis_label(r'$B_0$', 'mT'), fontsize=FS)
    ax.set_ylabel(axis_label(r'$f_\mathrm{peak}$', 'GHz'), fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 3, loc='upper left')

    label_panels(axes)
    save_fig(fig, "fig_prl_parametric")


# ===========================================================================
# Figure 3: Testable predictions
# ===========================================================================
def fig3_predictions():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.5))
    fig.subplots_adjust(wspace=0.35)

    # (a) eps0 threshold (sim21)
    ax = axes[0]
    cache21 = os.path.join(DATA_DIR, "sim21_eps_threshold.npz")
    if os.path.isfile(cache21):
        d = np.load(cache21)
        eps_vals = d['eps_values']
        f_saw = d['f_saw_values']
        sf = d['spectra_full']
        sm = d['spectra_mr_only']

        mask_fK = (f_saw > 2e9) & (f_saw < 4e9)
        mask_2fK = (f_saw > 4.5e9) & (f_saw < 7.5e9)

        peak_fK_mr = [np.max(sm[ei][mask_fK]) for ei in range(len(eps_vals))]
        peak_2fK_full = [np.max(sf[ei][mask_2fK]) for ei in range(len(eps_vals))]

        ax.loglog(eps_vals, peak_fK_mr, 's-', color=SKY_BLUE, ms=6,
                  mec='k', mew=0.3, label=r'MR @ $\omega_K$')
        ax.loglog(eps_vals, peak_2fK_full, 'o-', color=VERMILION, ms=6,
                  mec='k', mew=0.3, label=r'Full @ $2\omega_K$')

        # Linear-in-eps0 fit on the SAW-dominated regime
        # (eps0 >= 3e-4; below this baseline transients dominate)
        eps_arr = np.asarray(eps_vals, dtype=float)
        mr_arr = np.asarray(peak_fK_mr, dtype=float)
        sub_mask = (eps_arr >= 3e-4) & (mr_arr > 0)
        if np.sum(sub_mask) >= 2:
            slope, intercept = np.polyfit(np.log10(eps_arr[sub_mask]),
                                          np.log10(mr_arr[sub_mask]), 1)
            eps_line = np.array([3e-4, eps_arr.max() * 1.5])
            ax.loglog(eps_line, 10 ** intercept * eps_line ** slope,
                      '--', color=SKY_BLUE, lw=0.7, alpha=0.6,
                      label=rf'$\propto\varepsilon_0^{{{slope:.2f}}}$')

    ax.set_xlabel(r'$\varepsilon_0$', fontsize=FS)
    ax.set_ylabel(r'Peak $|m_\perp|$', fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 2)

    # (b) Fine angle sweep with dip (sim22)
    ax = axes[1]
    cache22 = os.path.join(DATA_DIR, "sim22_fine_angle.npz")
    if os.path.isfile(cache22):
        d = np.load(cache22)
        angles = d['angles']
        peaks = d['peaks']
        theta_c = float(d['theta_c'])
        norm = peaks[0] if peaks[0] > 0 else 1

        ax.plot(angles, peaks / norm, 'o-', color=SKY_BLUE, ms=6,
                mec='k', mew=0.3, lw=1.0, label='Micromagnetic', zorder=5)

        theta_an = np.linspace(0, np.pi / 2, 500)
        g_mel = GAMMA * abs(B1) * EPS0 * np.abs(np.sin(2 * theta_an)) / MS
        g_mr = GAMMA * KMR * XI * EPS0 * np.abs(np.cos(theta_an)) / (2 * MS)
        g_total = np.sqrt(g_mel**2 + g_mr**2)
        g_norm = g_total / g_total[0] if g_total[0] > 0 else g_total
        ax.plot(np.degrees(theta_an), g_norm, 'k:', lw=0.6, alpha=0.5,
                label=r'Linear $g_\mathrm{total}$')

        ax.axvline(theta_c, ls=':', color='gray', lw=0.5)

        # Arrow at dip
        dip_mask = angles < 3
        if np.sum(dip_mask) > 2:
            dip_idx = np.argmin(peaks[dip_mask])
            if peaks[dip_mask][dip_idx] < peaks[0] * 0.5:
                ax.annotate('', xy=(angles[dip_mask][dip_idx],
                            peaks[dip_mask][dip_idx] / norm),
                            xytext=(angles[dip_mask][dip_idx], 0.8),
                            arrowprops=dict(arrowstyle='->', color=VERMILION,
                                            lw=1.0))

    ax.set_xlabel(r'$\theta$ (deg)', fontsize=FS)
    ax.set_ylabel(r'Normalized $|m_\perp|$', fontsize=FS)
    ax.set_xlim(-0.5, 47)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 2)

    label_panels(axes)
    save_fig(fig, "fig_prl_predictions")


def save_fig(fig, name):
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=200)
    if os.path.isdir(PAPER_DIR):
        shutil.copy2(os.path.join(FIG_DIR, f'{name}.pdf'),
                     os.path.join(PAPER_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f"Saved: {name}.pdf")


def main():
    print("=" * 60)
    print("PRL Figure Assembly (v3: 3 figures)")
    print("=" * 60)
    fig1_silent_giant()
    fig2_parametric()
    fig3_predictions()
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
