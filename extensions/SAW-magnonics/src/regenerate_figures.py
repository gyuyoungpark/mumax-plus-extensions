"""Regenerate all SAW-magnonics figures (sim01--sim07, skipping sim04)
with PRL style: no legend boxes (frameon=False).

This script loads the cached .npy data files and recreates each figure
using the updated plot_style module.
"""

import os
import sys

# Ensure plot_style and other modules are importable
sys.path.insert(0, r'D:\mumax-plus-dev\extension')
sys.path.insert(0, r'D:\mumax-plus-dev\extension\SAW-magnonics')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks

import plot_style as ps

DATA_DIR = r'D:\mumax-plus-dev\extension\SAW-magnonics\data'
FIG_DIR  = r'D:\mumax-plus-dev\extension\SAW-magnonics\figures'
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, basename):
    """Save figure as both PDF and PNG."""
    pdf_path = os.path.join(FIG_DIR, basename + '.pdf')
    png_path = os.path.join(FIG_DIR, basename + '.png')
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# ======================================================================
# SIM 01 — SAW-driven domain wall motion
# ======================================================================
def regen_sim01():
    print("\n=== Regenerating sim01: fig_saw_domain_wall ===")
    cache = np.load(os.path.join(DATA_DIR, 'sim01_data.npy'),
                    allow_pickle=True).item()
    results = cache['results']

    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.40)
    colors = ps.COLORS_6

    # (a) DW position vs time
    for i, res in enumerate(results):
        t_ns = res['time'] * 1e9
        pos_nm = res['dw_pos'] * 1e9
        label = rf"$\varepsilon_0 = {res['epsilon_0']*1e3:g}\times10^{{-3}}$"
        axes[0].plot(t_ns, pos_nm, color=colors[i], linewidth=1.0, label=label)

    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$x_\mathrm{DW}$ (nm)')
    axes[0].legend(fontsize=6.5, loc='center right', frameon=False)
    ps.add_panel_label(axes[0], '(a)')

    # (b) Velocity vs epsilon_0^2
    eps_arr = np.array([r['epsilon_0'] for r in results])
    vel_arr = np.array([r['velocity'] for r in results])
    eps2 = eps_arr**2

    axes[1].plot(eps2 * 1e6, vel_arr, 'o', color=ps.BLUE,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.4,
                 zorder=10)

    axes[1].set_xlabel(r'$\varepsilon_0^2$ ($\times 10^{-6}$)')
    axes[1].set_ylabel(r'$v_\mathrm{DW}$ (m/s)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    ps.add_panel_label(axes[1], '(b)')

    save_fig(fig, 'fig_saw_domain_wall')


# ======================================================================
# SIM 02 — SAW-driven ferromagnetic resonance
# ======================================================================
def regen_sim02():
    print("\n=== Regenerating sim02: fig_saw_fmr ===")
    data = np.load(os.path.join(DATA_DIR, 'sim02_data.npy'),
                   allow_pickle=True).item()

    B0_values   = data['B0_values']
    f_saw_values = data['f_saw_values']
    spectra     = data['spectra']

    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.40)
    colors = ps.COLORS_4

    # (a) Absorption spectra
    for i, B0 in enumerate(B0_values):
        f_ghz = f_saw_values * 1e-9
        m_perp = spectra[B0]

        if np.max(m_perp) > 0:
            m_norm = m_perp / np.max(m_perp)
        else:
            m_norm = m_perp

        offset = i * 1.2
        axes[0].fill_between(f_ghz, offset, offset + m_norm * 0.9,
                             color=colors[i], alpha=0.5)
        axes[0].plot(f_ghz, offset + m_norm * 0.9,
                     color=colors[i], linewidth=0.8)

        label = rf'$B_0 = {B0*1e3:.0f}$ mT'
        axes[0].text(2.0, offset + 1.0, label, fontsize=6.5,
                     color=colors[i], va='center')

    axes[0].set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    axes[0].set_ylabel('Magnon amplitude (a.u.)')
    axes[0].set_yticks([])
    axes[0].set_xlim(1, 15)
    ps.add_panel_label(axes[0], '(a)')

    # (b) Peak frequency vs B_0 + Kittel curve
    ABS_THRESHOLD = 1e-3
    B0_list = sorted(spectra.keys())
    B0_arr = np.array(B0_list)
    f_peaks = []
    for B0 in B0_list:
        m_arr = spectra[B0]
        if np.max(m_arr) > ABS_THRESHOLD:
            peaks_idx, _ = find_peaks(m_arr, height=0.3 * np.max(m_arr))
            if len(peaks_idx) > 0:
                i_best = peaks_idx[np.argmax(m_arr[peaks_idx])]
                f_peaks.append(f_saw_values[i_best])
            else:
                f_peaks.append(np.nan)
        else:
            f_peaks.append(np.nan)
    f_peaks = np.array(f_peaks)
    valid = ~np.isnan(f_peaks)
    B0_arr_valid = B0_arr[valid]
    f_peaks_valid = f_peaks[valid]

    axes[1].plot(B0_arr_valid * 1e3, f_peaks_valid * 1e-9, 'o',
                 color=ps.BLUE, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.4,
                 label='Simulation', zorder=10)

    # Analytical Kittel curve
    # gamma * sqrt(B0 * (B0 + mu0*Ms)) / (2*pi)
    MSAT_02 = 1.0e6
    MU0 = 4e-7 * np.pi
    GAMMA = 1.7595e11  # rad/(T.s)
    B_theory = np.linspace(0, B0_arr[-1] * 1.2, 200)
    f_kittel = (GAMMA / (2 * np.pi)) * np.sqrt(
        B_theory * (B_theory + MU0 * MSAT_02))
    axes[1].plot(B_theory * 1e3, f_kittel * 1e-9, '--',
                 color=ps.VERMILION, linewidth=1.0,
                 label=r'Kittel: $\gamma\sqrt{B_0(B_0 + \mu_0 M_s)}/2\pi$')

    axes[1].set_xlabel(r'$B_0$ (mT)')
    axes[1].set_ylabel(r'$f_\mathrm{peak}$ (GHz)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=6.5, loc='lower right', frameon=False)
    ps.add_panel_label(axes[1], '(b)')

    save_fig(fig, 'fig_saw_fmr')


# ======================================================================
# SIM 03 — SAW-assisted magnetization switching
# ======================================================================
def regen_sim03():
    print("\n=== Regenerating sim03: fig_saw_switching ===")
    cache = np.load(os.path.join(DATA_DIR, 'sim03_data.npy'),
                    allow_pickle=True).item()

    traces  = cache['traces']
    eps_arr = cache['eps_arr']
    Ba_arr  = cache['Ba_arr']
    mz_map  = cache['mz_map']

    # Constants from the original script
    T_RUN = 10e-9
    T_BURST_CENTER = 2e-9
    T_BURST_SIGMA  = 1e-9

    def gaussian_envelope(t, t0=T_BURST_CENTER, sigma=T_BURST_SIGMA):
        return np.exp(-0.5 * ((t - t0) / sigma)**2)

    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.42)
    colors = ps.COLORS_4

    # (a) Time traces
    for i, tr in enumerate(traces):
        t_ns = tr['time'] * 1e9
        label = rf"$\varepsilon_0 = {tr['epsilon_0']*1e3:.0f}\times10^{{-3}}$"
        ls = '--' if round(tr['epsilon_0']*1e3) == 5 else '-'
        axes[0].plot(t_ns, tr['mz'], color=colors[i], linewidth=1.0,
                     linestyle=ls, label=label)

    # Mark burst window
    t_burst = np.linspace(0, T_RUN, 200) * 1e9
    env = gaussian_envelope(t_burst * 1e-9)
    axes[0].fill_between(t_burst, -1.05, -1.05 + 0.15 * env,
                         color='gray', alpha=0.3, zorder=0)
    axes[0].text(T_BURST_CENTER * 1e9, -0.88, 'SAW burst',
                 fontsize=6, color='gray', ha='center')

    axes[0].axhline(y=0, color='gray', linewidth=0.3, linestyle=':')
    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$\langle m_z \rangle$')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_xlim(0, T_RUN * 1e9)
    axes[0].legend(fontsize=6, loc='center right', frameon=False)
    ps.add_panel_label(axes[0], '(a)')

    # (b) Phase diagram
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    pcm = axes[1].pcolormesh(
        eps_arr * 1e3, Ba_arr * 1e3, mz_map,
        cmap='RdBu', norm=norm, shading='auto')

    eps_mid = 0.5 * (eps_arr[0] + eps_arr[-1]) * 1e3
    Ba_high = 0.75 * (Ba_arr[0] + Ba_arr[-1]) * 1e3
    Ba_low = 0.30 * (Ba_arr[0] + Ba_arr[-1]) * 1e3
    axes[1].text(eps_mid * 1.5, Ba_high, 'Switched',
                 fontsize=8, ha='center', va='center',
                 fontweight='bold', color='white')
    axes[1].text(eps_mid * 0.55, Ba_low, 'Not switched',
                 fontsize=8, ha='center', va='center',
                 fontweight='bold', color='white')

    cb = fig.colorbar(pcm, ax=axes[1], pad=0.02, shrink=0.9)
    cb.set_label(r'Final $\langle m_z \rangle$')

    axes[1].set_xlabel(r'$\varepsilon_0$ ($\times 10^{-3}$)')
    axes[1].set_ylabel(r'$B_\mathrm{assist}$ (mT)')
    ps.add_panel_label(axes[1], '(b)')

    save_fig(fig, 'fig_saw_switching')


# ======================================================================
# SIM 05 — Rotation field validation
# ======================================================================
def regen_sim05():
    print("\n=== Regenerating sim05: fig_rotation_field_validation ===")
    cache = np.load(os.path.join(DATA_DIR, 'sim05_data.npy'),
                    allow_pickle=True).item()
    results = cache['results']

    ps.apply_style()
    n = len(results)
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(n, 1,
                             figsize=(ps.SINGLE_COL,
                                      2.2 * n * ps.CM_TO_INCH * 2.54),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    for i, (ax, res) in enumerate(zip(axes, results)):
        x = res['x']
        for c, clabel in enumerate(['x', 'y', 'z']):
            h_num = res['H_num'][c, 0, 0, :]
            h_ana = res['H_ana'][c, 0, 0, :]
            if np.max(np.abs(h_ana)) < 1e-30 and np.max(np.abs(h_num)) < 1e-30:
                continue
            ax.plot(x, h_ana, '-', label=f'$H_{clabel}$ (ana)', alpha=0.8)
            ax.plot(x, h_num, '--', label=f'$H_{clabel}$ (num)', alpha=0.8)

        ax.set_ylabel(r'$H_\mathrm{mr}$ (T)')
        ax.legend(fontsize=6.5, ncol=3, loc='lower right', frameon=False)
        ps.add_panel_label(ax, panel_labels[i])

    axes[-1].set_xlabel(r'$x$ (nm)')
    fig.tight_layout()
    save_fig(fig, 'fig_rotation_field_validation')


# ======================================================================
# SIM 06 — Nonreciprocal SAW
# ======================================================================
def regen_sim06():
    print("\n=== Regenerating sim06: fig_nonreciprocal_saw ===")
    cache = np.load(os.path.join(DATA_DIR, 'sim06_data.npy'),
                    allow_pickle=True).item()

    m_plus  = cache['m_plus']
    m_minus = cache['m_minus']
    times   = cache['times']
    m_plus_nokmr = cache.get('m_plus_nokmr', None)

    # Vertical layout (single column)
    fig, axes = ps.double_panel_v(height_cm=10.0, hspace=0.35)

    t_ns = times * 1e9

    # Panel (a): time traces
    axes[0].plot(t_ns, m_plus * 1e3, '-', color=ps.BLUE,
                 label=r'SAW $+x$ (with $K_\mathrm{mr}$)')
    axes[0].plot(t_ns, m_minus * 1e3, '-', color=ps.VERMILION,
                 label=r'SAW $-x$ (with $K_\mathrm{mr}$)')
    if m_plus_nokmr is not None:
        axes[0].plot(t_ns, m_plus_nokmr * 1e3, '--', color='gray',
                     label=r'SAW $+x$ (no $K_\mathrm{mr}$)')

    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$|\delta m_\perp|$ ($\times 10^{-3}$)')
    axes[0].legend(fontsize=6.5, loc='upper left', frameon=False)
    ps.add_panel_label(axes[0], '(a)')

    # Panel (b): nonreciprocity ratio
    eps = 1e-30
    ratio = (m_plus + eps) / (m_minus + eps)
    axes[1].plot(t_ns, ratio, '-', color=ps.TEAL)
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel(r'$t$ (ns)')
    axes[1].set_ylabel(r'$|\delta m(+x)| / |\delta m(-x)|$')
    axes[1].set_ylim(0, max(3, np.max(ratio[len(ratio)//4:])))
    ps.add_panel_label(axes[1], '(b)')

    save_fig(fig, 'fig_nonreciprocal_saw')


# ======================================================================
# SIM 07 — Barnett field validation
# ======================================================================
def regen_sim07():
    print("\n=== Regenerating sim07: fig_barnett_field_validation ===")
    cache = np.load(os.path.join(DATA_DIR, 'sim07_data.npy'),
                    allow_pickle=True).item()
    results = cache['results']

    ps.apply_style()
    n = len(results)
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(n, 1,
                             figsize=(ps.SINGLE_COL,
                                      2.2 * n * ps.CM_TO_INCH * 2.54),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    for i, (ax, res) in enumerate(zip(axes, results)):
        x = res['x']
        for c, clabel in enumerate(['x', 'y', 'z']):
            h_num = res['H_num'][c, 0, 0, :]
            h_ana = res['H_ana'][c, 0, 0, :]
            if np.max(np.abs(h_ana)) < 1e-30 and np.max(np.abs(h_num)) < 1e-30:
                continue
            ax.plot(x, h_ana, '-', label=f'$H_{clabel}$ (ana)', alpha=0.8)
            ax.plot(x, h_num, '--', label=f'$H_{clabel}$ (num)', alpha=0.8)

        ax.set_ylabel(r'$H_\mathrm{Barnett}$ (T)')
        ax.legend(fontsize=6.5, ncol=3, loc='lower right', frameon=False)
        ps.add_panel_label(ax, panel_labels[i])

    axes[-1].set_xlabel(r'$x$ (nm)')
    fig.tight_layout()
    save_fig(fig, 'fig_barnett_field_validation')


# ======================================================================
# SIM 08 — Standing SAW spatially resolved coupling
# ======================================================================
def regen_sim08():
    print("\n=== Regenerating sim08: fig_standing_saw_ep ===")
    import json

    raw_dir = os.path.join(DATA_DIR, 'sim08_standing_ep')
    manifest = json.load(open(os.path.join(raw_dir, 'manifest.json')))

    # Parameters
    MU0 = 4e-7 * np.pi
    GAMMA = 1.7595e11
    MSAT = 1.2e6
    B1 = -8.8e6
    K_MR = manifest['K_mr']
    F_SAW = manifest['f_SAW']
    NX = manifest['NX']
    CX = manifest['CX']
    V_SAW = 3500.0
    LAMBDA_SAW = V_SAW / F_SAW
    B0 = 50e-3
    k = 2 * np.pi / LAMBDA_SAW
    L_total = NX * CX
    x_pos = np.arange(NX) * CX

    # Load raw data
    d_full = np.load(os.path.join(raw_dir, 'raw_full.npz'))
    d_mel = np.load(os.path.join(raw_dir, 'raw_mel.npz'))
    d_mr = np.load(os.path.join(raw_dir, 'raw_mr.npz'))

    my_full = d_full['m_y']
    my_mel = d_mel['m_y']
    my_mr = d_mr['m_y']
    time_arr = d_full['time']
    dt = time_arr[1] - time_arr[0]

    # Local spectral analysis (simplified)
    f_kittel = GAMMA * np.sqrt(B0 * (B0 + MU0 * MSAT)) / (2 * np.pi)

    def compute_spectra(my_data):
        """Compute power spectrum at each x position."""
        nt, nx = my_data.shape
        freqs = np.fft.rfftfreq(nt, d=dt)
        spectra = np.zeros((nx, len(freqs)))
        for ix in range(nx):
            ft = np.fft.rfft(my_data[:, ix])
            spectra[ix] = np.abs(ft)**2
        return freqs, spectra

    freqs, spec_full = compute_spectra(my_full)
    _, spec_mel = compute_spectra(my_mel)

    # Plot — double-column width for proper label sizing
    ps.apply_style()
    fig, axes = plt.subplots(2, 2,
                              figsize=(ps.DOUBLE_COL, 12.0 * ps.CM_TO_INCH))
    fig.subplots_adjust(wspace=0.35, hspace=0.55)
    for row in axes:
        for ax in row:
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    f_ghz = freqs * 1e-9
    f_max = min(3 * f_kittel * 1e-9, f_ghz[-1])
    f_min = max(0.1, f_ghz[1])
    f_mask = (f_ghz >= f_min) & (f_ghz <= f_max)
    x_um = x_pos * 1e6

    # Use same vmax for (a) and (b) — normalize to full coupling max
    S_full = spec_full[:, f_mask].T
    S_mel = spec_mel[:, f_mask].T
    global_max = np.max(S_full)
    if global_max > 0:
        S_full_norm = S_full / global_max
        S_mel_norm = S_mel / global_max
    else:
        S_full_norm = S_full
        S_mel_norm = S_mel
    ext = [x_um[0], x_um[-1], f_ghz[f_mask][0], f_ghz[f_mask][-1]]

    # (a) Full coupling
    axes[0, 0].imshow(S_full_norm, aspect='auto', origin='lower',
                      extent=ext, cmap='inferno', vmin=0, vmax=0.5)
    axes[0, 0].set_title('MEL + MR + Barnett', fontsize=9)
    axes[0, 0].set_xlabel(r'$x$ ($\mu$m)')
    axes[0, 0].set_ylabel(r'$f$ (GHz)')
    ps.add_panel_label(axes[0, 0], '(a)')

    # (b) MEL only — identically zero (dramatic contrast)
    axes[0, 1].imshow(S_mel_norm, aspect='auto', origin='lower',
                      extent=ext, cmap='inferno', vmin=0, vmax=0.5)
    axes[0, 1].set_title('MEL only', fontsize=9)
    axes[0, 1].set_xlabel(r'$x$ ($\mu$m)')
    axes[0, 1].set_ylabel(r'$f$ (GHz)')
    ps.add_panel_label(axes[0, 1], '(b)')

    # (c) Analytical coupling decomposition
    g_mel_norm = np.abs(np.cos(k * x_pos))
    g_mr_norm = np.abs(np.sin(k * x_pos))
    g_total = np.sqrt(g_mel_norm**2 + (K_MR / abs(B1))**2 * g_mr_norm**2)

    axes[1, 0].plot(x_um, g_mel_norm, color=ps.BLUE, linewidth=1.0,
                    label=r'$|g_\mathrm{MEL}|$')
    axes[1, 0].plot(x_um, g_mr_norm * K_MR / abs(B1),
                    color=ps.VERMILION, linewidth=1.0,
                    label=r'$|g_\mathrm{MR}|$')
    axes[1, 0].plot(x_um, g_total, '--', color='black', linewidth=0.8,
                    label=r'$g_\mathrm{total}$')

    for n in range(5):
        x_node = n * LAMBDA_SAW / 2
        if x_node <= L_total:
            axes[1, 0].axvline(x_node * 1e6, color=ps.BLUE,
                               linewidth=0.3, linestyle=':')
        x_rot = (n + 0.5) * LAMBDA_SAW / 2
        if x_rot <= L_total:
            axes[1, 0].axvline(x_rot * 1e6, color=ps.VERMILION,
                               linewidth=0.3, linestyle=':')

    axes[1, 0].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 0].set_ylabel('Coupling (norm.)')
    axes[1, 0].legend(fontsize=7, ncol=3, loc='lower center',
                      bbox_to_anchor=(0.5, 1.01), frameon=False)
    ps.add_panel_label(axes[1, 0], '(c)')

    # (d) Position-resolved max amplitude
    max_my_full = np.max(np.abs(my_full), axis=0)
    max_my_mel = np.max(np.abs(my_mel), axis=0)
    max_my_mr = np.max(np.abs(my_mr), axis=0)

    axes[1, 1].plot(x_um, max_my_full, color=ps.TEAL, linewidth=0.8,
                    label='Full')
    axes[1, 1].plot(x_um, max_my_mel, ':', color=ps.BLUE, linewidth=0.8,
                    label='MEL')
    axes[1, 1].plot(x_um, max_my_mr, '--', color=ps.VERMILION,
                    linewidth=0.8, label='MR')
    axes[1, 1].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 1].set_ylabel(r'max$|m_y|$')
    axes[1, 1].legend(fontsize=7, ncol=3, loc='lower center',
                      bbox_to_anchor=(0.5, 1.01), frameon=False)
    ps.add_panel_label(axes[1, 1], '(d)')

    save_fig(fig, 'fig_standing_saw_ep')


# ======================================================================
# Run all
# ======================================================================
if __name__ == '__main__':
    print("Regenerating figures with PRL style (frameon=False)...")
    regen_sim01()
    regen_sim02()
    regen_sim03()
    regen_sim05()
    regen_sim06()
    regen_sim07()
    regen_sim08()
    print("\nDone. All figures regenerated.")
