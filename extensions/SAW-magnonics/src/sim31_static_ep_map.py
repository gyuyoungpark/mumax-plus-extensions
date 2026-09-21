"""Simulation 31 (Paper 2): Static exceptional point map.

Analytic eigenvalue scan of the 2x2 non-Hermitian SAW-magnon Hamiltonian

    H(eps0, B0) = [ omega_m(B0) - i*kappa_m,  g(eps0)  ]
                  [ g*(eps0),                  omega_p - i*kappa_p ]

with chiral coupling

    g(eps0) = g_mr * eps0,            (real, MR channel at theta=0)
    g_mr   = (gamma * K_mr * xi) / (2 * Ms),

over a 200x200 grid in (eps0, B0).  The EP is the coalescence point
where both eigenvalues and eigenvectors merge,

    Re[Delta omega] = 0   AND   Im[Delta omega] = 0
    <=> g(eps0) = (kappa_p - kappa_m) / 2  AND  omega_m(B0) = omega_p.

Outputs:
  - data/sim31_ep_map.npz: eigenvalue grid + EP coordinates
  - figures/fig_sim31_ep_map.pdf: Re/Im gap maps + Riemann surface

Estimated runtime: < 1 minute (analytic).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim31_ep_map.npz")

# YIG parameters (sim20 unified)
GAMMA = 1.76e11
MS = 140e3
MU0 = 4 * np.pi * 1e-7
ALPHA = 5e-4
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0
F_REF = 3.0e9                  # SAW frequency = phonon mode freq
WAVELENGTH = V_SAW / F_REF

OMEGA_P = 2 * np.pi * F_REF
KAPPA_P = 2 * np.pi * 5e6      # phonon decay (5 MHz, soft-coupled)
KAPPA_M = ALPHA * OMEGA_P      # magnon decay (Gilbert)


def kittel_omega(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS))


def coupling(eps0):
    return GAMMA * KMR * XI / (2 * MS) * eps0


def H_eff(eps0, B0):
    om = kittel_omega(B0)
    g = coupling(eps0)
    H = np.array([[om - 1j * KAPPA_M, g],
                  [g, OMEGA_P - 1j * KAPPA_P]], dtype=complex)
    return H


def eigenvalues(eps0, B0):
    om = kittel_omega(B0)
    g = coupling(eps0)
    a = om - 1j * KAPPA_M
    b = OMEGA_P - 1j * KAPPA_P
    delta = (a - b) / 2
    avg = (a + b) / 2
    disc = np.sqrt(delta ** 2 + g ** 2)
    return avg + disc, avg - disc


def main():
    print("=" * 72)
    print("Sim 31 (Paper 2): Static EP map (analytic)")
    print(f"  YIG, omega_p/2pi = {F_REF*1e-9:.2f} GHz")
    print(f"  kappa_m/2pi = {KAPPA_M/(2*np.pi)*1e-6:.2f} MHz")
    print(f"  kappa_p/2pi = {KAPPA_P/(2*np.pi)*1e-6:.2f} MHz")
    print("=" * 72)

    # EP from analytic formula
    g_EP = abs(KAPPA_P - KAPPA_M) / 2
    eps0_EP = g_EP / (GAMMA * KMR * XI / (2 * MS))
    omega_K2 = OMEGA_P
    B0_EP = (np.sqrt((omega_K2 / GAMMA) ** 2 +
             (MU0 * MS / 2) ** 2) - MU0 * MS / 2)
    print(f"  Predicted EP at:")
    print(f"    eps0_EP = {eps0_EP:.3e}")
    print(f"    B0_EP  = {B0_EP*1e3:.3f} mT")
    print(f"    g_EP/2pi = {g_EP/(2*np.pi)*1e-6:.3f} MHz")

    if os.path.isfile(CACHE):
        print("  Cached. Loading...")
        d = dict(np.load(CACHE))
    else:
        n_e, n_b = 200, 200
        eps_arr = np.linspace(0.05 * eps0_EP, 4 * eps0_EP, n_e)
        B0_arr = np.linspace(0.6 * B0_EP, 1.4 * B0_EP, n_b)

        Re_gap = np.zeros((n_b, n_e))
        Im_gap = np.zeros((n_b, n_e))
        Re_avg = np.zeros((n_b, n_e))
        Im_avg = np.zeros((n_b, n_e))
        for j, b in enumerate(B0_arr):
            for i, e in enumerate(eps_arr):
                wp, wm = eigenvalues(e, b)
                Re_gap[j, i] = (wp - wm).real
                Im_gap[j, i] = (wp - wm).imag
                Re_avg[j, i] = ((wp + wm) / 2).real
                Im_avg[j, i] = ((wp + wm) / 2).imag

        d = {
            'eps_arr': eps_arr, 'B0_arr': B0_arr,
            'Re_gap': Re_gap, 'Im_gap': Im_gap,
            'Re_avg': Re_avg, 'Im_avg': Im_avg,
            'eps0_EP': eps0_EP, 'B0_EP': B0_EP,
            'g_EP': g_EP, 'kappa_m': KAPPA_M, 'kappa_p': KAPPA_P,
            'omega_p': OMEGA_P,
        }
        np.savez(CACHE, **d)
        print(f"  Saved: {CACHE}")
    plot(d)


def plot(d):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        DOUBLE_COL = 7.0

    eps = d['eps_arr']
    B0 = d['B0_arr'] * 1e3                 # mT
    eps_EP = float(d['eps0_EP'])
    B_EP = float(d['B0_EP']) * 1e3
    Re_g = d['Re_gap'] / (2 * np.pi) * 1e-6     # MHz
    Im_g = d['Im_gap'] / (2 * np.pi) * 1e-6     # MHz

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.4))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.97, top=0.92,
                        bottom=0.18)

    EE, BB = np.meshgrid(eps * 1e4, B0)        # x10^-4 strain

    ax = axes[0]
    im = ax.pcolormesh(EE, BB, np.abs(Re_g),
                       cmap='magma', shading='auto', rasterized=True,
                       vmin=0, vmax=np.percentile(np.abs(Re_g), 95))
    ax.contour(EE, BB, np.abs(Re_g), levels=[1.0],
               colors='white', linewidths=0.5)
    ax.plot(eps_EP * 1e4, B_EP, '+', color='cyan',
            ms=10, mew=1.3, label='EP')
    ax.set_xlabel(r'$\varepsilon_0$ ($10^{-4}$)')
    ax.set_ylabel(r'$B_0$ (mT)')
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r'$|\mathrm{Re}\,\Delta\omega|$ (MHz)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax = axes[1]
    im = ax.pcolormesh(EE, BB, np.abs(Im_g),
                       cmap='viridis', shading='auto', rasterized=True,
                       vmin=0, vmax=np.percentile(np.abs(Im_g), 95))
    ax.plot(eps_EP * 1e4, B_EP, '+', color='red',
            ms=10, mew=1.3, label='EP')
    ax.set_xlabel(r'$\varepsilon_0$ ($10^{-4}$)')
    ax.set_ylabel(r'$B_0$ (mT)')
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r'$|\mathrm{Im}\,\Delta\omega|$ (MHz)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    try:
        label_panels(axes)
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim31_ep_map.{ext}'), dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim31_ep_map.pdf/png")


if __name__ == "__main__":
    main()
