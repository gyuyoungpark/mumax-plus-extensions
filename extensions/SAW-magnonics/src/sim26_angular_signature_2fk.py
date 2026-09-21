"""Simulation 26: Angular dependence of the parametric (2 f_K) peak.

Answers referee question: "What experimental signature distinguishes
your geometry-locked SAW pumping from conventional Suhl pumping?"

Predicted falsifiable signature:
  Parametric MEL pump amplitude scales as cos^2(theta), vanishing at
  theta = 90 deg (m_0 perpendicular to k_SAW), because the MEL energy
  is B_1 * eps_xx * m_x^2 with m_x = cos(theta).

Suhl-type parallel pumping (uniform rf field h_z parallel to m_0) has
no angular dependence with respect to k_pump direction, since h is
spatially uniform.

Sweeps theta in {0, 15, 30, 45, 60, 75, 89} deg at f_SAW = 2 f_K.
Records peak |m_perp| for MEL-only and Full configurations.
Estimated runtime: ~30 minutes.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim26_angular_2fk.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9
EPS0 = 1e-4
B0 = 50e-3

T_RUN = 10e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

THETAS_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0])


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_one(theta_deg, enable_mel, K_mr, label=""):
    t0 = time.time()
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw
    th = np.radians(theta_deg)
    mx0, my0 = np.cos(th), np.sin(th)
    Bx, By = B0 * np.cos(th), B0 * np.sin(th)

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (mx0, my0, 0.005)
    magnet.bias_magnetic_field = (Bx, By, 0)
    magnet.enable_demag = True
    if enable_mel:
        magnet.B1 = B1

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        # Component perpendicular to m_0 in the film plane
        m_par = avg[0] * mx0 + avg[1] * my0
        m_perp_in = -avg[0] * my0 + avg[1] * mx0
        m_perp_arr[i] = np.sqrt(m_perp_in ** 2 + avg[2] ** 2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_half = NT_REC // 2
    peak = float(np.max(m_perp_arr[n_half:]))
    elapsed = time.time() - t0
    print(f"  [{label}] theta={theta_deg:5.1f}deg peak={peak:.3e} ({elapsed:.0f}s)")
    return peak


def main():
    print("=" * 60)
    print("Sim 26: Angular dependence at f_SAW = 2 f_K")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz")
    print(f"  Predict: |m_perp|^MEL ~ cos^2(theta)")
    print("=" * 60)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading.")
        data = dict(np.load(CACHE_FILE))
    else:
        peaks_mel = np.zeros_like(THETAS_DEG)
        peaks_full = np.zeros_like(THETAS_DEG)
        for i, th in enumerate(THETAS_DEG):
            peaks_mel[i] = run_one(th, enable_mel=True, K_mr=0,
                                    label="MEL")
            peaks_full[i] = run_one(th, enable_mel=True, K_mr=KMR,
                                     label="Full")
        data = {'thetas_deg': THETAS_DEG,
                'peaks_mel': peaks_mel,
                'peaks_full': peaks_full,
                'B0': B0, 'f_K': F_K}
        np.savez(CACHE_FILE, **data)
        print(f"  Saved: {CACHE_FILE}")

    plot_angular(data)


def plot_angular(data):
    try:
        from plot_style import (apply_style, axis_label, SINGLE_COL,
                                SKY_BLUE, VERMILION, BLACK)
        apply_style()
    except ImportError:
        SINGLE_COL = 3.4
        SKY_BLUE, VERMILION, BLACK = '#56B4E9', '#D55E00', '#000'

    thetas = data['thetas_deg']
    p_mel = data['peaks_mel']
    p_full = data['peaks_full']

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85))
    ax.plot(thetas, p_mel / p_mel[0], 'o-', color=VERMILION, ms=6,
            mec='k', mew=0.3, label=r'MEL only @ $2f_K$')
    ax.plot(thetas, p_full / p_full[0], 's--', color=BLACK, ms=5,
            mec='k', mew=0.3, label=r'Full @ $2f_K$', alpha=0.7)

    ax.axhline(1.0, color='gray', ls=':', lw=0.4, alpha=0.5)
    ax.axhspan(-0.05, 0.12, alpha=0.06, color='green')
    ax.text(50, 0.95, 'Suhl pumping\n(angle-independent)',
            fontsize=6, color='gray', ha='center', style='italic')

    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'Normalized peak $|m_\perp|$ at $2f_K$')
    ax.set_xlim(-2, 92)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc='center right')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim26_angular_2fk.{ext}'),
                    dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim26_angular_2fk.pdf/png")


if __name__ == "__main__":
    main()
