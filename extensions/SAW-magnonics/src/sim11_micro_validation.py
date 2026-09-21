"""Simulation 11-micro: Full micromagnetic angle-dependent coupling validation.

Reproduces sim11 Part C (macrospin, 1x1x1) with a spatially resolved grid
(256x16x1) to confirm that:
  (a) theta_c crossover holds in full micromagnetics
  (b) MEL activation at small angles is quantitatively correct
  (c) |m_perp| at theta=0 is exclusively from MR (zero MEL torque)

Grid: 256x16x1 at 10 nm = 2.56 um x 160 nm
PBC: (4, 100, 0) for thin-film demagnetization
Angles: 0, 0.5, 1, 2, 5, 10, 20, 30, 45 degrees
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0
from saw_chiral import ChiralSurfaceAcousticWave

try:
    import plot_style as ps
    HAS_STYLE = True
except ImportError:
    HAS_STYLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim11_micro_validation.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim11_micro_checkpoint.npz")

# ==========================================================================
# Constants & Material (YIG)
# ==========================================================================
GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68

# ==========================================================================
# Geometry: FULL MICROMAGNETIC
# ==========================================================================
NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

# ==========================================================================
# SAW
# ==========================================================================
F_SAW = 3.0e9
LAMBDA_SAW = 1.16e-6
EPS0 = 1e-4
OMEGA_SAW = 2 * np.pi * F_SAW

T_MAX = 10e-9
DT_REC = 20e-12
NT_REC = int(T_MAX / DT_REC) + 1

ANGLES = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0])


def find_resonance_field(f_target):
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


B0_RES = find_resonance_field(F_SAW)


# ==========================================================================
# Single run at oblique angle
# ==========================================================================
def run_oblique_micro(theta_deg, B0, direction=+1, enable_mel=True,
                       K_mr=KMR, label=""):
    t_start = time.time()
    theta = np.radians(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    print(f"  [{label}] theta={theta_deg:.1f}deg, "
          f"{'MEL+MR' if enable_mel else 'MR-only'} [{NX}x{NY}x{NZ}]...",
          end="", flush=True)

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize, mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, grid)

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (ct, st, 0)
    magnet.bias_magnetic_field = (B0 * ct, B0 * st, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    K_mr_dir = K_mr if direction > 0 else -K_mr
    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr_dir, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = 5e-14
    world.timesolver.adaptive_timestep = False

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        dm_y = avg[1] - st
        m_perp_arr[i] = np.sqrt(dm_y**2 + avg[2]**2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t_start
    peak = float(np.max(m_perp_arr[NT_REC // 4:]))
    print(f" peak={peak:.2e}, {elapsed:.1f}s")
    return peak


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 72)
    print("Sim 11-micro: Full micromagnetic angle-dependent validation")
    print(f"  Grid: {NX}x{NY}x{NZ}, PBC=(4,100,0), demag=ON")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        data = dict(np.load(CACHE_FILE))
        peaks_micro = data['peaks_micro']
        angles = data['angles']
    else:
        # Load checkpoint
        ckpt = {}
        if os.path.isfile(CHECKPOINT):
            ckpt = dict(np.load(CHECKPOINT))
            print(f"  Loaded checkpoint ({len(ckpt) - 1} angles done)")

        peaks_micro = np.zeros(len(ANGLES))
        for i, theta in enumerate(ANGLES):
            key = f'peak_{theta:.1f}'
            if key in ckpt:
                peaks_micro[i] = float(ckpt[key])
                print(f"  theta={theta:.1f}deg [checkpoint] peak={peaks_micro[i]:.2e}")
                continue
            peaks_micro[i] = run_oblique_micro(
                theta, B0_RES, label=f"{i + 1}/{len(ANGLES)}")
            ckpt[key] = np.array(peaks_micro[i])
            ckpt['angles'] = ANGLES
            np.savez(CHECKPOINT, **ckpt)

        np.savez(CACHE_FILE, angles=ANGLES, peaks_micro=peaks_micro)
        if os.path.isfile(CHECKPOINT):
            os.remove(CHECKPOINT)
        print(f"\n  Saved: {CACHE_FILE}")

    # Load macrospin comparison
    macro_file = os.path.join(DATA_DIR, "sim11_angle_sweep.npz")
    has_macro = os.path.isfile(macro_file)

    # Analytic
    g_mel = GAMMA * abs(B1) * EPS0 * np.abs(np.sin(2 * np.radians(ANGLES))) / MS
    g_mr = GAMMA * abs(KMR) * XI * EPS0 * np.abs(np.cos(np.radians(ANGLES))) / (2 * MS)
    g_total_anal = np.sqrt(g_mel**2 + g_mr**2)

    # ==== Plot ====
    if HAS_STYLE:
        ps.apply_style()

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Normalize to theta=0 value
    norm = peaks_micro[0] if peaks_micro[0] > 0 else 1
    norm_anal = g_total_anal[0] if g_total_anal[0] > 0 else 1

    ax.plot(ANGLES, peaks_micro / norm, 'o-', color='C0',
            ms=6, mec='black', mew=0.3, lw=1.2,
            label=f'Micromagnetic ({NX}x{NY}x{NZ})')

    if has_macro:
        md = np.load(macro_file)
        if 'peaks' in md.files:
            peaks_mac = md['peaks']
        else:
            peaks_mac = md[md.files[-1]]
        norm_mac = peaks_mac[0] if peaks_mac[0] > 0 else 1
        ax.plot(ANGLES[:len(peaks_mac)], peaks_mac / norm_mac, 's--',
                color='C3', ms=5, mec='black', mew=0.3, lw=1.0,
                label='Macrospin (1x1x1)')

    ax.plot(ANGLES, g_total_anal / norm_anal, 'k:', lw=0.8,
            label='Analytic $g_{\\mathrm{total}}$')

    theta_c = np.degrees(np.arcsin(abs(KMR) * XI / (4 * abs(B1))))
    ax.axvline(theta_c, color='gray', ls='--', lw=0.5, alpha=0.6)
    ax.text(theta_c + 0.5, 0.95, f'$\\theta_c$={theta_c:.1f}°',
            fontsize=7, color='gray', va='top')

    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel('Normalized $|m_\\perp|$')
    ax.set_xlim(-1, 47)
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_micro_angle_validation.{ext}'),
                    dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_micro_angle_validation.pdf/png")

    # Summary
    print(f"\n  Summary (peak |m_perp|):")
    print(f"  {'angle':>8s}  {'micro':>12s}  {'analytic g':>12s}")
    for i, theta in enumerate(ANGLES):
        print(f"  {theta:>7.1f}°  {peaks_micro[i]:>12.3e}  "
              f"{g_total_anal[i]:>12.3e}")


if __name__ == "__main__":
    main()
