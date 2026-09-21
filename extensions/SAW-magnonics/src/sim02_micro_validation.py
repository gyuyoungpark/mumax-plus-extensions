"""Simulation 02-micro: Full micromagnetic SAW-FMR spectroscopy.

Replaces the macrospin sim02 (4x4x1) with a spatially resolved grid
and corrected parameters:
  - Grid: 128x16x1 at 10 nm (micromagnetic with exchange + demag)
  - eps0: 1e-4 (linear regime; original 1e-3 caused nonlinear 2x peaks)
  - Material: YIG (consistent with PRL paper)
  - SAW: ChiralSurfaceAcousticWave (includes MEL + MR)

Protocol:
  Sweep f_SAW at 3 B0 values. For each (B0, f_SAW):
    (a) Full coupling (MEL+MR) -> record peak |m_perp|
    (b) MR-only -> confirm identical response at theta=0

Panel (a): FMR absorption spectra for different B0
Panel (b): Peak frequency vs B0 with Kittel curve overlay
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

CACHE_FILE = os.path.join(DATA_DIR, "sim02_micro_validation.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim02_micro_checkpoint.npz")

# ==========================================================================
# Material: YIG (matching PRL paper)
# ==========================================================================
GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

# ==========================================================================
# Geometry: FULL MICROMAGNETIC
# ==========================================================================
NX, NY, NZ = 128, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

# ==========================================================================
# SAW & sweep parameters
# ==========================================================================
EPS0 = 1e-4            # LINEAR regime (was 1e-3 in original -> nonlinear)
T_RUN = 10e-9          # 10 ns run per point
DT_REC = 20e-12        # 20 ps recording
NT_REC = int(T_RUN / DT_REC) + 1

B0_VALUES = np.array([20e-3, 50e-3, 100e-3])
F_SAW_VALUES = np.linspace(1e9, 8e9, 25)


def kittel_freq_hz(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


# ==========================================================================
# Single point
# ==========================================================================
def run_single(B0, f_saw, enable_mel=True, K_mr=KMR, label=""):
    t_start = time.time()
    wavelength = V_SAW / f_saw

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize, mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, grid)

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = 2e-13
    world.timesolver.adaptive_timestep = False

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        m_perp_arr[i] = np.sqrt(avg[1]**2 + avg[2]**2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_half = NT_REC // 2
    m_ss = np.mean(m_perp_arr[n_half:])
    elapsed = time.time() - t_start
    return m_ss, elapsed


# ==========================================================================
# Main sweep
# ==========================================================================
def main():
    print("=" * 72)
    print("Sim 02-micro: Full micromagnetic SAW-FMR spectroscopy")
    print(f"  Grid: {NX}x{NY}x{NZ}, PBC=(4,100,0), demag=ON")
    print(f"  eps0 = {EPS0:.0e} (linear regime)")
    print(f"  B0 sweep: {B0_VALUES*1e3} mT")
    print(f"  f_SAW sweep: {F_SAW_VALUES[0]*1e-9:.1f} - {F_SAW_VALUES[-1]*1e-9:.1f} GHz "
          f"({len(F_SAW_VALUES)} points)")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        data = dict(np.load(CACHE_FILE))
        spectra_full = data['spectra_full']
        spectra_mr = data['spectra_mr']
    else:
        # Load checkpoint
        ckpt = {}
        if os.path.isfile(CHECKPOINT):
            ckpt = dict(np.load(CHECKPOINT))
            print(f"  Loaded checkpoint")

        spectra_full = np.zeros((len(B0_VALUES), len(F_SAW_VALUES)))
        spectra_mr = np.zeros((len(B0_VALUES), len(F_SAW_VALUES)))

        for bi, B0 in enumerate(B0_VALUES):
            B0_mT = B0 * 1e3
            f_kittel = kittel_freq_hz(B0)
            key_full = f'full_{B0_mT:.0f}'
            key_mr = f'mr_{B0_mT:.0f}'

            if key_full in ckpt and key_mr in ckpt:
                spectra_full[bi] = ckpt[key_full]
                spectra_mr[bi] = ckpt[key_mr]
                print(f"\n  B0 = {B0_mT:.0f} mT [checkpoint]  "
                      f"Kittel = {f_kittel*1e-9:.2f} GHz")
                continue

            print(f"\n  B0 = {B0_mT:.0f} mT  (Kittel = {f_kittel*1e-9:.2f} GHz)")

            for j, f_saw in enumerate(F_SAW_VALUES):
                pct = 100 * (j + 1) / len(F_SAW_VALUES)

                # Full coupling
                m_ss, dt = run_single(B0, f_saw, enable_mel=True, K_mr=KMR)
                spectra_full[bi, j] = m_ss

                # MR only
                m_mr, _ = run_single(B0, f_saw, enable_mel=False, K_mr=KMR)
                spectra_mr[bi, j] = m_mr

                if (j + 1) % 5 == 0:
                    print(f"    [{pct:5.1f}%] f={f_saw*1e-9:.2f} GHz  "
                          f"|m_perp| full={m_ss:.2e}  MR={m_mr:.2e}  "
                          f"({dt:.1f}s)")

            ckpt[key_full] = spectra_full[bi]
            ckpt[key_mr] = spectra_mr[bi]
            np.savez(CHECKPOINT, **ckpt)

        np.savez(CACHE_FILE,
                 B0_values=B0_VALUES,
                 f_saw_values=F_SAW_VALUES,
                 spectra_full=spectra_full,
                 spectra_mr=spectra_mr)
        if os.path.isfile(CHECKPOINT):
            os.remove(CHECKPOINT)
        print(f"\n  Saved: {CACHE_FILE}")
        data = {'spectra_full': spectra_full, 'spectra_mr': spectra_mr}

    # ==== Plot ====
    if HAS_STYLE:
        ps.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # (a) FMR absorption spectra (full coupling)
    ax = axes[0]
    colors = ['C0', 'C3', 'C2']
    f_ghz = F_SAW_VALUES * 1e-9
    for bi, B0 in enumerate(B0_VALUES):
        spec = spectra_full[bi]
        if np.max(spec) > 0:
            spec_norm = spec / np.max(spec)
        else:
            spec_norm = spec
        offset = bi * 1.3
        ax.fill_between(f_ghz, offset, offset + spec_norm * 0.9,
                         color=colors[bi], alpha=0.5)
        ax.plot(f_ghz, offset + spec_norm * 0.9, color=colors[bi], lw=0.8)

        f_k = kittel_freq_hz(B0) * 1e-9
        ax.axvline(f_k, color=colors[bi], ls=':', lw=0.5, alpha=0.6)
        ax.text(max(f_ghz) * 0.75, offset + 1.0,
                f'$B_0={B0*1e3:.0f}$ mT', fontsize=6, color=colors[bi])

    ax.set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    ax.set_ylabel('$|m_\\perp|$ (a.u., offset)')
    ax.set_yticks([])
    ax.set_title(f'Micromagnetic ({NX}x{NY}x{NZ})', fontsize=9)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold')

    # (b) Peak freq vs B0 + Kittel curve + MEL vs MR comparison
    ax = axes[1]
    peak_full = np.zeros(len(B0_VALUES))
    peak_mr = np.zeros(len(B0_VALUES))

    for bi in range(len(B0_VALUES)):
        for spec, arr in [(spectra_full[bi], peak_full),
                          (spectra_mr[bi], peak_mr)]:
            if np.max(spec) > 1e-6:
                pks, _ = find_peaks(spec, height=0.3 * np.max(spec))
                if len(pks) > 0:
                    arr[bi] = F_SAW_VALUES[pks[np.argmax(spec[pks])]] * 1e-9
                else:
                    arr[bi] = np.nan
            else:
                arr[bi] = np.nan

    valid_f = ~np.isnan(peak_full)
    valid_m = ~np.isnan(peak_mr)

    ax.plot(B0_VALUES[valid_f] * 1e3, peak_full[valid_f], 'o',
            color='C0', ms=7, mec='black', mew=0.3, label='MEL+MR')
    ax.plot(B0_VALUES[valid_m] * 1e3, peak_mr[valid_m], 's',
            color='C3', ms=6, mec='black', mew=0.3, label='MR only')

    B_fit = np.linspace(5, 120, 200) * 1e-3
    f_k = np.array([kittel_freq_hz(b) * 1e-9 for b in B_fit])
    ax.plot(B_fit * 1e3, f_k, 'k--', lw=0.8, label='Kittel')

    ax.set_xlabel(r'$B_0$ (mT)')
    ax.set_ylabel(r'$f_\mathrm{peak}$ (GHz)')
    ax.legend(fontsize=7)
    ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_micro_fmr.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_micro_fmr.pdf/png")

    # Summary
    print(f"\n  Peak frequencies (GHz):")
    print(f"  {'B0 (mT)':>10s}  {'Kittel':>8s}  {'Full':>8s}  {'MR-only':>8s}")
    for bi, B0 in enumerate(B0_VALUES):
        f_k = kittel_freq_hz(B0) * 1e-9
        print(f"  {B0*1e3:>8.0f}  {f_k:>8.2f}  "
              f"{peak_full[bi]:>8.2f}  {peak_mr[bi]:>8.2f}")


if __name__ == "__main__":
    main()
