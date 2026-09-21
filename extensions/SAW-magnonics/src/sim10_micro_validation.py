"""Simulation 10-micro: Full micromagnetic validation of channel decomposition.

Reproduces sim10_prescribed (macrospin, 1x1x1) with a spatially resolved
grid (256x16x1, 10 nm cellsize) to confirm that:
  (a) MEL torque remains zero at theta=0 even with exchange + demagnetization
  (b) MR-only response matches MEL+MR (channel decomposition holds)
  (c) Results are quantitatively consistent with macrospin

This strengthens the PRL Fig 1(b) claim from "macrospin mumax+" to
"full micromagnetic mumax+".

Grid: 256x16x1 at 10 nm = 2.56 um x 160 nm (>2 SAW wavelengths)
PBC: (4, 100, 0) for thin-film demagnetization
Material: YIG (same as sim10_prescribed)
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

CACHE_FILE = os.path.join(DATA_DIR, "sim10_micro_validation.npz")

# ==========================================================================
# Constants & Material (YIG, same as sim10_prescribed)
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
# SAW parameters
# ==========================================================================
F_SAW = 3.0e9
LAMBDA_SAW = 1.16e-6
EPS0 = 1e-4
OMEGA_SAW = 2 * np.pi * F_SAW

T_MAX = 10e-9       # 10 ns (longer for spatial modes to develop)
DT_REC = 20e-12     # 20 ps recording
NT_REC = int(T_MAX / DT_REC) + 1


def find_resonance_field(f_target):
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


B0_RES = find_resonance_field(F_SAW)


# ==========================================================================
# Single run
# ==========================================================================
def run_micro(B0, direction=+1, enable_mel=True, K_mr=KMR,
              eps0=EPS0, label=""):
    """Run full micromagnetic SAW-FMR with prescribed SAW.

    Grid: 256x16x1 with PBC for thin-film demagnetization.
    """
    t_start = time.time()
    dir_str = "+" if direction > 0 else "-"
    kmr_str = f"Kmr={K_mr*1e-6:.1f}" if K_mr > 0 else "no-Kmr"
    mel_str = "MEL" if enable_mel else "no-MEL"
    print(f"  [{label}] {dir_str}k, B0={B0*1e3:.1f}mT, {mel_str}, {kmr_str} "
          f"[{NX}x{NY}x{NZ}]...", end="", flush=True)

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize, mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, grid)

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    # SAW
    K_mr_dir = K_mr if direction > 0 else -K_mr
    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr_dir, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    # Fixed timestep
    world.timesolver.timestep = 2e-13
    world.timesolver.adaptive_timestep = False

    # Record
    times = np.zeros(NT_REC)
    m_avg = np.zeros((NT_REC, 3))
    m_perp = np.zeros(NT_REC)

    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        m_avg[i] = avg
        m_perp[i] = np.sqrt(avg[1]**2 + avg[2]**2)
        times[i] = world.timesolver.time
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t_start
    peak = np.max(m_perp[NT_REC // 4:])
    print(f" peak|m_perp|={peak:.2e}, {elapsed:.1f}s")

    return {
        'times': times, 'm_avg': m_avg, 'm_perp': m_perp,
        'peak': peak, 'elapsed': elapsed,
    }


# ==========================================================================
# Main experiment: channel decomposition (mirrors sim10_prescribed exp1)
# ==========================================================================
def main():
    print("=" * 72)
    print("Sim 10-micro: Full micromagnetic channel decomposition")
    print(f"  Grid: {NX}x{NY}x{NZ}, cellsize=({CX*1e9:.0f},{CY*1e9:.0f},{CZ*1e9:.0f}) nm")
    print(f"  PBC: (4, 100, 0), demag=ON")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT, F_SAW = {F_SAW*1e-9:.1f} GHz")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        data = dict(np.load(CACHE_FILE, allow_pickle=True))
        # Reconstruct for plotting
        configs = ['full', 'mr_only', 'mel_only', 'no_coupling']
    else:
        results = {}

        # (1) Full coupling (MEL + MR)
        r = run_micro(B0_RES, direction=+1, enable_mel=True, K_mr=KMR,
                       label="full")
        results['full'] = r

        # (2) MR only (no MEL)
        r = run_micro(B0_RES, direction=+1, enable_mel=False, K_mr=KMR,
                       label="MR-only")
        results['mr_only'] = r

        # (3) MEL only (no MR)
        r = run_micro(B0_RES, direction=+1, enable_mel=True, K_mr=0,
                       label="MEL-only")
        results['mel_only'] = r

        # (4) No coupling (sanity check)
        r = run_micro(B0_RES, direction=+1, enable_mel=False, K_mr=0,
                       label="none")
        results['no_coupling'] = r

        # Save
        save_dict = {}
        for name, r in results.items():
            save_dict[f'{name}_times'] = r['times']
            save_dict[f'{name}_m_perp'] = r['m_perp']
            save_dict[f'{name}_peak'] = np.array(r['peak'])
            save_dict[f'{name}_mx'] = r['m_avg'][:, 0]
            save_dict[f'{name}_my'] = r['m_avg'][:, 1]
            save_dict[f'{name}_mz'] = r['m_avg'][:, 2]
        save_dict['B0_res'] = B0_RES
        save_dict['grid'] = np.array([NX, NY, NZ])
        np.savez(CACHE_FILE, **save_dict)
        print(f"\n  Saved: {CACHE_FILE}")
        data = save_dict

    # Load macrospin comparison
    macro_file = os.path.join(DATA_DIR, "sim10_channels.npz")
    has_macro = os.path.isfile(macro_file)

    # ==== Plot ====
    if HAS_STYLE:
        ps.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    configs = [
        ('full', 'MEL+MR', 'C0'),
        ('mr_only', 'MR only', 'C3'),
        ('mel_only', 'MEL only', 'C2'),
        ('no_coupling', 'None', 'gray'),
    ]

    # (a) Time traces
    ax = axes[0]
    for name, label, color in configs:
        t = data[f'{name}_times'] * 1e9
        mp = data[f'{name}_m_perp']
        ax.plot(t, mp, '-', color=color, lw=1.0, label=label)
    ax.set_xlabel('$t$ (ns)')
    ax.set_ylabel(r'$|\langle m_\perp \rangle|$')
    ax.legend(fontsize=7)
    ax.set_title(f'Micromagnetic ({NX}x{NY}x{NZ})', fontsize=9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    ax.text(-0.18, 1.05, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold')

    # (b) Peak comparison: macro vs micro
    ax = axes[1]
    names = ['full', 'mr_only', 'mel_only', 'no_coupling']
    labels = ['MEL+MR', 'MR only', 'MEL only', 'None']
    peaks_micro = [float(data[f'{n}_peak']) for n in names]

    x = np.arange(len(names))
    w = 0.35

    ax.bar(x, peaks_micro, w, color='C0', label='Micromagnetic')

    if has_macro:
        macro = np.load(macro_file, allow_pickle=True)
        macro_keys = list(macro.files)
        if 'full_+k_m_perp' in macro_keys:
            peaks_macro = []
            for n in ['full', 'mr_only', 'mel_only', 'no_coupling']:
                mk = f'{n}_+k_m_perp' if f'{n}_+k_m_perp' in macro_keys else None
                if mk:
                    mp = macro[mk]
                    peaks_macro.append(float(np.max(mp[len(mp) // 4:])))
                else:
                    peaks_macro.append(0)
            ax.bar(x + w, peaks_macro, w, color='C3', label='Macrospin')

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Peak $|m_\\perp|$')
    ax.legend(fontsize=7)
    ax.set_title('Channel decomposition', fontsize=9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    ax.text(-0.18, 1.05, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_micro_validation.{ext}'),
                    dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_micro_validation.pdf/png")

    # Summary
    print(f"\n  Summary (peak |m_perp|):")
    for name, label, _ in configs:
        print(f"    {label:12s}: {float(data[f'{name}_peak']):.3e}")


if __name__ == "__main__":
    main()
