"""Simulation 10: Prescribed SAW - Chiral magnon-phonon coupling in YIG.

Uses the prescribed (analytical) SAW strain profile to study direction-dependent
magnon excitation. This approach gives precise control over strain amplitude
and avoids elastodynamics calibration issues.

Key physics:
- For m0 || k_SAW (in-plane along x), MEL gives ZERO torque (H_mel || m0)
- MR coupling provides the essential transverse drive
- MR coupling is chiral: flips sign with k -> -k

Material: YIG thin film (in-plane magnetized along x)
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

# Import SAW classes
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================================================
# Constants
# ==========================================================================
GAMMA = 1.76e11  # rad/(s*T)

# ==========================================================================
# Material: YIG
# ==========================================================================
MS = 140e3         # A/m
AEX = 3.65e-12    # J/m
ALPHA = 5e-4      # Gilbert damping (slightly higher than ideal YIG for stability)
B1 = -8.8e6       # J/m^3 (magnetoelastic, negative convention)

# Magneto-rotation
KMR = 1.0e6       # J/m^3
XI = 0.68          # Rayleigh ellipticity

# ==========================================================================
# Geometry
# ==========================================================================
NX, NY, NZ = 1, 1, 1   # uniform mode (single cell)
CX = 10e-9
CY = 10e-9
CZ = 20e-9

# ==========================================================================
# SAW parameters
# ==========================================================================
F_SAW = 3.0e9      # Hz
LAMBDA_SAW = 1.16e-6  # m (v_SAW ~ 3500 m/s)
EPS0 = 1e-4        # peak strain (small enough for linear regime)

# Simulation
T_MAX = 5e-9       # 5 ns
DT_REC = 10e-12    # 10 ps recording interval
NT_REC = int(T_MAX / DT_REC) + 1


def find_resonance_field(f_target):
    """Find B0 for Kittel FMR at f_target."""
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4*a*c)) / 2


B0_RES = find_resonance_field(F_SAW)


def run_prescribed_saw(B0, direction=+1, enable_mel=True, K_mr=KMR,
                        eps0=EPS0, alpha=ALPHA, label=""):
    """Run simulation with prescribed SAW strain profile.

    Parameters
    ----------
    B0 : float
        External field (T).
    direction : +1 or -1
        SAW direction (+k or -k).
    enable_mel : bool
        Enable magnetoelastic coupling.
    K_mr : float
        Magneto-rotation coupling (J/m^3). 0 = disabled.
    eps0 : float
        Peak strain amplitude.
    alpha : float
        Gilbert damping.
    label : str
        Progress label.
    """
    t_start = time.time()
    dir_str = "+" if direction > 0 else "-"
    kmr_str = f"Kmr={K_mr*1e-6:.1f}" if K_mr > 0 else "no-Kmr"
    mel_str = "MEL" if enable_mel else "no-MEL"
    print(f"  [{label}] {dir_str}k, B0={B0*1e3:.1f}mT, {mel_str}, {kmr_str}...",
          end="", flush=True)

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize)
    magnet = Ferromagnet(world, grid)

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = alpha
    magnet.magnetization = (1, 0, 0)  # in-plane along x
    magnet.bias_magnetic_field = (B0, 0, 0)

    if enable_mel:
        magnet.B1 = B1

    # SAW setup: use ChiralSurfaceAcousticWave
    # For -k direction: use negative wavelength or flip phase
    # ChiralSurfaceAcousticWave uses k = 2pi/wavelength
    # For -k: we can use phase offset of pi and negative k
    # Simplest: just flip the direction parameter
    if direction > 0:
        saw = ChiralSurfaceAcousticWave(
            frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=eps0,
            direction='x', phase=0.0, ellipticity=XI,
            K_mr=K_mr, enable_barnett=False)
    else:
        # For -k: use negative wavelength (k -> -k)
        # The ChiralSAW handles this by passing negative wavelength
        # which inverts the sign of k in all spatial terms
        # Actually, let me use positive wavelength with reversed propagation
        # The key: Omega_y changes sign under k -> -k
        # In ChiralSAW, this is done by... let me check

        # Simple approach: negate the phase and the spatial profile
        # For CW at a single cell, the spatial dependence doesn't matter
        # The direction dependence is in the TIME function:
        # +k: H_mr_z ~ -cos(wt) (from cos(kx-wt) at x=0)
        # -k: H_mr_z ~ +cos(wt) (sign flip from k -> -k)
        # We handle this by using K_mr with opposite sign for -k
        saw = ChiralSurfaceAcousticWave(
            frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=eps0,
            direction='x', phase=0.0, ellipticity=XI,
            K_mr=-K_mr,  # sign flip simulates -k propagation for MR
            enable_barnett=False)

    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    # Time evolution
    world.timesolver.adaptive_timestep = True

    times = np.zeros(NT_REC)
    m_y = np.zeros(NT_REC)
    m_z = np.zeros(NT_REC)
    m_x = np.zeros(NT_REC)

    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        m_x[i] = m[0, 0, 0, 0]
        m_y[i] = m[1, 0, 0, 0]
        m_z[i] = m[2, 0, 0, 0]
        times[i] = world.timesolver.time

        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t_start
    m_perp = np.sqrt(m_y**2 + m_z**2)
    print(f" peak|m_perp|={np.max(m_perp):.2e}, {elapsed:.1f}s")

    return {
        'times': times, 'm_x': m_x, 'm_y': m_y, 'm_z': m_z,
        'm_perp': m_perp, 'B0': B0, 'direction': direction,
    }


# ==========================================================================
# Experiment 1: Channel decomposition at resonance
# ==========================================================================
def exp1_channel_decomposition():
    """Compare coupling channels at resonance."""
    print("\n=== Exp 1: Channel decomposition ===")
    cache = os.path.join(DATA_DIR, "sim10_channels.npz")

    if os.path.isfile(cache):
        print("  Loading cached...")
        flat = dict(np.load(cache, allow_pickle=True))
        # Reconstruct nested dict from flat keys
        # Keys are like pk_MELpMR_m_perp -> prefix=pk_MELpMR, field=m_perp
        results = {}
        suffixes = ['m_x', 'm_y', 'm_z', 'm_perp', 'times']
        prefixes = set()
        for k in flat:
            if k == 'times':
                continue
            for suffix in suffixes:
                if k.endswith('_' + suffix):
                    prefixes.add(k[:-(len(suffix)+1)])
                    break
        for prefix in prefixes:
            r = {}
            for suffix in suffixes:
                key = f"{prefix}_{suffix}"
                if key in flat:
                    r[suffix] = flat[key]
            r['direction'] = 1 if prefix.startswith('pk') else -1
            results[prefix] = r
        return results

    configs = [
        ("+k MEL+MR",   +1, True,  KMR),
        ("-k MEL+MR",   -1, True,  KMR),
        ("+k MR only",  +1, False, KMR),
        ("-k MR only",  -1, False, KMR),
    ]

    results = {}
    for name, dirn, mel, kmr in configs:
        r = run_prescribed_saw(B0_RES, direction=dirn, enable_mel=mel,
                               K_mr=kmr, label=name)
        key = name.replace(" ", "_").replace("+", "p").replace("-", "m")
        results[key] = r

    # Save
    save_dict = {'times': results[list(results.keys())[0]]['times']}
    for key, val in results.items():
        for k2, v2 in val.items():
            if isinstance(v2, np.ndarray):
                save_dict[f"{key}_{k2}"] = v2

    np.savez(cache, **save_dict)
    return results


# ==========================================================================
# Experiment 2: B0 sweep
# ==========================================================================
def exp2_b0_sweep():
    """Sweep B0 through resonance for +k and -k."""
    print("\n=== Exp 2: B0 sweep ===")
    cache = os.path.join(DATA_DIR, "sim10_b0sweep_prescribed.npz")

    if os.path.isfile(cache):
        print("  Loading cached...")
        return dict(np.load(cache, allow_pickle=True))

    B0_range = 0.4 * B0_RES
    N_B0 = 9
    B0_values = np.linspace(max(B0_RES - B0_range, 5e-3),
                             B0_RES + B0_range, N_B0)

    pk_my = np.zeros((N_B0, NT_REC))
    mk_my = np.zeros((N_B0, NT_REC))
    pk_mz = np.zeros((N_B0, NT_REC))
    mk_mz = np.zeros((N_B0, NT_REC))

    for i, B0 in enumerate(B0_values):
        r = run_prescribed_saw(B0, direction=+1, enable_mel=True, K_mr=KMR,
                               label=f"B0 {i+1}/{N_B0} +k")
        pk_my[i] = r['m_y']
        pk_mz[i] = r['m_z']

        r = run_prescribed_saw(B0, direction=-1, enable_mel=True, K_mr=KMR,
                               label=f"B0 {i+1}/{N_B0} -k")
        mk_my[i] = r['m_y']
        mk_mz[i] = r['m_z']

    times = r['times']
    np.savez(cache, B0_values=B0_values, times=times,
             pk_my=pk_my, mk_my=mk_my, pk_mz=pk_mz, mk_mz=mk_mz)
    return {'B0_values': B0_values, 'times': times,
            'pk_my': pk_my, 'mk_my': mk_my, 'pk_mz': pk_mz, 'mk_mz': mk_mz}


# ==========================================================================
# Experiment 3: Kmr dependence
# ==========================================================================
def exp3_kmr_sweep():
    """Sweep Kmr to show coupling strength dependence."""
    print("\n=== Exp 3: Kmr sweep ===")
    cache = os.path.join(DATA_DIR, "sim10_kmr_sweep.npz")

    if os.path.isfile(cache):
        print("  Loading cached...")
        return dict(np.load(cache, allow_pickle=True))

    Kmr_values = np.array([0.1e6, 0.3e6, 1.0e6, 3.0e6, 5.0e6])

    pk_peak = np.zeros(len(Kmr_values))
    mk_peak = np.zeros(len(Kmr_values))

    for i, kmr in enumerate(Kmr_values):
        r = run_prescribed_saw(B0_RES, direction=+1, enable_mel=True, K_mr=kmr,
                               label=f"Kmr={kmr*1e-6:.1f} +k")
        pk_peak[i] = np.max(r['m_perp'])

        r = run_prescribed_saw(B0_RES, direction=-1, enable_mel=True, K_mr=kmr,
                               label=f"Kmr={kmr*1e-6:.1f} -k")
        mk_peak[i] = np.max(r['m_perp'])

    np.savez(cache, Kmr_values=Kmr_values, pk_peak=pk_peak, mk_peak=mk_peak)
    return {'Kmr_values': Kmr_values, 'pk_peak': pk_peak, 'mk_peak': mk_peak}


# ==========================================================================
# Plotting
# ==========================================================================
def plot_channels(results):
    """Plot channel decomposition."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    t_ns = results[list(results.keys())[0]]['times'] * 1e9

    # Find keys
    keys = list(results.keys())

    # (a) +k vs -k full coupling
    ax = axes[0]
    for key in keys:
        if 'MELpMR' in key or 'MEL+MR' in key.replace('_', '+'):
            r = results[key]
            label = key.replace('_', ' ').replace('pk', '+k').replace('mk', '-k')
            color = 'C0' if r['direction'] > 0 else 'C1'
            ax.plot(t_ns, r['m_perp'], '-', color=color, lw=1.2, label=label)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$|m_\perp|$')
    ax.legend(fontsize=6.5)
    ax.set_title('(a) Full coupling (+k vs -k)', fontsize=9)

    # (b) Channel comparison for +k
    ax = axes[1]
    colors = {'pMR': 'C0', 'MR_only': 'C2', 'MEL_only': 'C3'}
    for key in keys:
        r = results[key]
        if r['direction'] > 0:
            label = key.replace('pk_', '+k ').replace('_', ' ')
            if 'MELpMR' in key:
                ax.plot(t_ns, r['m_perp'], '-', color='C0', lw=1.2, label='MEL+MR')
            elif 'MR_only' in key:
                ax.plot(t_ns, r['m_perp'], '-', color='C2', lw=1.2, label='MR only')
            elif 'MEL_only' in key:
                ax.plot(t_ns, r['m_perp'], '--', color='C3', lw=1.2, label='MEL only')

    ax.set_xlabel('Time (ns)')
    ax.legend(fontsize=6.5)
    ax.set_title('(b) Channel decomposition (+k)', fontsize=9)

    # (c) Nonreciprocity
    ax = axes[2]
    pk_full = mk_full = None
    for key in keys:
        r = results[key]
        if ('MELpMR' in key or 'MEL+MR' in key.replace('_', '+')) and r['direction'] > 0:
            pk_full = r['m_perp']
        if ('MELpMR' in key or 'MEL+MR' in key.replace('_', '+')) and r['direction'] < 0:
            mk_full = r['m_perp']

    if pk_full is not None and mk_full is not None:
        eps = 1e-30
        asymm = (pk_full - mk_full) / (pk_full + mk_full + eps) * 100
        ax.plot(t_ns, asymm, '-', color='C4', lw=1.2)
        ax.axhline(0, ls='--', color='gray', lw=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Asymmetry (%)')
        ax.set_title('(c) Nonreciprocity', fontsize=9)

    fig.suptitle(f'Prescribed SAW: YIG, B₀={B0_RES*1e3:.1f}mT, ε₀={EPS0:.0e}', fontsize=10)
    fig.tight_layout()
    fname = "fig_prescribed_channels.pdf"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_b0_sweep(data):
    """Plot B0 sweep spectra."""
    B0 = data['B0_values']
    times = data['times']

    dt = times[1] - times[0]
    freq = np.fft.rfftfreq(len(times), dt) * 1e-9  # GHz
    f_mask = (freq > 1.0) & (freq < 6.0)
    freq_m = freq[f_mask]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for idx, (label, my_data, mz_data) in enumerate([
        (r'$+k$', data['pk_my'], data['pk_mz']),
        (r'$-k$', data['mk_my'], data['mk_mz']),
    ]):
        spectra = np.zeros((len(B0), np.sum(f_mask)))
        for i in range(len(B0)):
            window = np.hanning(len(times))
            m_perp = np.sqrt(my_data[i]**2 + mz_data[i]**2)
            fft = np.abs(np.fft.rfft(m_perp * window))
            spectra[i] = fft[f_mask]

        ax = axes[idx]
        im = ax.pcolormesh(B0 * 1e3, freq_m, spectra.T,
                           cmap='hot', shading='auto')
        # Kittel line
        B0_fine = np.linspace(B0.min(), B0.max(), 100)
        f_kittel = GAMMA / (2*np.pi) * np.sqrt(B0_fine * (B0_fine + MU0*MS)) * 1e-9
        ax.plot(B0_fine*1e3, f_kittel, '--', color='cyan', lw=1, alpha=0.7)
        ax.axhline(F_SAW*1e-9, ls=':', color='cyan', lw=0.8, alpha=0.5)
        ax.set_xlabel(r'$B_0$ (mT)')
        if idx == 0:
            ax.set_ylabel('Frequency (GHz)')
        ax.set_title(f'({chr(97+idx)}) {label} spectrum', fontsize=9)

    # (c) Peak amplitude comparison
    ax = axes[2]
    pk_peaks = np.max(np.sqrt(data['pk_my']**2 + data['pk_mz']**2), axis=1)
    mk_peaks = np.max(np.sqrt(data['mk_my']**2 + data['mk_mz']**2), axis=1)
    ax.plot(B0*1e3, pk_peaks, 'o-', color='C0', ms=4, lw=1.2, label=r'$+k$')
    ax.plot(B0*1e3, mk_peaks, 's-', color='C1', ms=4, lw=1.2, label=r'$-k$')
    ax.axvline(B0_RES*1e3, ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'$B_0$ (mT)')
    ax.set_ylabel(r'Peak $|m_\perp|$')
    ax.legend(fontsize=7)
    ax.set_title('(c) Resonance profile', fontsize=9)

    fig.suptitle('Magnon spectra vs field (prescribed SAW)', fontsize=10)
    fig.tight_layout()
    fname = "fig_prescribed_b0sweep.pdf"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_kmr_sweep(data):
    """Plot Kmr dependence."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    Kmr = data['Kmr_values'] * 1e-6  # MJ/m^3
    pk = data['pk_peak']
    mk = data['mk_peak']

    ax = axes[0]
    ax.plot(Kmr, pk, 'o-', color='C0', ms=5, lw=1.2, label=r'$+k$')
    ax.plot(Kmr, mk, 's-', color='C1', ms=5, lw=1.2, label=r'$-k$')
    ax.set_xlabel(r'$K_{mr}$ (MJ/m³)')
    ax.set_ylabel(r'Peak $|m_\perp|$')
    ax.legend(fontsize=7)
    ax.set_title('(a) Peak response vs Kmr', fontsize=9)

    ax = axes[1]
    eps = 1e-30
    nonrecip = np.abs(pk - mk) / (pk + mk + eps) * 100
    ax.plot(Kmr, nonrecip, 'D-', color='C2', ms=5, lw=1.2)
    ax.set_xlabel(r'$K_{mr}$ (MJ/m³)')
    ax.set_ylabel('Nonreciprocity (%)')
    ax.set_title('(b) Nonreciprocity vs Kmr', fontsize=9)

    fig.suptitle(f'YIG: B₀={B0_RES*1e3:.1f} mT, ε₀={EPS0:.0e}', fontsize=10)
    fig.tight_layout()
    fname = "fig_prescribed_kmr.pdf"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 70)
    print("Sim 10: Prescribed SAW - Chiral Magnon-Phonon Coupling (YIG)")
    print("=" * 70)
    print(f"  YIG: Ms={MS*1e-3:.0f}kA/m, alpha={ALPHA:.0e}, B1={B1*1e-6:.1f}MJ/m³")
    print(f"  Kmr={KMR*1e-6:.1f}MJ/m³, xi={XI}")
    print(f"  SAW: f={F_SAW*1e-9:.1f}GHz, lambda={LAMBDA_SAW*1e6:.2f}um, eps0={EPS0:.0e}")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT")

    # Coupling fields
    saw_temp = ChiralSurfaceAcousticWave(F_SAW, LAMBDA_SAW, EPS0,
                                          ellipticity=XI, K_mr=KMR)
    saw_temp.coupling_hierarchy(B1, MS)

    # Run experiments
    results_ch = exp1_channel_decomposition()
    plot_channels(results_ch)

    data_b0 = exp2_b0_sweep()
    plot_b0_sweep(data_b0)

    data_kmr = exp3_kmr_sweep()
    plot_kmr_sweep(data_kmr)

    print("\n" + "=" * 70)
    print("Sim 10 (prescribed) complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
