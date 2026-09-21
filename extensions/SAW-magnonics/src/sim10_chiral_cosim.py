"""Simulation 10: Co-simulation of chiral SAW-magnon coupling in YIG.

Full elastodynamics + micromagnetics co-simulation demonstrating:
1. Strong magnon-phonon coupling via magneto-rotation in YIG
2. Direction-dependent (nonreciprocal) magnon excitation
3. MEL vs MR coupling channel comparison

The SAW is launched by a traction pulse at one boundary, propagates
through the YIG film, and couples to spin waves via:
  - Magnetoelastic coupling (B1, B2 through full strain tensor)
  - Magneto-rotation coupling (Kmr through rotation vector)
  - Barnett coupling (through angular velocity)

Material: YIG thin film (in-plane magnetized along x)
SAW: broadband pulse centered at ~3 GHz
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0

# ==========================================================================
# Physical constants
# ==========================================================================
GAMMA = 1.76e11        # rad/(s*T)

# ==========================================================================
# Material parameters: YIG
# ==========================================================================
MS = 140e3             # A/m
AEX = 3.65e-12         # J/m
ALPHA = 2e-4           # Gilbert damping
B1_MEL = -3.48e6       # J/m^3 (magnetoelastic B1, negative convention)
B2_MEL = -6.96e6       # J/m^3 (magnetoelastic B2, negative convention)

# Elastic constants (YIG)
RHO = 5170             # kg/m^3
C11 = 268e9            # Pa
C44 = 76.5e9           # Pa
C12 = C11 - 2 * C44

# Magneto-rotation
KMR = 1.0e6            # J/m^3

# ==========================================================================
# Geometry
# ==========================================================================
NX, NY, NZ = 256, 1, 1
CX = 10e-9             # 10 nm cellsize along x
CY = 10e-9
CZ = 20e-9             # 20 nm film thickness
LX = NX * CX           # 2.56 um

# ==========================================================================
# SAW pulse parameters
# ==========================================================================
F_CENTER = 3.0e9       # center frequency (Hz)
OMEGA_CENTER = 2 * np.pi * F_CENTER
PULSE_CYCLES = 5       # number of cycles in pulse
PULSE_DURATION = PULSE_CYCLES / F_CENTER  # ~1.67 ns
TRACTION_AMP = 1e8     # Pa (traction amplitude)

# Simulation time
T_MAX = 2e-9           # 2 ns total (reduced for speed)
DT_REC = 10e-12        # recording interval (10 ps)
NT_REC = int(T_MAX / DT_REC) + 1

# ==========================================================================
# Resonance condition
# ==========================================================================
def kittel_fmr(B0):
    """Kittel FMR frequency for in-plane film (Hz)."""
    return GAMMA / (2 * np.pi) * np.sqrt(B0 * (B0 + MU0 * MS))


def find_resonance_field(f_target):
    """Find B0 that gives FMR at f_target."""
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4*a*c)) / 2


B0_RES = find_resonance_field(F_CENTER)
print(f"Resonance field: B0 = {B0_RES*1e3:.2f} mT for f = {F_CENTER*1e-9:.1f} GHz")


# ==========================================================================
# Single simulation run
# ==========================================================================
def run_cosim(B0, direction=+1, enable_kmr=True, enable_barnett=False,
              enable_mel=True, label=""):
    """Run a single co-simulation.

    Parameters
    ----------
    B0 : float
        External field (T), applied along x.
    direction : +1 or -1
        SAW propagation direction.
    enable_kmr : bool
        Enable magneto-rotation coupling.
    enable_barnett : bool
        Enable Barnett effect.
    enable_mel : bool
        Enable magnetoelastic coupling (B1, B2).
    label : str
        Label for progress printing.

    Returns
    -------
    dict with time traces and metadata.
    """
    t_start = time.time()
    dir_str = "+" if direction > 0 else "-"
    print(f"  [{label}] Running {dir_str}k SAW, B0={B0*1e3:.1f}mT, "
          f"Kmr={'ON' if enable_kmr else 'OFF'}, MEL={'ON' if enable_mel else 'OFF'}...")

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize)
    magnet = Ferromagnet(world, grid)

    # --- Magnetic parameters ---
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0)  # in-plane along x

    # External field along x
    magnet.bias_magnetic_field = (B0, 0, 0)

    # Magnetoelastic coupling
    if enable_mel:
        magnet.B1 = B1_MEL
        magnet.B2 = B2_MEL

    # Magneto-rotation coupling
    if enable_kmr:
        magnet.Kmr = KMR
        magnet.anisU = (0, 0, 1)  # symmetry axis perpendicular to film

    # Barnett effect
    if enable_barnett:
        magnet.enable_barnett = True

    # --- Elastodynamics ---
    magnet.enable_elastodynamics = True
    magnet.rho = RHO
    magnet.C11 = C11
    magnet.C44 = C44
    magnet.C12 = C12
    magnet.elastic_displacement = (0, 0, 0)

    # Small elastic damping to prevent boundary reflections
    magnet.eta = 5e5  # viscous damping coefficient

    # Fixed timestep for stability
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 5e-14  # 50 fs

    # --- SAW excitation ---
    excite_width = 3 * CX  # 3 cells at boundary
    _amp = TRACTION_AMP
    _omega = OMEGA_CENTER
    _dur = PULSE_DURATION
    _dir = direction

    # Traction pulse: windowed sinusoidal
    def traction_time(t):
        if t > _dur:
            return (0., 0., 0.)
        env = np.sin(np.pi * t / _dur)**2  # Hanning envelope
        # Elliptical excitation (both normal and shear for Rayleigh-like wave)
        fx = _amp * env * np.sin(_omega * t) * _dir
        fz = _amp * env * np.cos(_omega * t)
        return (fx, 0., fz)

    # Apply at left or right boundary depending on direction
    if direction > 0:
        # Excite from left for +k propagation
        def spatial_mask(x, y, z):
            return (1.0 if x < excite_width else 0.0,
                    0.0,
                    1.0 if x < excite_width else 0.0)
    else:
        # Excite from right for -k propagation
        x_right = LX - excite_width
        def spatial_mask(x, y, z):
            return (1.0 if x > x_right else 0.0,
                    0.0,
                    1.0 if x > x_right else 0.0)

    magnet.external_body_force.add_time_term(traction_time, spatial_mask)

    # --- Recording ---
    times = np.zeros(NT_REC)
    m_avg = np.zeros((NT_REC, 3))  # average magnetization
    m_perp_rms = np.zeros(NT_REC)  # RMS transverse deviation

    # Record spatially resolved m at center region (avoid boundaries)
    center_start = NX // 4
    center_end = 3 * NX // 4
    n_center = center_end - center_start

    # Get initial magnetization
    m0 = np.array(magnet.magnetization.eval())  # shape: (3, NZ, NY, NX)

    for i in range(NT_REC):
        m = np.array(magnet.magnetization.eval())

        # Average magnetization
        m_avg[i, 0] = np.mean(m[0])
        m_avg[i, 1] = np.mean(m[1])
        m_avg[i, 2] = np.mean(m[2])

        # Transverse deviation in center region
        dm_y = m[1, 0, 0, center_start:center_end] - m0[1, 0, 0, center_start:center_end]
        dm_z = m[2, 0, 0, center_start:center_end] - m0[2, 0, 0, center_start:center_end]
        m_perp_rms[i] = np.sqrt(np.mean(dm_y**2 + dm_z**2))

        times[i] = world.timesolver.time

        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t_start
    print(f"    Done in {elapsed:.1f}s. Peak |dm_perp| = {np.max(m_perp_rms):.2e}")

    return {
        'times': times,
        'm_avg': m_avg,
        'm_perp_rms': m_perp_rms,
        'B0': B0,
        'direction': direction,
        'enable_kmr': enable_kmr,
        'enable_mel': enable_mel,
        'elapsed': elapsed,
    }


# ==========================================================================
# Experiment 1: Direction comparison at resonance
# ==========================================================================
def experiment_direction_comparison():
    """Compare +k and -k SAW at resonance field."""
    print("\n=== Experiment 1: Direction comparison at resonance ===")

    cache_file = os.path.join(DATA_DIR, "sim10_direction.npz")

    run_keys = ['pk_full', 'mk_full', 'pk_mr_only', 'mk_mr_only', 'pk_mel_only']
    result_fields = ['times', 'm_avg', 'm_perp_rms', 'B0', 'direction',
                     'enable_kmr', 'enable_mel', 'elapsed']

    if os.path.isfile(cache_file):
        print("  Loading cached data...")
        flat = dict(np.load(cache_file, allow_pickle=True))
        # Reconstruct nested dict from flat keys
        results = {}
        for rk in run_keys:
            results[rk] = {}
            for rf in result_fields:
                fk = f"{rk}_{rf}"
                if fk in flat:
                    v = flat[fk]
                    results[rk][rf] = v.item() if v.ndim == 0 else v
        return results

    results = {}

    # +k with MEL+MR
    r = run_cosim(B0_RES, direction=+1, enable_kmr=True, enable_mel=True,
                  label="Exp1a")
    results['pk_full'] = r

    # -k with MEL+MR
    r = run_cosim(B0_RES, direction=-1, enable_kmr=True, enable_mel=True,
                  label="Exp1b")
    results['mk_full'] = r

    # +k with MR only (no MEL)
    r = run_cosim(B0_RES, direction=+1, enable_kmr=True, enable_mel=False,
                  label="Exp1c")
    results['pk_mr_only'] = r

    # -k with MR only
    r = run_cosim(B0_RES, direction=-1, enable_kmr=True, enable_mel=False,
                  label="Exp1d")
    results['mk_mr_only'] = r

    # +k with MEL only (no MR) - should show effect from shear strain
    r = run_cosim(B0_RES, direction=+1, enable_kmr=False, enable_mel=True,
                  label="Exp1e")
    results['pk_mel_only'] = r

    # Save
    save_dict = {}
    for key, val in results.items():
        for k2, v2 in val.items():
            save_dict[f"{key}_{k2}"] = v2
    save_dict['times'] = results['pk_full']['times']

    np.savez(cache_file, **save_dict)
    print(f"  Data saved to {cache_file}")

    return results


# ==========================================================================
# Experiment 2: B0 sweep (dispersion mapping)
# ==========================================================================
def experiment_b0_sweep():
    """Sweep B0 through resonance to map dispersion."""
    print("\n=== Experiment 2: B0 sweep ===")

    cache_file = os.path.join(DATA_DIR, "sim10_b0sweep.npz")

    if os.path.isfile(cache_file):
        print("  Loading cached data...")
        data = dict(np.load(cache_file, allow_pickle=True))
        return data

    # B0 sweep range: ±40% around resonance
    B0_range = 0.4 * B0_RES
    N_B0 = 9  # number of field points (keep small for speed)
    B0_values = np.linspace(B0_RES - B0_range, B0_RES + B0_range, N_B0)
    B0_values = B0_values[B0_values > 5e-3]  # ensure positive and reasonable

    pk_perp = np.zeros((len(B0_values), NT_REC))
    mk_perp = np.zeros((len(B0_values), NT_REC))
    times = None

    for i, B0 in enumerate(B0_values):
        # +k
        r = run_cosim(B0, direction=+1, enable_kmr=True, enable_mel=True,
                      label=f"B0sweep +k {i+1}/{len(B0_values)}")
        pk_perp[i] = r['m_perp_rms']
        if times is None:
            times = r['times']

        # -k
        r = run_cosim(B0, direction=-1, enable_kmr=True, enable_mel=True,
                      label=f"B0sweep -k {i+1}/{len(B0_values)}")
        mk_perp[i] = r['m_perp_rms']

    np.savez(cache_file, B0_values=B0_values, pk_perp=pk_perp,
             mk_perp=mk_perp, times=times)
    print(f"  Data saved to {cache_file}")

    return {
        'B0_values': B0_values,
        'pk_perp': pk_perp,
        'mk_perp': mk_perp,
        'times': times,
    }


# ==========================================================================
# Plotting
# ==========================================================================
def plot_direction_comparison(results):
    """Plot direction comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    times = results['pk_full']['times'] * 1e9  # ns

    # (a) Time traces: +k vs -k (full coupling)
    ax = axes[0]
    ax.plot(times, results['pk_full']['m_perp_rms'] * 1e3,
            '-', color='C0', lw=1.2, label=r'$+k$ (MEL+MR)')
    ax.plot(times, results['mk_full']['m_perp_rms'] * 1e3,
            '-', color='C1', lw=1.2, label=r'$-k$ (MEL+MR)')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$|\delta m_\perp|$ ($\times 10^{-3}$)')
    ax.legend(fontsize=7)
    ax.set_title('(a) Full coupling', fontsize=9)

    # (b) MR only comparison
    ax = axes[1]
    ax.plot(times, results['pk_mr_only']['m_perp_rms'] * 1e3,
            '-', color='C0', lw=1.2, label=r'$+k$ (MR only)')
    ax.plot(times, results['mk_mr_only']['m_perp_rms'] * 1e3,
            '-', color='C1', lw=1.2, label=r'$-k$ (MR only)')
    ax.plot(times, results['pk_mel_only']['m_perp_rms'] * 1e3,
            '--', color='gray', lw=1.0, label=r'$+k$ (MEL only)')
    ax.set_xlabel('Time (ns)')
    ax.legend(fontsize=7)
    ax.set_title('(b) Channel decomposition', fontsize=9)

    # (c) Nonreciprocity ratio
    ax = axes[2]
    eps = 1e-30
    pk = results['pk_full']['m_perp_rms'] + eps
    mk = results['mk_full']['m_perp_rms'] + eps
    ratio = (pk - mk) / (pk + mk) * 100

    # Smooth with moving average
    window = max(1, len(ratio) // 50)
    if window > 1:
        ratio_smooth = np.convolve(ratio, np.ones(window)/window, mode='same')
    else:
        ratio_smooth = ratio

    ax.plot(times, ratio_smooth, '-', color='C2', lw=1.2)
    ax.axhline(0, ls='--', color='gray', lw=0.8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Nonreciprocity (%)')
    ax.set_title('(c) Direction asymmetry', fontsize=9)

    fig.suptitle(f'YIG co-simulation: B₀={B0_RES*1e3:.1f} mT, Kmr={KMR*1e-6:.1f} MJ/m³',
                 fontsize=10)
    fig.tight_layout()

    fname = "fig_cosim_direction.pdf"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_b0_sweep(data):
    """Plot B0 sweep results."""
    B0 = data['B0_values']
    times = data['times']
    pk_perp = data['pk_perp']
    mk_perp = data['mk_perp']

    # Compute FFT for each B0
    dt = times[1] - times[0]
    freq = np.fft.rfftfreq(len(times), dt) * 1e-9  # GHz

    # Frequency mask: 1-6 GHz
    f_mask = (freq > 1.0) & (freq < 6.0)
    freq_masked = freq[f_mask]

    pk_spectra = np.zeros((len(B0), np.sum(f_mask)))
    mk_spectra = np.zeros((len(B0), np.sum(f_mask)))

    for i in range(len(B0)):
        # Apply Hanning window
        window = np.hanning(len(times))
        pk_fft = np.abs(np.fft.rfft(pk_perp[i] * window))
        mk_fft = np.abs(np.fft.rfft(mk_perp[i] * window))
        pk_spectra[i] = pk_fft[f_mask]
        mk_spectra[i] = mk_fft[f_mask]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    B0_mT = B0 * 1e3

    # (a) +k spectrum
    ax = axes[0]
    im = ax.pcolormesh(B0_mT, freq_masked, pk_spectra.T,
                       cmap='hot', shading='auto')
    # Overlay Kittel dispersion
    B0_fine = np.linspace(B0.min(), B0.max(), 100)
    f_kittel = np.array([kittel_fmr(b) * 1e-9 for b in B0_fine])
    ax.plot(B0_fine * 1e3, f_kittel, '--', color='cyan', lw=1, alpha=0.7)
    ax.axhline(F_CENTER * 1e-9, ls=':', color='cyan', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$B_0$ (mT)')
    ax.set_ylabel('Frequency (GHz)')
    ax.set_title(r'(a) $+k$ SAW spectrum', fontsize=9)

    # (b) -k spectrum
    ax = axes[1]
    ax.pcolormesh(B0_mT, freq_masked, mk_spectra.T,
                  cmap='hot', shading='auto')
    ax.plot(B0_fine * 1e3, f_kittel, '--', color='cyan', lw=1, alpha=0.7)
    ax.axhline(F_CENTER * 1e-9, ls=':', color='cyan', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$B_0$ (mT)')
    ax.set_title(r'(b) $-k$ SAW spectrum', fontsize=9)

    # (c) Asymmetry
    ax = axes[2]
    diff = pk_spectra - mk_spectra
    vmax = np.max(np.abs(diff)) * 0.8
    ax.pcolormesh(B0_mT, freq_masked, diff.T,
                  cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
    ax.plot(B0_fine * 1e3, f_kittel, '--', color='k', lw=1, alpha=0.7)
    ax.set_xlabel(r'$B_0$ (mT)')
    ax.set_title('(c) Asymmetry (+k) - (-k)', fontsize=9)

    fig.suptitle('Magnon spectrum vs field (co-simulation)', fontsize=10)
    fig.tight_layout()

    fname = "fig_cosim_b0sweep.pdf"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 70)
    print("Sim 10: Co-simulation of Chiral SAW-Magnon Coupling (YIG)")
    print("=" * 70)
    print(f"  Material: YIG (Ms={MS*1e-3:.0f} kA/m, alpha={ALPHA:.0e}, "
          f"B1={B1_MEL*1e-6:.2f} MJ/m³)")
    print(f"  Kmr = {KMR*1e-6:.1f} MJ/m³")
    print(f"  Elastic: rho={RHO} kg/m³, C11={C11*1e-9:.0f} GPa, "
          f"C44={C44*1e-9:.1f} GPa")
    print(f"  Grid: {NX}x{NY}x{NZ}, cellsize=({CX*1e9:.0f}, {CY*1e9:.0f}, "
          f"{CZ*1e9:.0f}) nm")
    print(f"  SAW pulse: f_center={F_CENTER*1e-9:.1f} GHz, "
          f"T_pulse={PULSE_DURATION*1e9:.2f} ns")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT")

    # Coupling hierarchy (analytical)
    eps_ref = 1e-4  # reference strain for comparison
    H_mel = 2 * abs(B1_MEL) * eps_ref / MS  # Tesla
    H_mr = KMR * 0.68 * eps_ref / (2 * MS)  # Tesla
    print(f"\n  Coupling hierarchy (at eps0=1e-4):")
    print(f"    H_mel = {H_mel*1e3:.3f} mT (from B1, normal strain)")
    print(f"    H_mr  = {H_mr*1e6:.1f} uT (from Kmr, rotation)")

    # Experiment 1: Direction comparison
    results_dir = experiment_direction_comparison()
    if isinstance(results_dir, dict) and 'pk_full' in results_dir:
        plot_direction_comparison(results_dir)

        # Print summary
        pk_peak = np.max(results_dir['pk_full']['m_perp_rms'])
        mk_peak = np.max(results_dir['mk_full']['m_perp_rms'])
        pk_mr = np.max(results_dir['pk_mr_only']['m_perp_rms'])
        pk_mel = np.max(results_dir['pk_mel_only']['m_perp_rms'])

        print(f"\n  Direction comparison at B0={B0_RES*1e3:.1f} mT:")
        print(f"    +k (MEL+MR): peak |dm_perp| = {pk_peak:.2e}")
        print(f"    -k (MEL+MR): peak |dm_perp| = {mk_peak:.2e}")
        if mk_peak > 0:
            print(f"    Nonreciprocity: {abs(pk_peak-mk_peak)/(pk_peak+mk_peak)*100:.1f}%")
        print(f"    +k MR only:  peak = {pk_mr:.2e}")
        print(f"    +k MEL only: peak = {pk_mel:.2e}")

    # Experiment 2: B0 sweep (only if Exp1 succeeded)
    print("\n  Proceeding to B0 sweep...")
    data_sweep = experiment_b0_sweep()
    if isinstance(data_sweep, dict) and 'B0_values' in data_sweep:
        plot_b0_sweep(data_sweep)

    print("\n" + "=" * 70)
    print("Sim 10 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
