"""Simulation 2: SAW-driven ferromagnetic resonance.

A Rayleigh SAW at frequency f_SAW drives magnon excitations through
magnetoelastic coupling.  When f_SAW matches the Kittel ferromagnetic
resonance frequency

    f_K(B_0) = (gamma / 2pi) sqrt(B_0 (B_0 + mu_0 M_s))

the transverse magnetization amplitude |m_perp| is resonantly enhanced.

Protocol:
  (a) Fix external field B_0 at several values; sweep f_SAW.
      Record steady-state |m_perp| = sqrt(my^2 + mz^2).
      Plot absorption spectra.
  (b) Extract peak frequency for each B_0; overlay Kittel curve.

Material: CoFeB in-plane film
    Msat  = 1.0 MA/m
    Aex   = 15 pJ/m
    alpha = 0.005
    B1    = -8.8 MJ/m^3
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import plot_style as ps
from saw import SurfaceAcousticWave, GAMMA, MU0

from mumaxplus import Ferromagnet, Grid, World

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim02_data.npy")


# -----------------------------------------------------------------------
# Material and geometry parameters
# -----------------------------------------------------------------------
MSAT = 1.0e6         # A/m
AEX = 1.5e-11        # J/m
ALPHA = 0.005        # low damping for FMR linewidth
B1 = -8.8e6          # J/m^3
B2 = 0.0
EPS0 = 1e-3          # SAW strain amplitude

# Grid (small, uniform mode)
NX, NY, NZ = 4, 4, 1
CX, CY, CZ = 5e-9, 5e-9, 1e-9

# SAW
V_SAW = 3500.0       # m/s (LiNbO3 substrate)

# Simulation
T_RUN = 30e-9        # s  (longer for steady-state at weak drive)
N_STEPS = 3000


def run_single_point(B0, f_saw):
    """Run a simulation at a single (B_0, f_SAW) point.

    Parameters
    ----------
    B0 : float
        External field along x (T).
    f_saw : float
        SAW frequency (Hz).

    Returns
    -------
    m_perp_max : float
        Maximum transverse magnetization amplitude |m_perp|.
    m_perp_ss : float
        Steady-state (time-averaged) transverse amplitude.
    """
    cellsize = (CX, CY, CZ)
    # PBC in x,y for thin-film demagnetization (Nz -> 1)
    mastergrid = Grid((NX, NY, 0))
    world = World(cellsize, mastergrid=mastergrid,
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)  # small tilt to seed FMR
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True   # needed for Kittel formula

    # Magnetoelastic
    magnet.B1 = B1
    magnet.B2 = B2

    # SAW
    wavelength = V_SAW / f_saw
    saw = SurfaceAcousticWave(f_saw, wavelength, EPS0, direction='x')
    saw.apply(magnet)

    # Time array
    time_arr = np.linspace(0, T_RUN, N_STEPS + 1)

    # Track transverse magnetization
    def m_perp():
        m = magnet.magnetization.average()
        return np.sqrt(m[1]**2 + m[2]**2)

    quantity_dict = {"m_perp": m_perp}
    output = world.timesolver.solve(time_arr, quantity_dict)

    m_p = np.array(output["m_perp"])

    # Steady-state: average over the last 50% of the run
    n_half = len(m_p) // 2
    m_perp_ss = np.mean(m_p[n_half:])
    m_perp_max = np.max(m_p[n_half:])

    return m_perp_max, m_perp_ss


def run_fmr_sweep():
    """Sweep B_0 and f_SAW, record magnon absorption spectra.

    Returns
    -------
    results : dict
        'B0_values': array of field values (T)
        'f_saw_values': array of SAW frequencies (Hz)
        'spectra': dict mapping B0 -> array of |m_perp| values
        'peak_freqs': dict mapping B0 -> peak frequency (Hz)
    """
    B0_values = np.array([20e-3, 50e-3, 100e-3, 150e-3])
    f_saw_values = np.linspace(1e9, 15e9, 50)

    spectra = {}
    peak_freqs = {}

    for B0 in B0_values:
        B0_mT = B0 * 1e3
        print(f"  B0 = {B0_mT:.0f} mT:")

        # Kittel prediction
        f_kittel = SurfaceAcousticWave.kittel_frequency(B0, MSAT) / (2 * np.pi)
        print(f"    Kittel prediction: f_K = {f_kittel*1e-9:.2f} GHz")

        m_perp_arr = np.zeros(len(f_saw_values))

        for j, f_saw in enumerate(f_saw_values):
            _, m_ss = run_single_point(B0, f_saw)
            m_perp_arr[j] = m_ss

            if (j + 1) % 5 == 0:
                print(f"    [{100*(j+1)/len(f_saw_values):5.1f}%] "
                      f"f_SAW = {f_saw*1e-9:.2f} GHz, "
                      f"|m_perp| = {m_ss:.2e}")

        spectra[B0] = m_perp_arr

        # Find peak (require absolute threshold to filter noise)
        ABS_THRESHOLD = 1e-4  # minimum m_perp to count as genuine peak
        if np.max(m_perp_arr) > ABS_THRESHOLD:
            peaks, _ = find_peaks(m_perp_arr, height=0.3 * np.max(m_perp_arr))
            if len(peaks) > 0:
                i_peak = peaks[np.argmax(m_perp_arr[peaks])]
                peak_freqs[B0] = f_saw_values[i_peak]
                print(f"    Peak: f = {f_saw_values[i_peak]*1e-9:.2f} GHz")
            else:
                peak_freqs[B0] = np.nan
                print(f"    No clear peak found")
        else:
            peak_freqs[B0] = np.nan
            print(f"    No excitation detected")

    return {
        'B0_values': B0_values,
        'f_saw_values': f_saw_values,
        'spectra': spectra,
        'peak_freqs': peak_freqs,
    }


def plot_results(data, filename="fig_saw_fmr.pdf"):
    """Create the SAW-FMR figure.

    Panel (a): Absorption spectra |m_perp| vs f_SAW for each B_0.
    Panel (b): Peak frequency vs B_0 with Kittel curve overlay.
    """
    fig, axes = ps.double_panel_v(height_cm=12.0, hspace=0.35)

    B0_values = data['B0_values']
    f_saw_values = data['f_saw_values']
    spectra = data['spectra']
    peak_freqs = data['peak_freqs']

    colors = ps.COLORS_4

    # (a) Absorption spectra
    for i, B0 in enumerate(B0_values):
        f_ghz = f_saw_values * 1e-9
        m_perp = spectra[B0]

        # Normalize for visibility
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
    # Re-extract peaks with absolute threshold to filter noise
    ABS_THRESHOLD = 1e-4
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
    # Filter out NaN peaks
    valid = ~np.isnan(f_peaks)
    B0_arr_valid = B0_arr[valid]
    f_peaks_valid = f_peaks[valid]

    axes[1].plot(B0_arr_valid * 1e3, f_peaks_valid * 1e-9, 'o',
                 color=ps.BLUE, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.4,
                 label='Simulation', zorder=10)

    # Analytical Kittel curve
    B_theory = np.linspace(0, B0_arr[-1] * 1.2, 200)
    f_kittel = SurfaceAcousticWave.kittel_frequency(B_theory, MSAT) / (2 * np.pi)
    axes[1].plot(B_theory * 1e3, f_kittel * 1e-9, '--',
                 color=ps.VERMILION, linewidth=1.0,
                 label=r'Kittel: $\gamma\sqrt{B_0(B_0 + \mu_0 M_s)}/2\pi$')

    axes[1].set_xlabel(r'$B_0$ (mT)')
    axes[1].set_ylabel(r'$f_\mathrm{peak}$ (GHz)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=6.5, loc='lower right')
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    print("=== Sim 02: SAW-driven ferromagnetic resonance ===")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        data = np.load(CACHE_FILE, allow_pickle=True).item()
    else:
        data = run_fmr_sweep()
        np.save(CACHE_FILE, data)
        print(f"  Data saved to {CACHE_FILE}")

    plot_results(data)


if __name__ == "__main__":
    main()
