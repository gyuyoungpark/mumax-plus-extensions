"""Simulation 4: Magnon-phonon anticrossing dispersion.

Demonstrates the magnon-phonon anticrossing gap in the omega(k) dispersion
relation using the built-in elastodynamics solver.  A broadband sinc pulse
excites both magnon and phonon modes simultaneously; 2D FFT of m(t,x) and
u(t,x) reveals the coupled dispersion with an avoided crossing where the
magnon and transverse phonon branches would otherwise intersect.

This script follows the pattern of examples/magnetoelastic_dispersion.py
using CoFeB material parameters consistent with sim02.

Protocol:
  1. Set up a long 1D strip (4096 cells) with periodic boundaries.
  2. Enable elastodynamics (full bidirectional magnon-phonon coupling).
  3. Apply broadband sinc pulse (force + magnetic field) at the center.
  4. Record m(t,x) and u(t,x) time series.
  5. Compute 2D FFT to obtain omega(k) dispersion map.
  6. Overlay analytical uncoupled and coupled dispersion curves.
  7. Identify the anticrossing gap.

Material: CoFeB in-plane film
    Msat  = 1.0 MA/m
    Aex   = 15 pJ/m
    alpha = 0 (undamped for clean dispersion)
    B1    = -8.8 MJ/m^3
    rho   = 8000 kg/m^3
    C11   = 283 GPa
    C44   = 80 GPa
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import plot_style as ps

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import GAMMALL_DEFAULT, MU0
from mumaxplus.util.formulary import exchange_length
from mumaxplus.util.shape import XRange

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Material parameters (CoFeB, matching sim02)
# -----------------------------------------------------------------------
MSAT = 1.0e6          # A/m
AEX = 1.5e-11         # J/m
BDC = 50e-3           # T, external field along x

# Magnetoelastic parameters (CoFeB)
RHO = 8000            # kg/m^3
B1_MEL = -8.8e6       # J/m^3
B2_MEL = 0.0
C11 = 283e9           # Pa
C44 = 80e9            # Pa
C12 = C11 - 2 * C44   # isotropic assumption

# Angle between magnetization and wave propagation (M along x, k along x)
THETA = np.pi / 6     # pi/6 for visibility of both transverse branches

# -----------------------------------------------------------------------
# Time and grid settings
# -----------------------------------------------------------------------
FMAX = 20e9                         # maximum frequency for sinc pulse (Hz)
TIME_MAX = 10e-9                    # total simulation time (s)
DT = 1 / (2 * FMAX)                # Nyquist sampling interval
NT = 1 + int(TIME_MAX / DT)        # number of time points

NX, NY, NZ = 4096, 1, 1
L_EX = exchange_length(AEX, MSAT)
CX = L_EX                          # cellsize ~ exchange length
CY, CZ = 30e-9, 30e-9

# Excitation parameters
FAC = 1e13            # force pulse strength (N/m^3)
BAC = 1e-3            # magnetic pulse strength (T)
PULSE_WIDTH = 200e-9  # spatial width of excitation region (m)

CACHE_FILE = os.path.join(DATA_DIR, "sim04_data.npy")


# -----------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------

def simulation(theta=THETA, Bdc=BDC):
    """Run a magnon-phonon dispersion simulation.

    Parameters
    ----------
    theta : float
        Angle between magnetization and wave propagation direction (rad).
    Bdc : float
        External magnetic field strength (T).

    Returns
    -------
    m, u : ndarray
        Magnetization and elastic displacement arrays of shape
        (NT, 3, NZ, NY, NX).
    """
    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize, mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 100, 0))
    magnet = Ferromagnet(world, grid)

    # Magnetic parameters
    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = 0.01  # small damping for relax, will be zeroed later
    magnet.magnetization = (np.cos(theta), np.sin(theta), 0)
    magnet.bias_magnetic_field = (Bdc * np.cos(theta),
                                   Bdc * np.sin(theta), 0)

    # Relax to ground state before enabling elastodynamics
    magnet.relax()

    # Enable elastodynamics (full bidirectional coupling)
    magnet.enable_elastodynamics = True
    magnet.rho = RHO
    magnet.B1 = B1_MEL
    magnet.B2 = B2_MEL
    magnet.C11 = C11
    magnet.C44 = C44
    magnet.C12 = C12

    # Zero initial displacement
    magnet.elastic_displacement = (0, 0, 0)

    # Zero damping for clean dispersion lines
    magnet.alpha = 0
    magnet.eta = 0

    # Fixed timestep for consistent FFT
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 1e-13

    # Broadband sinc excitation at the center
    shape = XRange(-PULSE_WIDTH / 2,
                    PULSE_WIDTH / 2).translate(*magnet.center)
    mask = magnet._get_mask_array(shape, grid, world, "mask")

    Fac_dir = np.array([FAC, FAC, FAC]) / np.sqrt(3)
    Bac_dir = np.array([BAC, BAC, BAC]) / np.sqrt(3)

    def time_force_field(t):
        sinc = np.sinc(2 * FMAX * (t - TIME_MAX / 2))
        return tuple(sinc * Fac_dir)

    def time_magnetic_field(t):
        sinc = np.sinc(2 * FMAX * (t - TIME_MAX / 2))
        return tuple(sinc * Bac_dir)

    magnet.external_body_force.add_time_term(time_force_field, mask=mask)
    magnet.bias_magnetic_field.add_time_term(time_magnetic_field, mask=mask)

    # Allocate output arrays
    m = np.zeros(shape=(NT, 3, NZ, NY, NX))
    u = np.zeros(shape=(NT, 3, NZ, NY, NX))

    # Record initial state
    m[0, ...] = magnet.magnetization.eval()
    u[0, ...] = magnet.elastic_displacement.eval()

    print(f"  Running {NT} steps ({TIME_MAX*1e9:.1f} ns)...")
    for i in tqdm(range(1, NT)):
        world.timesolver.run(DT)
        m[i, ...] = magnet.magnetization.eval()
        u[i, ...] = magnet.elastic_displacement.eval()

    return m, u


# -----------------------------------------------------------------------
# Post-processing: 2D FFT and analytical dispersion
# -----------------------------------------------------------------------

def compute_dispersion(m, u):
    """Compute 2D FFT of magnetization and displacement.

    Returns
    -------
    ks : ndarray
        Wavenumber axis (rad/um).
    fs : ndarray
        Frequency axis (GHz).
    extent : list
        Image extent for imshow [k_min, k_max, f_min, f_max].
    FT_tot : ndarray
        Combined normalized spectral intensity.
    """
    ks = np.fft.fftshift(np.fft.fftfreq(NX, CX) * 2 * np.pi) * 1e-6
    fs = np.fft.fftshift(np.fft.fftfreq(NT, DT)) * 1e-9
    dk = ks[1] - ks[0]
    df = fs[1] - fs[0]
    extent = [ks[0] - dk / 2, ks[-1] + dk / 2,
              fs[0] - df / 2, fs[-1] + df / 2]

    # Sum 2D FFT over all 3 vector components
    u_FT = np.zeros((NT, NX))
    m_FT = np.zeros((NT, NX))
    for i in range(3):
        u_FT += np.abs(np.fft.fftshift(np.fft.fft2(u[:, i, 0, 0, :])))
        m_FT += np.abs(np.fft.fftshift(np.fft.fft2(m[:, i, 0, 0, :])))

    return ks, fs, extent, m_FT, u_FT


def plot_anticrossing(ks, fs, extent, m_FT, u_FT, theta=THETA, Bdc=BDC,
                       filename="fig_anticrossing_dispersion.pdf"):
    """Plot the magnon-phonon anticrossing dispersion.

    Panel (a): 2D FFT color map with analytical dispersion overlays.
    Panel (b): Zoom into the anticrossing region.
    """
    fig, axes = ps.double_panel_h(height_cm=7.0, wspace=0.40)

    # Plotting ranges (must include anticrossing at ~21 rad/um, ~10.6 GHz)
    xmin, xmax = 2, 25   # rad/um
    ymin, ymax = 2, 14   # GHz

    # Crop and normalize spectra
    x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * NX)
    x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * NX)
    y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * NT)
    y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * NT)

    u_max = np.max(u_FT[y_start:y_end, x_start:x_end])
    m_max = np.max(m_FT[y_start:y_end, x_start:x_end])

    if u_max > 0 and m_max > 0:
        FT_tot = u_FT / u_max + m_FT / m_max
    else:
        FT_tot = m_FT + u_FT

    # ---- Analytical uncoupled dispersion ----
    lambda_exch = (2 * AEX) / (MU0 * MSAT**2)
    k = np.linspace(xmin * 1e6, xmax * 1e6, 2000)

    # Elastic waves
    vt = np.sqrt(C44 / RHO)
    vl = np.sqrt(C11 / RHO)
    omega_t = vt * k
    omega_l = vl * k

    # Spin waves (Damon-Eshbach-like for thin film)
    omega_0 = GAMMALL_DEFAULT * Bdc
    omega_M = GAMMALL_DEFAULT * MU0 * MSAT
    P = 1 - (1 - np.exp(-k * CZ)) / (k * CZ)
    omega_fx = omega_0 + omega_M * (lambda_exch * k**2
                                     + P * np.sin(theta)**2)
    omega_fy = omega_0 + omega_M * (lambda_exch * k**2 + 1 - P)
    omega_fm = np.sqrt(omega_fx * omega_fy)

    # ---- Exact coupled dispersion (secular equation contour) ----
    J = GAMMALL_DEFAULT * B1_MEL**2 / (RHO * MSAT)
    omega_arr = np.linspace(ymin * 2 * np.pi * 1e9,
                            ymax * 2 * np.pi * 1e9, 2000)
    k_mesh, omega_mesh = np.meshgrid(k, omega_arr)

    # Recompute k-dependent quantities on mesh
    P_mesh = 1 - (1 - np.exp(-k_mesh * CZ)) / (k_mesh * CZ)
    omega_fx_mesh = omega_0 + omega_M * (lambda_exch * k_mesh**2
                                          + P_mesh * np.sin(theta)**2)
    omega_fy_mesh = omega_0 + omega_M * (lambda_exch * k_mesh**2
                                          + 1 - P_mesh)
    omega_fm_mesh = np.sqrt(omega_fx_mesh * omega_fy_mesh)
    omega_t_mesh = vt * k_mesh
    omega_l_mesh = vl * k_mesh

    # Secular equation for coupled magnon-phonon-phonon system
    equation = (omega_mesh**2 - omega_l_mesh**2) * \
               ((omega_mesh**2 - omega_t_mesh**2)**2
                * (omega_mesh**2 - omega_fm_mesh**2)
                - (omega_mesh**2 - omega_t_mesh**2)
                * J * k_mesh**2
                * (omega_fx_mesh * np.cos(theta)**2
                   + omega_fy_mesh * np.cos(2 * theta)**2)
                - J**2 * k_mesh**4
                * np.cos(2 * theta)**2 * np.cos(theta)**2)
    equation += -(omega_mesh**2 - omega_t_mesh**2) * J * k_mesh**2 * \
                (omega_fy_mesh * (omega_mesh**2 - omega_t_mesh**2)
                 * np.sin(2 * theta)**2
                 + J * k_mesh**2
                 * np.sin(2 * theta)**2 * np.cos(theta)**2)

    # ---- Panel (a): Full dispersion color map ----
    axes[0].imshow(FT_tot**2, aspect='auto', origin='lower', extent=extent,
                   vmin=0, vmax=0.6, cmap='inferno')

    # Uncoupled elastic branches (dashed red)
    axes[0].plot(k * 1e-6, omega_t / (2 * np.pi * 1e9),
                 '--', color='red', linewidth=1.0, alpha=0.8,
                 label='Elastic (uncoupled)')
    axes[0].plot(k * 1e-6, omega_l / (2 * np.pi * 1e9),
                 '--', color='red', linewidth=1.0, alpha=0.8)

    # Uncoupled magnon branch (dashed green)
    axes[0].plot(k * 1e-6, omega_fm / (2 * np.pi * 1e9),
                 '--', color='lime', linewidth=1.0, alpha=0.8,
                 label='Magnon (uncoupled)')

    # Coupled dispersion (white contour)
    axes[0].contour(k_mesh * 1e-6,
                    omega_mesh / (2 * np.pi * 1e9),
                    equation, [0], colors='white',
                    linewidths=1.2)
    axes[0].plot([], [], '-', color='white', linewidth=1.2,
                 label='Coupled (exact)')

    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_xlabel(r'Wavenumber $k$ (rad/$\mu$m)')
    axes[0].set_ylabel(r'Frequency $f$ (GHz)')
    axes[0].legend(fontsize=6, loc='upper left')
    ps.add_panel_label(axes[0], '(a)', x=0.03, y=0.92)

    # ---- Panel (b): Zoom into anticrossing region ----
    # Find approximate crossing point: omega_fm(k) = omega_t(k)
    diff = np.abs(omega_fm - omega_t)
    i_cross = np.argmin(diff)
    k_cross = k[i_cross] * 1e-6  # rad/um
    f_cross = omega_fm[i_cross] / (2 * np.pi * 1e9)  # GHz

    # Zoom window (centered on crossing point, no clamping)
    k_zoom_min = k_cross - 3
    k_zoom_max = k_cross + 3
    f_zoom_min = f_cross - 1.5
    f_zoom_max = f_cross + 1.5

    axes[1].imshow(FT_tot**2, aspect='auto', origin='lower', extent=extent,
                   vmin=0, vmax=0.6, cmap='inferno')

    axes[1].plot(k * 1e-6, omega_t / (2 * np.pi * 1e9),
                 '--', color='red', linewidth=1.0, alpha=0.8)
    axes[1].plot(k * 1e-6, omega_fm / (2 * np.pi * 1e9),
                 '--', color='lime', linewidth=1.0, alpha=0.8)
    axes[1].contour(k_mesh * 1e-6,
                    omega_mesh / (2 * np.pi * 1e9),
                    equation, [0], colors='white',
                    linewidths=1.5)

    # Mark the anticrossing gap
    axes[1].annotate(r'$2g_\mathrm{eff}$',
                     xy=(k_cross, f_cross),
                     fontsize=8, color='cyan',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2',
                              fc='black', alpha=0.6))

    axes[1].set_xlim(k_zoom_min, k_zoom_max)
    axes[1].set_ylim(f_zoom_min, f_zoom_max)
    axes[1].set_xlabel(r'$k$ (rad/$\mu$m)')
    axes[1].set_ylabel(r'$f$ (GHz)')
    ps.add_panel_label(axes[1], '(b)', x=0.03, y=0.92)

    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== Sim 04: Magnon-phonon anticrossing dispersion ===")
    print(f"  Material: CoFeB (Msat={MSAT*1e-6:.1f} MA/m, "
          f"B1={B1_MEL*1e-6:.1f} MJ/m^3)")
    print(f"  Grid: {NX}x{NY}x{NZ}, cellsize={CX*1e9:.1f} nm")
    print(f"  theta = {np.degrees(THETA):.0f} deg, B_ext = {BDC*1e3:.0f} mT")

    # Check for cached data
    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        cache = np.load(CACHE_FILE, allow_pickle=True).item()
        m = cache['m']
        u = cache['u']
    else:
        m, u = simulation()
        np.save(CACHE_FILE, {'m': m, 'u': u})
        print(f"  Data saved to {CACHE_FILE}")

    # Compute dispersion
    ks, fs, extent, m_FT, u_FT = compute_dispersion(m, u)

    # Plot anticrossing
    plot_anticrossing(ks, fs, extent, m_FT, u_FT)


if __name__ == "__main__":
    main()
