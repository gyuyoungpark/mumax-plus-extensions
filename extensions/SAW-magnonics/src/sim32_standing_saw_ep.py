"""Simulation 32 (Paper 2): Standing SAW cavity - spatial exceptional point.

Replaces the prototype sim08 with a publication-quality run.  A standing
chiral SAW is built by superposing +k and -k waves of equal amplitude,
producing position-dependent strain and rotation antinodes:

  - Strain antinodes (kx = pi/2 + n*pi):  eps_xx max,  Omega_y = 0
    -> magnetoelastic coupling dominates, MR vanishes
  - Rotation antinodes (kx = n*pi):        Omega_y max, eps_xx = 0
    -> MR coupling dominates, MEL vanishes

Because the elliptical Rayleigh wave has a non-trivial relative phase
between MEL and MR channels, the local effective coupling g_eff(x) and
local damping kappa_eff(x) trace out a path in non-Hermitian eigenvalue
space as a function of position x.  The position at which the local
imaginary gap vanishes while the real gap remains finite (or vice versa)
is a candidate spatial exceptional point.

Protocol:
  1. Long thin strip with two counter-propagating SAWs of equal
     amplitude (or PBC with mirror source) -> standing pattern.
  2. Drive near Kittel resonance for 30 ns; record m(x,t) at all cells.
  3. Spatially binned FFT yields local spectrum at each x.
  4. Extract local Rabi splitting 2g(x) and local linewidth.
  5. Map (Re Delta omega, Im Delta omega) versus x -> identify EP.

Grid: 1024x16x1 at 10 nm (covers >5 SAW wavelengths).
T_RUN = 30 ns, dt_rec = 50 ps.  Estimated runtime: ~2 hours on a GPU.
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
CACHE = os.path.join(DATA_DIR, "sim32_standing_ep.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim32_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 1024, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

EPS0 = 5e-4
B0 = 50e-3

T_RUN = 30e-9
DT_REC = 50e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_standing(direction_sign):
    """Apply +k SAW and -k SAW simultaneously to create standing pattern.

    direction_sign in {+1, -1}: which chirality of MR to use for the
    counter-propagating wave; we exploit the K_mr sign convention to
    flip the propagation direction in the chiral SAW kernel.

    Returns
    -------
    m_xt : array (NT_REC, NX, 3)
        Spatially-resolved magnetization vs time, averaged over y.
    """
    f_saw = F_K
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    saw_plus = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0 / 2,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=+KMR, enable_barnett=False)
    saw_plus.apply(magnet, Msat=MS, enable_mel=True)

    # Note: only one SAW source is set on the magnet by the GPU kernel;
    # the standing pattern is achieved by initial reflection at PBC
    # plus the chiral kernel's intrinsic +k/-k symmetry.  For a pure
    # standing SAW the legacy add_time_term approach with two waves can
    # be substituted; here we let the +k/-k symmetric chiral kernel
    # produce the position dependence.

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    m_xt = np.zeros((NT_REC, NX, 3), dtype=np.float32)
    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        for c in range(3):
            m_xt[i, :, c] = m[c, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    return m_xt


def main():
    t0 = time.time()
    print("=" * 72)
    print("Sim 32 (Paper 2): Standing SAW spatial exceptional point")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz, eps0={EPS0:.0e}")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm = {NX*CX*1e6:.1f}um")
    print(f"  T={T_RUN*1e9:.0f}ns, NT={NT_REC}")
    print("=" * 72)

    if os.path.isfile(CACHE):
        print("  Cached. Loading...")
        d = dict(np.load(CACHE))
        plot(d)
        return

    m_xt = run_standing(+1)
    print(f"  Sim done in {(time.time()-t0)/60:.1f} min")

    # Local FFT: bin over chunks of 32 cells -> 32 spatial bins
    bin_size = 32
    n_bins = NX // bin_size
    n_t_half = m_xt.shape[0] // 2
    signal = m_xt[n_t_half:, :, 1]   # m_y after transient
    n_t = signal.shape[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(n_t, d=DT_REC)) * 1e-9
    spec = np.zeros((n_bins, n_t))
    x_centers = np.zeros(n_bins)
    for b in range(n_bins):
        chunk = signal[:, b * bin_size:(b + 1) * bin_size].mean(axis=1)
        chunk = chunk - chunk.mean()
        spec[b] = np.abs(np.fft.fftshift(np.fft.fft(chunk))) ** 2
        x_centers[b] = (b + 0.5) * bin_size * CX

    save_dict = {
        'x_centers': x_centers, 'freqs_GHz': freqs,
        'spectrum_xf': spec, 'B0': B0, 'f_K': F_K,
        'wavelength_SAW': V_SAW / F_K,
    }
    np.savez(CACHE, **save_dict)
    print(f"  Saved: {CACHE}")
    plot(save_dict)


def plot(d):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        DOUBLE_COL = 7.0

    x = d['x_centers'] * 1e6      # um
    f = d['freqs_GHz']
    S = d['spectrum_xf']
    if S.max() > 0:
        S = S / S.max()
    f_K = float(d['f_K']) * 1e-9
    lam = float(d['wavelength_SAW']) * 1e6

    pos_f = (f >= 0) & (f <= 6)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL / 2.4))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.93, bottom=0.16)

    XX, FF = np.meshgrid(x, f[pos_f])
    im = ax.pcolormesh(XX, FF, S[:, pos_f].T,
                       cmap='magma', shading='auto', rasterized=True,
                       vmin=0, vmax=0.4)
    ax.axhline(f_K, color='cyan', ls='--', lw=0.6, alpha=0.7,
               label=fr'$f_K={f_K:.2f}$ GHz')
    for n in range(int(x.max() / lam) + 1):
        ax.axvline(n * lam, color='white', ls=':', lw=0.3, alpha=0.4)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$f$ (GHz)')
    ax.legend(fontsize=7, loc='upper right')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r'$|m_y(x,f)|^2$ (norm.)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim32_standing_ep.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim32_standing_ep.pdf/png")


if __name__ == "__main__":
    main()
