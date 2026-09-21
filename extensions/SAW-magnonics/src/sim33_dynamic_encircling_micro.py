"""Simulation 33 (Paper 2): Dynamic EP encircling in full LLG.

Promotes the lightweight ODE result of sim13 to a full mumax+
micromagnetic demonstration: the (eps_0, B_0) parameters are slowly
ramped along a closed loop in parameter space while the LLG equation
is solved with the chiral SAW kernel.  Topological signature:

  - Loop encloses the EP (inside): an initial magnon-like state ends
    up with non-zero phonon-like content (chiral state transfer).
  - Loop does NOT enclose the EP (outside): adiabatic; final state ~
    initial state.

Protocol per run:
  Phase 1 (0 <= t < T_loop):  loop trajectory in parameter space
      eps_0(t) = eps_EP + dr * cos(phi(t))
      B_0(t)   = B_EP   + dr * sin(phi(t))   [direction CW or CCW]
  Phase 2 (t >= T_loop):       parameters held fixed, ringdown.

Initial state seeded by tilting m towards z-axis (small m_z perturbation
that couples to the magnon mode).  We record the in-plane vs out-of-plane
oscillation amplitude versus time to read off the chiral state transfer.

Configurations: {inside CW, outside CW}.  Only two runs to keep total
runtime under ~2 hours; the analytic ODE in sim13 already covers the
CCW-vs-CW direction asymmetry.

Grid: 64x16x1 at 10 nm. T_loop = 25 ns, total run = 40 ns.
Estimated runtime: ~1 hour for two runs.
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
CACHE = os.path.join(DATA_DIR, "sim33_dynamic_encircling.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim33_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 64, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

# EP center (taken from sim31 analytic). We re-center the sweep around
# (eps_EP, B_EP); the radius dr controls inside/outside.
EPS_EP = 1.5e-4         # placeholder -- updated from sim31 cache if available
B_EP = 50e-3
DR_INSIDE = 1.0e-4
DR_OUTSIDE = 0.5e-4
LOOP_OFFSET_OUTSIDE_EPS = 4.0e-4   # offset center so the loop misses EP

T_LOOP = 25e-9
T_TOTAL = 40e-9
DT_REC = 100e-12
NT_REC = int(T_TOTAL / DT_REC) + 1
DT_STEP = 5e-13


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


def loop_params(t, kind, direction):
    """Parameter values along the encircling loop.

    kind: 'inside' (encloses EP) or 'outside' (misses EP).
    direction: 'CW' or 'CCW'.
    """
    if kind == 'inside':
        eps_c, B_c, dr = EPS_EP, B_EP, DR_INSIDE
    else:
        eps_c, B_c, dr = EPS_EP + LOOP_OFFSET_OUTSIDE_EPS, B_EP, DR_OUTSIDE
    if t >= T_LOOP:
        phi = 2 * np.pi  # parked at end of loop
    else:
        phi = 2 * np.pi * t / T_LOOP
    if direction == 'CCW':
        phi = -phi
    eps_t = eps_c + dr * np.cos(phi)
    B_t = B_c + dr / EPS_EP * 5e-3 * np.sin(phi)   # rescaled to mT range
    return float(max(eps_t, 1e-7)), float(B_t)


def run_one(kind, direction, label):
    t0 = time.time()
    f_saw = kittel_freq_hz(B_EP)     # near EP center
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.05)     # seed magnon mode (m_z!=0)
    eps0_init, B0_init = loop_params(0, kind, direction)
    magnet.bias_magnetic_field = (B0_init, 0, 0)
    magnet.enable_demag = True

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0_init,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=KMR, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=True)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    eps_arr = np.zeros(NT_REC)
    B_arr = np.zeros(NT_REC)
    m_arr = np.zeros((NT_REC, 3))

    for i in range(NT_REC):
        t = i * DT_REC
        eps_t, B_t = loop_params(t, kind, direction)
        magnet.saw_amplitude = eps_t
        magnet.bias_magnetic_field = (B_t, 0, 0)
        eps_arr[i], B_arr[i] = eps_t, B_t
        avg = magnet.magnetization.average()
        m_arr[i] = avg
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    print(f"  [{label}] done {(time.time()-t0)/60:.1f} min")
    return eps_arr, B_arr, m_arr


def main():
    print("=" * 72)
    print("Sim 33 (Paper 2): Dynamic EP encircling in full LLG")
    print(f"  EP center: eps_EP={EPS_EP:.2e}, B_EP={B_EP*1e3:.1f} mT")
    print(f"  Inside  loop: dr_eps={DR_INSIDE:.2e}")
    print(f"  Outside loop: offset center, dr_eps={DR_OUTSIDE:.2e}")
    print(f"  T_loop={T_LOOP*1e9:.0f} ns, T_total={T_TOTAL*1e9:.0f} ns")
    print("=" * 72)

    if os.path.isfile(CACHE):
        print("  Cached. Loading...")
        d = dict(np.load(CACHE))
        plot(d)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        print(f"  Checkpoint loaded ({len(ckpt)} entries)")

    runs = [('inside', 'CW'), ('outside', 'CW')]
    results = {}
    for kind, direction in runs:
        key = f'{kind}_{direction}'
        if f'{key}_m' in ckpt:
            results[f'{key}_eps'] = ckpt[f'{key}_eps']
            results[f'{key}_B'] = ckpt[f'{key}_B']
            results[f'{key}_m'] = ckpt[f'{key}_m']
            print(f"  [{key}] from checkpoint")
            continue
        eps_a, B_a, m_a = run_one(kind, direction, key)
        results[f'{key}_eps'] = eps_a
        results[f'{key}_B'] = B_a
        results[f'{key}_m'] = m_a
        ckpt[f'{key}_eps'] = eps_a
        ckpt[f'{key}_B'] = B_a
        ckpt[f'{key}_m'] = m_a
        np.savez(CHECKPOINT, **ckpt)

    save_dict = {**results, 'EPS_EP': EPS_EP, 'B_EP': B_EP,
                 'T_LOOP': T_LOOP, 'T_TOTAL': T_TOTAL, 'DT_REC': DT_REC}
    np.savez(CACHE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)
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

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 2.6))
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                        top=0.92, bottom=0.18)

    dt = float(d['DT_REC'])
    t_ns = np.arange(d['inside_CW_m'].shape[0]) * dt * 1e9

    # (a) parameter loop trajectories
    ax = axes[0]
    for kind, color, ls in [('inside_CW', VERMILION, '-'),
                            ('outside_CW', TEAL, '--')]:
        ax.plot(d[f'{kind}_eps'] * 1e4, d[f'{kind}_B'] * 1e3,
                ls, color=color, lw=0.9, label=kind.replace('_', ' '))
    ax.plot(float(d['EPS_EP']) * 1e4, float(d['B_EP']) * 1e3, '+',
            color='black', ms=12, mew=1.5, label='EP')
    ax.set_xlabel(r'$\varepsilon_0$ ($10^{-4}$)')
    ax.set_ylabel(r'$B_0$ (mT)')
    ax.legend(fontsize=7)

    # (b) m_z trajectory (magnon mode amplitude proxy)
    ax = axes[1]
    for kind, color in [('inside_CW', VERMILION), ('outside_CW', TEAL)]:
        m = d[f'{kind}_m']
        ax.plot(t_ns, m[:, 2], '-', color=color, lw=0.7,
                label=kind.replace('_', ' '))
    ax.axvline(float(d['T_LOOP']) * 1e9, color='gray', ls=':', lw=0.4)
    ax.set_xlabel(r'$t$ (ns)')
    ax.set_ylabel(r'$m_z$')
    ax.legend(fontsize=7)

    # (c) final state Bloch components
    ax = axes[2]
    bars_x = np.arange(2)
    in_final = d['inside_CW_m'][-1]
    out_final = d['outside_CW_m'][-1]
    width = 0.35
    ax.bar(bars_x - width / 2, [in_final[1], in_final[2]], width,
           color=VERMILION, label='inside')
    ax.bar(bars_x + width / 2, [out_final[1], out_final[2]], width,
           color=TEAL, label='outside')
    ax.set_xticks(bars_x)
    ax.set_xticklabels([r'$\langle m_y\rangle$', r'$\langle m_z\rangle$'])
    ax.legend(fontsize=7)

    try:
        label_panels(axes)
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim33_dynamic_encircling.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim33_dynamic_encircling.pdf/png")


if __name__ == "__main__":
    main()
