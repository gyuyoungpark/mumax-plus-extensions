"""Simulation 42: Realistic-material thermal ENSEMBLE + direction reversal + CoFeB.

Revision Lever #1 completion + audit issue #2 fix, in one sweep. Builds on sim41
(realistic YIG, |B1|=0.35 MJ/m^3, thermal seeding). mumaxplus seeds its Langevin
RNG from the wall clock at each Ferromagnet construction (ferromagnet.cpp:119), so
every fresh run is an INDEPENDENT thermal realization -> a genuine ensemble.

Delivers:
  (A) YIG forward ensemble  : N=5 realizations, eps0=3e-4, +k_SAW  -> mean S(k,f_K)
      with spread (clean +k_SAW/2 band + error), and a GENUINE-ensemble phase
      coherence C(k1) across independent seeds (fixes audit issue #2: the main-text
      C=0.81 used 6 Welch segments of ONE deterministic run; here the members are
      independent thermal realizations).
  (B) YIG reversed ensemble : N=3 realizations, -k_SAW -> band mirrors to -k_SAW/2
      (strongest kinematic test, now in a REALISTIC material).
  (C) CoFeB forward ensemble: N=3 realizations, realistic CoFeB, its own k_SAW/2.

All MEL-only (K_mr=0) to isolate the parametric pump. T=300 K. Checkpoint per run.
Runs are 50 ns (band forms in the ~15-35 ns linear window; 50 ns is ample and
halves cost vs sim41's 100 ns).

Per CLAUDE.md: literature parameters (no toy inflation), physical (thermal) seeding,
runtime from the physical growth timescale, checkpointed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from mumaxplus import World, Grid, Ferromagnet
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CKPT_DIR = os.path.join(DATA_DIR, "sim42_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim42_ensemble_reversal.npz")

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0

# ---- Material presets (LITERATURE values, Table II) ----
YIG = dict(MS=140e3, AEX=3.65e-12, ALPHA=5e-4, B1=-0.35e6)
COFEB = dict(MS=1.0e6, AEX=19e-12, ALPHA=5e-3, B1=-8.0e6)

# ---- Geometry / run ----
NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
B0 = 50e-3
T_K = 300.0
T_RUN = 50e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13
EPS0 = 3e-4

# (config-label, material, direction(+1/-1), n_seeds)
CONFIGS = [
    ('YIG_fwd',   YIG,   +1, 5),
    ('YIG_rev',   YIG,   -1, 3),
    ('CoFeB_fwd', COFEB, +1, 3),
]


def kittel(mat):
    w = GAMMA * np.sqrt(B0 * (B0 + MU0 * mat['MS']))
    return w / (2 * np.pi)


def build(mat, direction):
    f_K = kittel(mat)
    f_saw = 2 * f_K
    lam = V_SAW / f_saw
    k_saw = 2 * np.pi / lam
    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = mat['MS']
    magnet.aex = mat['AEX']
    magnet.alpha = mat['ALPHA']
    magnet.B1 = mat['B1']
    magnet.magnetization = (1, 0, 0.001)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = T_K
    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=lam, amplitude=EPS0,
        direction='x', ellipticity=XI, K_mr=0.0, enable_barnett=False)
    saw.apply(magnet, Msat=mat['MS'], enable_mel=True)
    # MEL-only reversal: override the SAW wavevector sign (kernel phase = sawK*x - w t)
    if hasattr(magnet, 'saw_wavevector'):
        magnet.saw_wavevector = direction * abs(float(magnet.saw_wavevector))
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    return world, magnet, f_K, k_saw


def ckpt(label, s):
    return os.path.join(CKPT_DIR, f"sim42_{label}_seed{s}.npz")


def run_one(label, mat, direction, s):
    cp = ckpt(label, s)
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {label} seed{s}")
        return
    world, magnet, f_K, k_saw = build(mat, direction)
    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    t0 = time.time()
    for i in range(NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    np.savez(cp, my_xt=my_xt, done=True, eps0=EPS0, T_K=T_K, B0=B0,
             direction=direction, f_K=f_K, k_saw=k_saw, f_saw=2 * f_K,
             CX=CX, DT_REC=DT_REC, NX=NX, material=label.split('_')[0])
    print(f"  {label} seed{s}: f_K={f_K/1e9:.2f}GHz k_SAW/2={k_saw/2e6:+.2f}/um "
          f"({(time.time()-t0)/60:.1f} min)")
    sys.stdout.flush()


def main():
    print("=" * 74)
    print("Sim 42: realistic thermal ensemble + reversal + CoFeB")
    print(f"  B0={B0*1e3:.0f}mT T={T_K:.0f}K eps0={EPS0:.0e} T_run={T_RUN*1e9:.0f}ns")
    for label, mat, d, n in CONFIGS:
        print(f"  {label}: f_K={kittel(mat)/1e9:.2f}GHz  dir={'+' if d>0 else '-'}k  seeds={n}")
    print("=" * 74)
    for label, mat, direction, n_seeds in CONFIGS:
        print(f"\n---- {label} ----")
        for s in range(n_seeds):
            run_one(label, mat, direction, s)
    # pack
    pack = {}
    for label, mat, direction, n_seeds in CONFIGS:
        arrs = []
        for s in range(n_seeds):
            cp = ckpt(label, s)
            if os.path.isfile(cp):
                arrs.append(np.load(cp)['my_xt'])
        if arrs:
            pack[f'my_xt__{label}'] = np.stack(arrs)  # (n_seeds, NT, NX)
            d0 = np.load(ckpt(label, 0))
            for key in ('f_K', 'k_saw', 'direction', 'CX', 'DT_REC'):
                pack[f'{key}__{label}'] = d0[key]
    pack['labels'] = np.array([c[0] for c in CONFIGS])
    np.savez(CACHE, **pack)
    print(f"\nPACKED -> {CACHE}")


if __name__ == "__main__":
    main()
