"""Simulation 43: CLEAN finite-momentum band at REALISTIC parameters (T=0, controlled seed).

Lever #1, definitive version. sim42 showed that realistic YIG + T=300 K at
eps0=3e-4 saturates by ~10 ns with strong thermal mode-competition, so the
coherent +k_SAW/2 selection is washed out (huge seed-to-seed variance). That is
the OPPOSITE failure of the toy paper (too MUCH noise vs the paper's too little).

Clean approach: realistic YIG/CoFeB (LITERATURE |B1|), T=0, seeded by a small
uniform-random TRANSVERSE perturbation (amplitude SEED_AMP=1e-4 per cell, flat in
k so no k is preferred a priori). The parametric gain then selects +k_SAW/2 out of
the flat seed, cleanly and reproducibly, over a ~70 ns run -- exactly the benchmark
demonstration but at realistic B1, with a slightly larger deterministic seed to
offset the 25x weaker gain over feasible runtime (removing the FP-noise->long-runtime
problem WITHOUT inflating the coupling). This is the honest "does the mechanism
survive at literature parameters" test.

Independent rng seeds -> independent realizations for:
  (i) reproducibility of the +k_SAW/2 band,
  (ii) a genuine-ensemble pair coherence C(k1) (audit issue #2), now WITHOUT the
       thermal-reseeding that destroyed coherence in sim42.

Configs: YIG_fwd x3, YIG_rev x2 (band -> -k_SAW/2), CoFeB_fwd x2. MEL-only.
Checkpoint per run. Per CLAUDE.md: literature params, physical (small magnon) seed,
runtime from growth timescale, checkpointed.
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
CKPT_DIR = os.path.join(DATA_DIR, "sim43_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim43_clean_seed.npz")

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0

YIG = dict(MS=140e3, AEX=3.65e-12, ALPHA=5e-4, B1=-0.35e6)
COFEB = dict(MS=1.0e6, AEX=19e-12, ALPHA=5e-3, B1=-8.0e6)

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
B0 = 50e-3
EPS0 = 3e-4
SEED_AMP = 1e-4
T_RUN = 70e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

CONFIGS = [
    ('YIG_fwd',   YIG,   +1, 3),
    ('YIG_rev',   YIG,   -1, 2),
    ('CoFeB_fwd', COFEB, +1, 2),
]


def kittel(mat):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * mat['MS'])) / (2 * np.pi)


def build(mat, direction, rng_seed):
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
    # controlled small uniform-random transverse seed (flat in k), T=0
    rng = np.random.default_rng(rng_seed)
    mag = np.zeros((3, NZ, NY, NX))
    mag[0] = 1.0
    mag[1] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    mag[2] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    magnet.magnetization = mag
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = 0.0
    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=lam, amplitude=EPS0,
        direction='x', ellipticity=XI, K_mr=0.0, enable_barnett=False)
    saw.apply(magnet, Msat=mat['MS'], enable_mel=True)
    if hasattr(magnet, 'saw_wavevector'):
        magnet.saw_wavevector = direction * abs(float(magnet.saw_wavevector))
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    return world, magnet, f_K, k_saw


def ckpt(label, s):
    return os.path.join(CKPT_DIR, f"sim43_{label}_seed{s}.npz")


def run_one(label, mat, direction, s):
    cp = ckpt(label, s)
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {label} seed{s}", flush=True)
        return
    world, magnet, f_K, k_saw = build(mat, direction, rng_seed=1000 * hash(label) % 7919 + s)
    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    t0 = time.time()
    for i in range(NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    np.savez(cp, my_xt=my_xt, done=True, eps0=EPS0, seed_amp=SEED_AMP, B0=B0,
             direction=direction, f_K=f_K, k_saw=k_saw, f_saw=2 * f_K,
             CX=CX, DT_REC=DT_REC, NX=NX, material=label.split('_')[0])
    print(f"  {label} seed{s}: f_K={f_K/1e9:.2f}GHz k_SAW/2={direction*k_saw/2e6:+.2f}/um "
          f"|m_y|max={np.abs(my_xt).max():.2e} ({(time.time()-t0)/60:.1f} min)", flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None   # optional: run one label
    print("=" * 74, flush=True)
    print(f"Sim 43: CLEAN realistic-param band (T=0, seed={SEED_AMP:.0e}) eps0={EPS0:.0e}", flush=True)
    print("=" * 74, flush=True)
    for label, mat, direction, n_seeds in CONFIGS:
        if only and label != only:
            continue
        print(f"\n---- {label} ----", flush=True)
        for s in range(n_seeds):
            run_one(label, mat, direction, s)
    if only:
        return
    pack = {}
    for label, mat, direction, n_seeds in CONFIGS:
        arrs = [np.load(ckpt(label, s))['my_xt'] for s in range(n_seeds)
                if os.path.isfile(ckpt(label, s))]
        if arrs:
            pack[f'my_xt__{label}'] = np.stack(arrs)
            d0 = np.load(ckpt(label, 0))
            for key in ('f_K', 'k_saw', 'direction', 'CX', 'DT_REC'):
                pack[f'{key}__{label}'] = d0[key]
    pack['labels'] = np.array([c[0] for c in CONFIGS])
    np.savez(CACHE, **pack)
    print(f"\nPACKED -> {CACHE}", flush=True)


if __name__ == "__main__":
    main()
