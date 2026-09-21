"""Simulation 44: TRUE parametric threshold at realistic parameters (T=0, clean seed).

sim43 revealed that realistic YIG at eps0=3e-4, T=0 does NOT grow (|m_y| stayed at
the 1e-4 seed). Hypothesis: the parametric Mathieu tongue half-width
Gamma = gamma|B1|eps0/M_s is, at realistic (weak) B1, NARROWER than the dispersion
detuning delta = 2pi[omega(k_SAW/2) - omega_K] of the degenerate k_SAW/2 pair, so
the instability does not fire until eps0 exceeds delta*M_s/(gamma|B1|). The toy
paper's B1=25x YIG widened the tongue enough to fire; realistic YIG may be
dispersion-limited. CoFeB (B1~23x YIG) should still fire cleanly.

This scan establishes, at T=0 with a controlled 1e-4 flat-k transverse seed, for
each (material, eps0): does the instability fire, at what k does the band sit, and
the growth rate. Determines the HONEST realistic-material threshold and whether the
k_SAW/2 band forms.

Grid 1024x8x1 @ 5nm, 40 ns (enough to see growth or its absence). Checkpoint per run.
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
CKPT_DIR = os.path.join(DATA_DIR, "sim44_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0

YIG = dict(MS=140e3, AEX=3.65e-12, ALPHA=5e-4, B1=-0.35e6)
COFEB = dict(MS=1.0e6, AEX=19e-12, ALPHA=5e-3, B1=-8.0e6)

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
B0 = 50e-3
SEED_AMP = 1e-4
T_RUN = 40e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

# (material-label, material, [eps0 list])
SCAN = [
    ('YIG',   YIG,   [3e-4, 6e-4, 1e-3, 2e-3]),
    ('CoFeB', COFEB, [1e-4, 3e-4]),
]


def kittel(mat):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * mat['MS'])) / (2 * np.pi)


def run_one(mlabel, mat, eps0):
    tag = f"{mlabel}_eps{eps0:.0e}"
    cp = os.path.join(CKPT_DIR, f"sim44_{tag}.npz")
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {tag}", flush=True)
        return
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
    rng = np.random.default_rng(12345)
    mag = np.zeros((3, NZ, NY, NX)); mag[0] = 1.0
    mag[1] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    mag[2] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    magnet.magnetization = mag
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = 0.0
    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=lam, amplitude=eps0,
        direction='x', ellipticity=XI, K_mr=0.0, enable_barnett=False)
    saw.apply(magnet, Msat=mat['MS'], enable_mel=True)
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    t0 = time.time()
    for i in range(NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    np.savez(cp, my_xt=my_xt, done=True, eps0=eps0, f_K=f_K, k_saw=k_saw,
             material=mlabel, CX=CX, DT_REC=DT_REC, NX=NX, seed_amp=SEED_AMP)
    # quick on-the-fly report
    rms = np.sqrt((my_xt.astype(float) ** 2).mean(axis=1))
    grew = np.abs(my_xt).max() > 5 * SEED_AMP
    print(f"  {tag}: f_K={f_K/1e9:.2f}GHz k_SAW/2={k_saw/2e6:.2f}/um "
          f"|m_y|max={np.abs(my_xt).max():.2e} grew={grew} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)


def main():
    print("=" * 74, flush=True)
    print(f"Sim 44: TRUE threshold scan (T=0, seed={SEED_AMP:.0e}, {T_RUN*1e9:.0f}ns)", flush=True)
    for ml, mat, epss in SCAN:
        print(f"  {ml}: f_K={kittel(mat)/1e9:.2f}GHz  eps0={epss}", flush=True)
    print("=" * 74, flush=True)
    for mlabel, mat, epss in SCAN:
        print(f"\n---- {mlabel} ----", flush=True)
        for eps0 in epss:
            run_one(mlabel, mat, eps0)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
