"""Simulation 48: tuned threshold at the dispersion-corrected pump frequency.

sim47 proved: at B0=50 mT, tuning f_SAW UP from 2*f_K (5.96 GHz, does NOT fire) to
~6.1 GHz restores firing at eps0=1e-3 (below the off-resonance 2e-3 threshold). The
correct pump matches the actual pair resonance 2*omega(k_SAW/2) ~ 6.1 GHz, not
2*omega_K(analytic). Now find the achievable-strain threshold at the tuned point:
fine f_SAW scan at eps0=3e-4 (narrow tongue), then lower strain at the winner.

Realistic YIG, B0=50 mT, T=0, 1e-4 seed, 40 ns. Checkpoint per run.
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CKPT_DIR = os.path.join(DATA_DIR, "sim48_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0
MS, AEX, ALPHA, B1 = 140e3, 3.65e-12, 5e-4, -0.35e6

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
B0 = 50e-3
SEED_AMP = 1e-4
T_RUN = 40e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

# (f_saw, eps0)
RUNS = (
    [(f, 3e-4) for f in [6.00e9, 6.05e9, 6.10e9, 6.15e9, 6.20e9]] +
    [(6.10e9, 2e-4), (6.10e9, 1e-4), (6.15e9, 2e-4)]
)


def run_one(f_saw, eps0):
    tag = f"f{f_saw/1e9:.2f}_eps{eps0:.0e}"
    cp = os.path.join(CKPT_DIR, f"sim48_{tag}.npz")
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {tag}", flush=True)
        return
    lam = V_SAW / f_saw
    k_saw = 2 * np.pi / lam
    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS; magnet.aex = AEX; magnet.alpha = ALPHA; magnet.B1 = B1
    rng = np.random.default_rng(5)
    mag = np.zeros((3, NZ, NY, NX)); mag[0] = 1.0
    mag[1] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    mag[2] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    magnet.magnetization = mag
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = 0.0
    saw = ChiralSurfaceAcousticWave(frequency=f_saw, wavelength=lam, amplitude=eps0,
                                    direction='x', ellipticity=XI, K_mr=0.0,
                                    enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=True)
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    t0 = time.time()
    for i in range(NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    sub = my_xt[NT_REC // 2:].astype(float); sub -= sub.mean(0)
    w = np.hanning(sub.shape[0])[:, None]
    F = np.abs(np.fft.fftshift(np.fft.fft2(sub * w), axes=(0, 1)))
    fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=CX)); kd = -kax
    o = np.argsort(kd); kd = kd[o]; F = F[:, o]
    S = F[int(np.argmin(np.abs(fax - f_saw / 2)))] ** 2
    mm = np.abs(kd) < 2.5 * k_saw
    peakk = kd[mm][np.argmax(S[mm])] / 1e6
    grew = np.abs(my_xt).max() > 5 * SEED_AMP
    np.savez(cp, my_xt=my_xt, done=True, B0=B0, eps0=eps0, f_saw=f_saw,
             k_saw=k_saw, CX=CX, DT_REC=DT_REC, NX=NX, grew=grew, peakk=peakk)
    print(f"  {tag}: k/2={k_saw/2e6:.2f} |m_y|max={np.abs(my_xt).max():.2e} "
          f"grew={'YES' if grew else 'no '} peak_k={peakk:+.2f} ({(time.time()-t0)/60:.1f}min)",
          flush=True)


def main():
    print("=" * 74, flush=True)
    print("Sim 48: tuned threshold at dispersion-corrected f_SAW (realistic YIG, B0=50mT)", flush=True)
    print("=" * 74, flush=True)
    for f_saw, eps0 in RUNS:
        run_one(f_saw, eps0)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
