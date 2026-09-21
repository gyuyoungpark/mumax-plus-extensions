"""Simulation 46: confirm firing at the RESONANT operating point B0* (Option B).

sim45 (dispersion) predicts delta = omega(k_SAW/2)-omega_K crosses zero near
B0* ~ 170 mT for realistic YIG (delta ~ -48 MHz at 50 mT -> +13 MHz at 200 mT).
At delta=0 the parametric threshold should collapse from the dispersion-limited
~1.3e-3 (at 50 mT) toward the damping limit (~1e-4), restoring firing at achievable
strain in REALISTIC YIG (|B1|=0.35 MJ/m^3).

Direct test: realistic YIG, T=0, 1e-4 seed, B0 in {130,150,170,190} mT, eps0 in
{3e-4, 1e-4}. Does it fire (|m_y| -> O(1)) at achievable strain, and is the band at
k_SAW/2? Compare against the 50 mT baseline (which needs ~2e-3).

40 ns, 1024x8x1 @ 5nm. Checkpoint per run.
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
CKPT_DIR = os.path.join(DATA_DIR, "sim46_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0
MS, AEX, ALPHA, B1 = 140e3, 3.65e-12, 5e-4, -0.35e6   # realistic YIG

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
SEED_AMP = 1e-4
T_RUN = 40e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

# (B0_mT, eps0)
RUNS = [
    (130e-3, 3e-4), (150e-3, 3e-4), (170e-3, 3e-4), (190e-3, 3e-4),
    (170e-3, 1e-4), (170e-3, 2e-4),
    (50e-3, 3e-4),   # baseline (expected NOT to fire) for contrast
]


def f_kittel(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


def run_one(B0, eps0):
    tag = f"B{B0*1e3:.0f}_eps{eps0:.0e}"
    cp = os.path.join(CKPT_DIR, f"sim46_{tag}.npz")
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {tag}", flush=True)
        return
    f_K = f_kittel(B0)
    f_saw = 2 * f_K
    lam = V_SAW / f_saw
    k_saw = 2 * np.pi / lam
    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS; magnet.aex = AEX; magnet.alpha = ALPHA; magnet.B1 = B1
    rng = np.random.default_rng(3)
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
    # band location on 2nd half
    sub = my_xt[NT_REC // 2:].astype(float); sub -= sub.mean(0)
    w = np.hanning(sub.shape[0])[:, None]
    F = np.abs(np.fft.fftshift(np.fft.fft2(sub * w), axes=(0, 1)))
    fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=CX)); kd = -kax
    o = np.argsort(kd); kd = kd[o]; F = F[:, o]
    S = F[int(np.argmin(np.abs(fax - f_K)))] ** 2
    mm = np.abs(kd) < 2.5 * k_saw
    peakk = kd[mm][np.argmax(S[mm])] / 1e6
    grew = np.abs(my_xt).max() > 5 * SEED_AMP
    np.savez(cp, my_xt=my_xt, done=True, B0=B0, eps0=eps0, f_K=f_K, k_saw=k_saw,
             CX=CX, DT_REC=DT_REC, NX=NX, grew=grew, peakk=peakk)
    print(f"  {tag}: f_K={f_K/1e9:.2f}GHz k/2={k_saw/2e6:.1f} |m_y|max={np.abs(my_xt).max():.2e} "
          f"grew={'YES' if grew else 'no '} peak_k={peakk:+.1f} ({(time.time()-t0)/60:.1f}min)",
          flush=True)


def main():
    print("=" * 74, flush=True)
    print("Sim 46: confirm firing at resonant B0* (realistic YIG, Option B)", flush=True)
    print("=" * 74, flush=True)
    for B0, eps0 in RUNS:
        run_one(B0, eps0)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
