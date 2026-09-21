"""Simulation 47: f_SAW tuning restores firing at LOW strain (Option B, decisive).

Key realization from sim44-46: at realistic YIG the parametric instability is
detuned because the pump frequency f_SAW=2*f_K puts the DEGENERATE pair at k_SAW/2
off the magnon dispersion by delta=omega(k_SAW/2)-omega_K (~-48 MHz at 50 mT). The
paper tuned the pump to the k=0 mode (2*f_K), not to the actual pair. At low strain
the Mathieu tongue is narrow (~10 MHz at 3e-4, ~70 MHz at 1e-3), so a ~100 MHz
mistune misses the resonance entirely -> needs ~2e-3 to fire off-resonance (sim44).

Fix: tune f_SAW to the SELF-CONSISTENT pair resonance omega(k_SAW/2)=omega_SAW/2.
This drops the threshold to the damping limit at ANY B0. Here we prove it at the
paper's own field B0=50 mT: scan f_SAW at eps0=1e-3 (below the off-resonance 2e-3
threshold). Firing at some f_SAW* proves frequency tuning restores low-strain
operation. Then narrow strain at f_SAW*.

40 ns, T=0, 1e-4 seed, realistic YIG. Checkpoint per run.
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
CKPT_DIR = os.path.join(DATA_DIR, "sim47_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
XI = 0.68
V_SAW = 3500.0
MS, AEX, ALPHA, B1 = 140e3, 3.65e-12, 5e-4, -0.35e6   # realistic YIG

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
B0 = 50e-3
SEED_AMP = 1e-4
T_RUN = 40e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

F_K = GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)   # analytic 2.98 GHz
# f_SAW scan around 2 f_K (analytic 5.96 GHz); measured omega(0)~3.10 so pair
# resonance is near 2*3.05~6.1 GHz. Bracket both.
F_SAW_LIST = [5.70e9, 5.85e9, 5.96e9, 6.05e9, 6.15e9, 6.25e9]
# stage 2: strain narrowing at the winning f_SAW is added after we see results
EXTRA = []   # filled below if needed


def run_one(f_saw, eps0):
    tag = f"f{f_saw/1e9:.2f}GHz_eps{eps0:.0e}"
    cp = os.path.join(CKPT_DIR, f"sim47_{tag}.npz")
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
    # response frequency = subharmonic; look at f_saw/2 slice
    S = F[int(np.argmin(np.abs(fax - f_saw / 2)))] ** 2
    mm = np.abs(kd) < 2.5 * k_saw
    peakk = kd[mm][np.argmax(S[mm])] / 1e6
    grew = np.abs(my_xt).max() > 5 * SEED_AMP
    np.savez(cp, my_xt=my_xt, done=True, B0=B0, eps0=eps0, f_saw=f_saw, f_K=F_K,
             k_saw=k_saw, CX=CX, DT_REC=DT_REC, NX=NX, grew=grew, peakk=peakk)
    print(f"  {tag}: k/2={k_saw/2e6:.1f}/um |m_y|max={np.abs(my_xt).max():.2e} "
          f"grew={'YES' if grew else 'no '} peak_k={peakk:+.1f} ({(time.time()-t0)/60:.1f}min)",
          flush=True)


def main():
    print("=" * 74, flush=True)
    print(f"Sim 47: f_SAW scan @ B0=50mT eps0=1e-3 (analytic 2f_K={2*F_K/1e9:.2f}GHz)", flush=True)
    print("=" * 74, flush=True)
    for f_saw in F_SAW_LIST:
        run_one(f_saw, 1e-3)
    print("\nDONE stage1", flush=True)


if __name__ == "__main__":
    main()
