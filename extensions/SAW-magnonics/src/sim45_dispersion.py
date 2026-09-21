"""Simulation 45: magnon dispersion omega(k) for realistic YIG, to find the
resonant operating point B0* where the degenerate pair k_SAW/2 sits at resonance
(delta = omega(k_SAW/2) - omega_K = 0), which drops the parametric threshold from
the dispersion-limited value (~2e-3 at B0=50mT) toward the damping limit (~3e-4).

Method: free ring-down. Realistic YIG, NO SAW, small broadband transverse seed
(1e-4 flat in k), T=0, low damping -> all k-modes ring at their eigenfrequency.
2D FFT of m_y(x,t) -> bright omega(k) dispersion. For each B0 read off
  omega_K = omega(k=0)   and   omega(k_SAW/2),  k_SAW/2 = 2*pi*f_K(B0)/v_SAW
then delta(B0) = omega(k_SAW/2) - omega_K. Root delta(B0*)=0 is the target point.

Scan B0 in {20..200 mT}. Fast (20 ns, no SAW). Checkpoint per B0.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from mumaxplus import World, Grid, Ferromagnet

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CKPT_DIR = os.path.join(DATA_DIR, "sim45_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4e-7 * np.pi
V_SAW = 3500.0
# realistic YIG
MS, AEX, ALPHA = 140e3, 3.65e-12, 5e-4

NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9
SEED_AMP = 1e-4
T_RUN = 20e-9
DT_REC = 10e-12          # 100 GHz Nyquist, fine omega resolution
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

B0_LIST = np.array([20, 30, 40, 50, 70, 100, 140, 200]) * 1e-3


def f_kittel(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


def run_one(B0):
    tag = f"B{B0*1e3:.0f}mT"
    cp = os.path.join(CKPT_DIR, f"sim45_{tag}.npz")
    if os.path.isfile(cp) and bool(np.load(cp)['done']):
        print(f"  [skip] {tag}", flush=True)
        return
    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    rng = np.random.default_rng(7)
    mag = np.zeros((3, NZ, NY, NX)); mag[0] = 1.0
    mag[1] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    mag[2] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
    magnet.magnetization = mag
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = 0.0
    # NO SAW: free ring-down
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    t0 = time.time()
    for i in range(NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    np.savez(cp, my_xt=my_xt, done=True, B0=B0, CX=CX, DT_REC=DT_REC, NX=NX)
    print(f"  {tag}: f_K={f_kittel(B0)/1e9:.2f}GHz done ({(time.time()-t0)/60:.1f}min)", flush=True)


def analyze():
    """Extract omega(k), omega_K, omega(k_SAW/2), delta(B0)."""
    print("\n=== dispersion analysis ===", flush=True)
    print(f"{'B0(mT)':>7} {'f_K(GHz)':>9} {'kSAW/2(/um)':>12} "
          f"{'f(kSAW/2)':>10} {'delta(MHz)':>11}", flush=True)
    rows = []
    for B0 in B0_LIST:
        cp = os.path.join(CKPT_DIR, f"sim45_B{B0*1e3:.0f}mT.npz")
        if not os.path.isfile(cp):
            continue
        d = np.load(cp); my = d['my_xt']; cx = float(d['CX']); dt = float(d['DT_REC'])
        sub = my.astype(float); sub -= sub.mean(0)
        w = np.hanning(sub.shape[0])[:, None]
        F = np.abs(np.fft.fftshift(np.fft.fft2(sub * w), axes=(0, 1)))
        fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=dt))
        kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=cx))
        kd = -kax; o = np.argsort(kd); kd = kd[o]; F = F[:, o]
        fpos = fax > 0
        # dispersion: for each k, peak-frequency (over positive f)
        def fpeak_at(kq):
            j = int(np.argmin(np.abs(kd - kq)))
            col = F[:, j].copy(); col[~fpos] = 0
            return fax[np.argmax(col)]
        fK = f_kittel(B0)
        kh = 2 * np.pi * fK / V_SAW          # k_SAW/2
        f0 = fpeak_at(0.0)                    # measured omega_K (k=0)
        fkh = fpeak_at(kh)                    # measured omega(k_SAW/2)
        delta_MHz = (fkh - f0) / 1e6
        rows.append((B0, fK, kh, f0, fkh, delta_MHz))
        print(f"{B0*1e3:7.0f} {fK/1e9:9.2f} {kh/1e6:12.2f} "
              f"{fkh/1e9:10.3f} {delta_MHz:11.1f}", flush=True)
    if len(rows) >= 2:
        B = np.array([r[0] for r in rows]); dl = np.array([r[5] for r in rows])
        sign = np.sign(dl)
        cross = np.where(np.diff(sign) != 0)[0]
        if cross.size:
            i = cross[0]
            # linear interp for B0*
            Bstar = B[i] - dl[i] * (B[i + 1] - B[i]) / (dl[i + 1] - dl[i])
            print(f"\n  ==> delta=0 crossing near B0* = {Bstar*1e3:.1f} mT "
                  f"(f_K={f_kittel(Bstar)/1e9:.2f} GHz)", flush=True)
        else:
            print(f"\n  ==> no delta=0 crossing in scanned range; delta sign "
                  f"{'>0 (exchange-dominated)' if dl.mean()>0 else '<0 (dipolar-dominated)'}", flush=True)
        np.savez(os.path.join(DATA_DIR, "sim45_dispersion.npz"),
                 B0=B, delta_MHz=dl, f_K=np.array([r[1] for r in rows]),
                 k_half=np.array([r[2] for r in rows]))


def main():
    print("=" * 70, flush=True)
    print("Sim 45: realistic-YIG dispersion -> resonant operating point B0*", flush=True)
    print("=" * 70, flush=True)
    for B0 in B0_LIST:
        run_one(B0)
    analyze()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
