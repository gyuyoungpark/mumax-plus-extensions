"""Simulation 41: Realistic-YIG finite-momentum parametric band under THERMAL seeding.

Revision Lever #1 — remove the toy-parameter crutch. The main-text benchmark
uses |B1|=8.8 MJ/m^3 (25x YIG) so the parametric instability grows out of the
floating-point noise floor within ~10 ns. Here we instead use the REALISTIC YIG
magnetoelastic constant |B1|=0.35 MJ/m^3 (same geometry, same M_s, same k_SAW as
the benchmark) and seed the instability THERMALLY at T=300 K. This is both
physically honest (mu-BLS detects thermally populated magnons) and removes the
enhanced-coupling regime from the central demonstration.

Physics check (why thermal seeding is the right move):
  growth rate  Gamma = gamma|B1|eps0/M_s - alpha*omega_K
  at YIG (B1=0.35e6), eps0=3e-4:  Gamma ~ 1.2e8 /s  (tau ~ 8 ns/e-fold)
  FP-noise seed (~1e-8..1e-15) -> needs ~300-800 ns to become visible.
  Thermal magnon seed at 300 K is ~1e-3..1e-2 per mode -> ~5-8 e-foldings
  (~50-80 ns) suffice. So realistic YIG at achievable strain (1-3e-4) shows
  the k_SAW/2 pair band within a ~80-100 ns run once thermally seeded.

Deliverable: k-resolved m_y(x,t) -> S(k, f_K) pair band centered at +k_SAW/2 in
realistic YIG at achievable strain, plus a T=0 control (no band within the run,
confirming the thermal seed is what makes it observable at realistic params).

Grid: 1024 x 8 x 1 @ 5 nm x-cell  (same strip as sim40, resolves k_SAW/2).
MEL-only (K_mr=0) to isolate the parametric pump channel.
Checkpoint: per (eps0, T) run, every SAVE_EVERY ns; saves full magnetization
state + accumulated m_y(x,t) + next time index -> resumable (approx. RNG resume).

Per CLAUDE.md: physically-reasonable realistic YIG params, SI units, checkpointed.
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
CKPT_DIR = os.path.join(DATA_DIR, "sim41_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim41_realistic_thermal.npz")

# ---------------- REALISTIC YIG (Serga2010 / Table II) ----------------------
GAMMA = 1.76e11
MS    = 140e3          # A/m   (same as benchmark -> same f_K, same k_SAW)
AEX   = 3.65e-12       # J/m
ALPHA = 5e-4           # Gilbert damping (low-damping YIG)
B1    = -0.35e6        # J/m^3  REALISTIC YIG magnetoelastic constant
XI    = 0.68
V_SAW = 3500.0
MU0   = 4e-7 * np.pi

# ---------------- Geometry (same strip as sim40) --------------------------
NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9

# ---------------- Run parameters ----------------
F_K   = 3.0e9
F_SAW = 2.0 * F_K       # parametric drive at 2 f_K
T_RUN = 100e-9          # 100 ns to accommodate slow realistic growth
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13
SAVE_EVERY_NS = 10.0    # checkpoint cadence

# (eps0, temperature_K, label)
RUNS = [
    (3e-4, 300.0, 'eps3e-4_T300'),   # achievable upper strain, thermal seed
    (2e-4, 300.0, 'eps2e-4_T300'),
    (1e-4, 300.0, 'eps1e-4_T300'),
    (3e-4,   0.0, 'eps3e-4_T0'),     # control: no thermal seed
]


def resonance_field(f_target):
    w0 = 2 * np.pi * f_target
    return (-MU0 * MS + np.sqrt((MU0 * MS)**2 + 4 * (w0 / GAMMA)**2)) / 2


B0 = resonance_field(F_K)
LAM = V_SAW / F_SAW
K_SAW = 2 * np.pi / LAM
SAVE_STRIDE = int(round((SAVE_EVERY_NS * 1e-9) / DT_REC))


def build(eps0, T_K):
    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.001)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = T_K
    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=LAM, amplitude=eps0,
        direction='x', ellipticity=XI, K_mr=0.0, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=True)
    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    return world, magnet


def ckpt_path(label):
    return os.path.join(CKPT_DIR, f"sim41_{label}.npz")


def run_one(eps0, T_K, label):
    cp = ckpt_path(label)
    world, magnet = build(eps0, T_K)

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    start_i = 0
    if os.path.isfile(cp):
        d = np.load(cp)
        if bool(d['done']):
            print(f"  [skip] {label} (complete)")
            return d['my_xt']
        my_xt = d['my_xt'].copy()
        start_i = int(d['next_i'])
        # approximate resume: reload full magnetization state
        magnet.magnetization = d['mstate']
        print(f"  [resume] {label} from i={start_i}/{NT_REC}")

    t0 = time.time()
    for i in range(start_i, NT_REC):
        my_xt[i] = magnet.magnetization.eval()[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
        if (i % SAVE_STRIDE == 0 and i > start_i) or i == NT_REC - 1:
            mstate = magnet.magnetization.eval().astype(np.float32)
            tmp = cp + ".tmp.npz"
            np.savez(tmp, my_xt=my_xt, next_i=i + 1, mstate=mstate,
                     done=(i == NT_REC - 1),
                     eps0=eps0, T_K=T_K, B0=B0, K_SAW=K_SAW,
                     F_K=F_K, F_SAW=F_SAW, CX=CX, DT_REC=DT_REC, NX=NX)
            os.replace(tmp, cp)
            print(f"    {label}  i={i}/{NT_REC-1}  t={i*DT_REC*1e9:.0f} ns  "
                  f"|m_y|max={np.abs(my_xt[:i+1]).max():.2e}  "
                  f"({(time.time()-t0)/60:.1f} min)")
            sys.stdout.flush()
    return my_xt


def main():
    print("=" * 74)
    print("Sim 41: REALISTIC YIG (|B1|=0.35 MJ/m^3) thermal parametric band")
    print(f"  B0={B0*1e3:.3f} mT  f_K={F_K/1e9:.2f} GHz  f_SAW={F_SAW/1e9:.2f} GHz")
    print(f"  k_SAW={K_SAW:.3e}/m  k_SAW/2={K_SAW/2e6:.3f}/um")
    print(f"  grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm  L_x={NX*CX*1e6:.2f}um  T_run={T_RUN*1e9:.0f}ns")
    print("=" * 74)
    pack = {'B0': B0, 'K_SAW': K_SAW, 'F_K': F_K, 'F_SAW': F_SAW,
            'CX': CX, 'DT_REC': DT_REC, 'NX': NX, 'labels': np.array([r[2] for r in RUNS])}
    for eps0, T_K, label in RUNS:
        print(f"\n---- {label}: eps0={eps0:.0e}, T={T_K:.0f} K ----")
        my_xt = run_one(eps0, T_K, label)
        pack[f'my_xt__{label}'] = my_xt
        pack[f'eps0__{label}'] = eps0
        pack[f'T__{label}'] = T_K
    np.savez(CACHE, **pack)
    print(f"\nPACKED -> {CACHE}")


if __name__ == "__main__":
    main()
