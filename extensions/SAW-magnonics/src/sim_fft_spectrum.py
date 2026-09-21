"""sim_fft_spectrum: Single-shot FFT spectrum at B0=50 mT, f_SAW=2f_K.

Drives the system at the parametric resonance condition (f_SAW = 2 f_K)
and records the SIGNED transverse magnetization m_y(t), m_z(t) for FFT
analysis. The resulting spectrum directly visualizes the parametric
subharmonic generation:

  - MEL drive at 2f_K populates a peak at f_K (subharmonic, parametric)
  - MR drive responds at the driven frequency 2f_K (linear)
  - Full channel shows BOTH peaks

Three channels (full / mr_only / mel_only) run sequentially with
per-channel checkpoint. T_RUN = 100 ns gives ~10 MHz FFT resolution.

Output: data/sim_fft_spectrum.npz  (consumed by prl_figures_v2.py Fig 3(a))

Material / geometry mirrors sim20_parametric_channels.py for consistency.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim_fft_spectrum.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim_fft_spectrum_ckpt.npz")

# ==========================================================================
# Material: YIG  (identical to sim20)
# ==========================================================================
GAMMA = 1.76e11
MS    = 140e3
AEX   = 3.65e-12
ALPHA = 5e-4
B1    = -8.8e6
KMR   = 1.0e6
XI    = 0.68
V_SAW = 3500.0

NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

EPS0    = 1e-4
B0      = 50e-3                  # representative bias field
T_RUN   = 300e-9                 # 300 ns -> 3.3 MHz FFT resolution
DT_REC  = 20e-12                 # 20 ps -> Nyquist 25 GHz
DT_STEP = 5e-13
NT_REC  = int(round(T_RUN / DT_REC)) + 1

CHANNELS = [
    ('full',     True,  KMR),
    ('mr_only',  False, KMR),
    ('mel_only', True,  0.0),
]


def kittel_freq_hz(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


F_K   = kittel_freq_hz(B0)
F_SAW = 2.0 * F_K                # parametric resonance condition


def run_one_channel(enable_mel, K_mr):
    """Run a single channel; record m_y(t), m_z(t) for FFT."""
    wavelength = V_SAW / F_SAW

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    if enable_mel:
        magnet.B1 = B1

    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    my_arr = np.zeros(NT_REC)
    mz_arr = np.zeros(NT_REC)
    t0 = time.time()
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        my_arr[i] = float(avg[1])
        mz_arr[i] = float(avg[2])
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
        if (i + 1) % max(NT_REC // 10, 1) == 0:
            pct = 100.0 * (i + 1) / NT_REC
            print(f"      [{pct:5.1f}%] t={(i+1)*DT_REC*1e9:.1f} ns  "
                  f"my={my_arr[i]:+.3e}  ({time.time()-t0:.0f}s)")
    return my_arr, mz_arr


def atomic_savez(path, **kwargs):
    """Save .npz atomically: write to path.tmp.npz then os.replace."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kwargs)
    os.replace(tmp, path)


def main():
    print("=" * 72)
    print(f"sim_fft_spectrum: single-shot FFT at parametric resonance")
    print(f"  Grid: {NX}x{NY}x{NZ}, cellsize=({CX*1e9:.0f},{CY*1e9:.0f},"
          f"{CZ*1e9:.0f}) nm, PBC=(4,4,0)")
    print(f"  B0 = {B0*1e3:.0f} mT")
    print(f"  f_K   = {F_K*1e-9:.3f} GHz")
    print(f"  f_SAW = 2 f_K = {F_SAW*1e-9:.3f} GHz")
    print(f"  T_RUN = {T_RUN*1e9:.0f} ns,  DT_REC = {DT_REC*1e12:.0f} ps,  "
          f"NT_REC = {NT_REC}")
    print(f"  FFT resolution: {1.0/T_RUN*1e-6:.1f} MHz")
    print(f"  Nyquist: {0.5/DT_REC*1e-9:.1f} GHz")
    print(f"  Channels: {[c[0] for c in CHANNELS]}")
    print("=" * 72)

    # Already cached and parameters match?
    if os.path.isfile(CACHE_FILE):
        cached = dict(np.load(CACHE_FILE))
        match = (np.isclose(float(cached.get('T_RUN', 0)), T_RUN)
                 and np.isclose(float(cached.get('B0', 0)), B0)
                 and np.isclose(float(cached.get('DT_REC', 0)), DT_REC))
        if match and all(f'my_{c[0]}' in cached for c in CHANNELS):
            print("  Cache present and matches parameters. Nothing to do.")
            return
        print("  Cache present but parameters differ; recomputing.")

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        # Drop checkpoint if parameters changed
        ck_T = float(ckpt.get('T_RUN', np.array(0.0)))
        ck_dt = float(ckpt.get('DT_REC', np.array(0.0)))
        if not (np.isclose(ck_T, T_RUN) and np.isclose(ck_dt, DT_REC)):
            print(f"  Checkpoint parameters differ (T_RUN/DT_REC). Discarding.")
            ckpt = {}
        else:
            done = sum(1 for c in CHANNELS if f'my_{c[0]}' in ckpt)
            print(f"  Checkpoint loaded ({done}/{len(CHANNELS)} channels done)")

    # Record run params in the checkpoint so we can validate on restart.
    ckpt['T_RUN']  = np.array(T_RUN)
    ckpt['DT_REC'] = np.array(DT_REC)
    ckpt['B0']     = np.array(B0)
    ckpt['F_K']    = np.array(F_K)
    ckpt['F_SAW']  = np.array(F_SAW)
    ckpt['NT_REC'] = np.array(NT_REC)

    total_start = time.time()
    for name, enable_mel, K_mr in CHANNELS:
        key_my = f'my_{name}'
        key_mz = f'mz_{name}'

        if key_my in ckpt and key_mz in ckpt:
            print(f"\n  [{name:8s}] cached")
            continue

        print(f"\n  [{name:8s}] running...")
        ch_start = time.time()
        my_arr, mz_arr = run_one_channel(enable_mel, K_mr)
        ch_elapsed = time.time() - ch_start

        ckpt[key_my] = my_arr
        ckpt[key_mz] = mz_arr
        atomic_savez(CHECKPOINT, **ckpt)
        print(f"    done ({ch_elapsed/60:.1f} min). Checkpoint saved.")

    # All channels done -> promote checkpoint to final cache
    atomic_savez(CACHE_FILE, **ckpt)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)

    total_min = (time.time() - total_start) / 60.0
    print(f"\n  Total: {total_min:.1f} min")
    print(f"  Saved: {CACHE_FILE}")


if __name__ == "__main__":
    main()
