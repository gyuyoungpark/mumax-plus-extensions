"""Simulation 27: CoFeB validation of key PRL results.

Answers referee question: "Reproduce key results with CoFeB parameters."

CoFeB thin film (PMA, perpendicular):
  M_s = 1.0 MA/m, |B_1| = 8.0 MJ/m^3, K_mr ~ K_u + mu0*Ms^2/2 ~ 1.5 MJ/m^3,
  alpha = 0.005, A_ex = 19 pJ/m, K_u (PMA) = 600 kJ/m^3
  but for in-plane bias geometry along x, Ku is set to zero so that the
  film is in-plane along the SAW direction.

Three reduced runs:
  (a) f_SAW sweep at B_0 = 50 mT to find direct (f_K) and parametric
      (2f_K) peaks
  (b) eps0 threshold sweep
  (c) angular dependence near theta_c

Reduced sweeps for tractability: 30 f_SAW, 6 eps0, 9 theta.
Estimated runtime: ~45 minutes total.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim27_cofeb_validation.npz")

# CoFeB params
GAMMA = 1.76e11
MS = 1.0e6
AEX = 19e-12
ALPHA = 5e-3
B1 = -8.0e6
KMR = 1.5e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 5e-9, 5e-9, 5e-9
B0 = 0.30  # 300 mT — for CoFeB Kittel ~3 GHz region with high Ms

T_RUN = 5e-9
DT_REC = 50e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13


def kittel_freq_hz(B0_val, Ms=MS):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * Ms)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_single(f_saw, eps0, theta_deg, enable_mel, K_mr):
    wavelength = V_SAW / f_saw
    th = np.radians(theta_deg)
    mx0, my0 = np.cos(th), np.sin(th)
    Bx, By = B0 * np.cos(th), B0 * np.sin(th)

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (mx0, my0, 0.005)
    magnet.bias_magnetic_field = (Bx, By, 0)
    magnet.enable_demag = True
    if enable_mel:
        magnet.B1 = B1

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    m_perp = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        m_par = avg[0] * mx0 + avg[1] * my0
        m_p_ip = -avg[0] * my0 + avg[1] * mx0
        m_perp[i] = np.sqrt(m_p_ip ** 2 + avg[2] ** 2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    return float(np.max(m_perp[NT_REC // 2:]))


def main():
    print("=" * 60)
    print("Sim 27: CoFeB validation")
    print(f"  M_s={MS*1e-3:.0f} kA/m, |B1|={abs(B1)*1e-6:.1f} MJ/m^3, "
          f"K_mr={KMR*1e-6:.1f} MJ/m^3")
    print(f"  B0={B0*1e3:.0f} mT, f_K={F_K*1e-9:.2f} GHz, 2f_K={2*F_K*1e-9:.2f} GHz")
    print("=" * 60)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading.")
        data = dict(np.load(CACHE_FILE))
    else:
        results = {}

        # (a) f_SAW sweep
        print("\n  (a) f_SAW sweep, eps=1e-4, theta=0")
        f_vals = np.linspace(0.5, 4.0, 30) * F_K
        s_full = np.zeros_like(f_vals)
        s_mr = np.zeros_like(f_vals)
        s_mel = np.zeros_like(f_vals)
        t0 = time.time()
        for i, f in enumerate(f_vals):
            s_full[i] = run_single(f, 1e-4, 0.0, True, KMR)
            s_mr[i] = run_single(f, 1e-4, 0.0, False, KMR)
            s_mel[i] = run_single(f, 1e-4, 0.0, True, 0.0)
            print(f"    {i+1}/{len(f_vals)}: f={f*1e-9:.2f} GHz "
                  f"full={s_full[i]:.2e} mr={s_mr[i]:.2e} mel={s_mel[i]:.2e}")
        results['f_vals'] = f_vals
        results['s_full'] = s_full
        results['s_mr'] = s_mr
        results['s_mel'] = s_mel
        print(f"    elapsed: {(time.time()-t0)/60:.1f} min")

        # (b) eps0 threshold at f_SAW = 2f_K
        print("\n  (b) eps0 threshold, theta=0, f=2f_K")
        eps_vals = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
        t_full = np.zeros_like(eps_vals)
        t_mr = np.zeros_like(eps_vals)
        t0 = time.time()
        for i, eps in enumerate(eps_vals):
            t_full[i] = run_single(2 * F_K, eps, 0.0, True, KMR)
            t_mr[i] = run_single(F_K, eps, 0.0, False, KMR)
            print(f"    eps={eps:.0e}: full@2fK={t_full[i]:.2e} mr@fK={t_mr[i]:.2e}")
        results['eps_vals'] = eps_vals
        results['eps_full_2fk'] = t_full
        results['eps_mr_fk'] = t_mr
        print(f"    elapsed: {(time.time()-t0)/60:.1f} min")

        # (c) angular sweep at f=f_K (full coupling) to confirm theta_c
        print("\n  (c) angular sweep at f=f_K (full), theta=0..15 deg")
        thetas = np.array([0.0, 2.0, 4.0, 5.5, 7.0, 10.0, 15.0])
        a_full = np.zeros_like(thetas)
        t0 = time.time()
        for i, th in enumerate(thetas):
            a_full[i] = run_single(F_K, 1e-4, th, True, KMR)
            print(f"    theta={th:5.1f}deg: full={a_full[i]:.2e}")
        results['thetas'] = thetas
        results['a_full'] = a_full
        print(f"    elapsed: {(time.time()-t0)/60:.1f} min")

        results['MS'] = MS
        results['B1'] = B1
        results['KMR'] = KMR
        results['ALPHA'] = ALPHA
        results['B0'] = B0
        results['f_K'] = F_K
        np.savez(CACHE_FILE, **results)
        print(f"  Saved: {CACHE_FILE}")
        data = results

    plot_results(data)


def plot_results(data):
    try:
        from plot_style import (apply_style, label_panels, axis_label,
                                DOUBLE_COL, SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        DOUBLE_COL = 7.0
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'

    f_K = float(data['f_K'])
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 2.8))
    fig.subplots_adjust(wspace=0.42, left=0.07, right=0.98,
                        top=0.93, bottom=0.18)

    # (a) spectrum
    ax = axes[0]
    f = data['f_vals'] * 1e-9
    ax.plot(f, data['s_full'] / data['s_full'].max(), '-', color=BLACK,
            lw=1.0, label='Full')
    ax.plot(f, data['s_mr'] / data['s_full'].max(), '--', color=SKY_BLUE,
            lw=0.9, label='MR only')
    ax.plot(f, data['s_mel'] / data['s_full'].max(), ':', color=VERMILION,
            lw=1.2, label='MEL only')
    ax.axvline(f_K * 1e-9, color='gray', ls=':', lw=0.5)
    ax.axvline(2 * f_K * 1e-9, color=VERMILION, ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    ax.set_ylabel('Norm. $|m_\\perp|$')
    ax.set_title('(a) CoFeB spectra', fontsize=8)
    ax.legend(fontsize=6)

    # (b) threshold
    ax = axes[1]
    ax.loglog(data['eps_vals'], data['eps_mr_fk'], 's-', color=SKY_BLUE,
              ms=5, mec='k', mew=0.3, label=r'MR @ $f_K$')
    ax.loglog(data['eps_vals'], data['eps_full_2fk'], 'o-', color=VERMILION,
              ms=5, mec='k', mew=0.3, label=r'Full @ $2f_K$')
    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel(r'Peak $|m_\perp|$')
    ax.set_title('(b) Threshold', fontsize=8)
    ax.legend(fontsize=6)

    # (c) angular
    ax = axes[2]
    ax.plot(data['thetas'], data['a_full'] / data['a_full'][0], 'o-',
            color=BLACK, ms=5, mec='k', mew=0.3)
    theta_c_deg = np.degrees(np.arcsin(KMR * XI / (4 * abs(B1))))
    ax.axvline(theta_c_deg, color='gray', ls=':', lw=0.5,
               label=r'$\theta_c$')
    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'Norm. $|m_\perp|$ @ $f_K$')
    ax.set_title(rf'(c) Dip at $\theta_c\approx${theta_c_deg:.1f}$^\circ$',
                 fontsize=8)
    ax.legend(fontsize=6)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim27_cofeb.{ext}'),
                    dpi=200)
    plt.close(fig)
    print(f"  Saved: fig_sim27_cofeb.pdf/png ; theta_c (CoFeB) = {theta_c_deg:.2f} deg")


if __name__ == "__main__":
    main()
