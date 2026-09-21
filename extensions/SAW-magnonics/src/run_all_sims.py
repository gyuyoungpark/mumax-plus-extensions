"""Master runner for all new SAW-magnon simulations (sim05-sim08).

Runs optimized versions of all simulations with reduced sweep points
for feasibility, then produces figures and saves data.
"""

import json
import os
import sys
import time
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from saw import MU0, GAMMA
from saw_chiral import ChiralSurfaceAcousticWave, StandingSAW
from analysis import (windowed_local_spectrum, extract_peaks,
                      classify_coupling_regime, fit_lorentzian)

from mumaxplus import Ferromagnet, Grid, World

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

for d in ["figures",
          "data/raw/sim05_mr", "data/raw/sim06_nonreciprocal",
          "data/raw/sim07_barnett", "data/raw/sim08_standing_ep",
          "data/processed/spectra", "data/processed/maps",
          "data/logs"]:
    os.makedirs(os.path.join(SCRIPT_DIR, d), exist_ok=True)


# ===================================================================
# Common parameters
# ===================================================================
MSAT = 1.0e6       # A/m
AEX = 1.5e-11      # J/m
B1 = -8.8e6         # J/m^3
XI = 0.68           # Rayleigh ellipticity
V_SAW = 3500.0      # m/s


def make_fmr_point(f_saw, B0, alpha, K_mr=0.0, enable_barnett=False,
                   eps0=1e-3, phase=0.0, nx=16, t_run=15e-9, n_steps=1500):
    """Generic single FMR simulation point.

    Returns steady-state |m_perp| averaged over last half of simulation.
    """
    cellsize = (5e-9, 5e-9, 1e-9)
    world = World(cellsize)
    magnet = Ferromagnet(world, Grid((nx, nx, 1)))

    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = alpha
    magnet.magnetization = (1, 0, 0)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = False
    magnet.B1 = B1
    magnet.B2 = 0.0

    wavelength = V_SAW / f_saw
    saw = ChiralSurfaceAcousticWave(
        f_saw, wavelength, eps0,
        direction='x', phase=phase,
        ellipticity=XI, K_mr=K_mr,
        enable_barnett=enable_barnett)
    saw.apply(magnet, Msat=MSAT)

    time_arr = np.linspace(0, t_run, n_steps + 1)

    def m_perp():
        m = magnet.magnetization.average()
        return np.sqrt(m[1]**2 + m[2]**2)

    output = world.timesolver.solve(time_arr, {"m_perp": m_perp})
    m_p = np.array(output["m_perp"])

    n_half = len(m_p) // 2
    return np.mean(m_p[n_half:])


# ===================================================================
# SIM 05: Magneto-rotation validation
# ===================================================================

def run_sim05():
    """Magneto-rotation coupling validation."""
    print("\n" + "="*60)
    print("SIM 05: Magneto-rotation coupling validation")
    print("="*60)

    B0 = 50e-3
    ALPHA = 0.005
    K_MR = 5e6
    EPS0 = 1e-3

    # Show coupling hierarchy
    saw_temp = ChiralSurfaceAcousticWave(
        2e9, V_SAW/2e9, EPS0, ellipticity=XI,
        K_mr=K_MR, enable_barnett=True)
    saw_temp.coupling_hierarchy(B1, MSAT)

    f_kittel = GAMMA * np.sqrt(B0 * (B0 + MU0 * MSAT)) / (2*np.pi)
    print(f"  Kittel frequency: {f_kittel*1e-9:.2f} GHz")

    # (a) Spectral sweep: 20 frequency points for 3 configs
    f_saw_values = np.linspace(1e9, 10e9, 20)
    configs = {
        'MEL only':    {'K_mr': 0.0, 'barnett': False},
        'MEL+MR':      {'K_mr': K_MR, 'barnett': False},
        'MEL+MR+B':    {'K_mr': K_MR, 'barnett': True},
    }

    spectra = {}
    for name, cfg in configs.items():
        print(f"\n  Sweep: {name}")
        t0 = time.time()
        arr = np.zeros(len(f_saw_values))
        for j, f in enumerate(f_saw_values):
            arr[j] = make_fmr_point(f, B0, ALPHA, K_mr=cfg['K_mr'],
                                     enable_barnett=cfg['barnett'],
                                     eps0=EPS0)
            if (j+1) % 5 == 0:
                print(f"    [{100*(j+1)/len(f_saw_values):5.1f}%] "
                      f"f={f*1e-9:.1f} GHz, |m_p|={arr[j]:.2e}")
        spectra[name] = arr
        print(f"    Done in {time.time()-t0:.0f} s")

    # (b) K_mr scaling at Kittel frequency
    print("\n  K_mr scaling sweep...")
    K_mr_values = np.array([0, 1, 2, 5, 8, 10, 15, 20]) * 1e6
    kmr_responses = np.zeros(len(K_mr_values))
    for j, K in enumerate(K_mr_values):
        kmr_responses[j] = make_fmr_point(f_kittel, B0, ALPHA,
                                           K_mr=K, eps0=EPS0)
        print(f"    K_mr={K*1e-6:.0f}: |m_p|={kmr_responses[j]:.2e}")

    # Save data
    raw_dir = os.path.join(DATA_DIR, "raw", "sim05_mr")
    np.savez(os.path.join(raw_dir, "spectra.npz"),
             f_saw=f_saw_values, **{k.replace('+','_').replace(' ','_'): v
                                     for k,v in spectra.items()})
    np.savez(os.path.join(raw_dir, "kmr_scaling.npz"),
             K_mr=K_mr_values, response=kmr_responses,
             f_kittel=f_kittel)
    with open(os.path.join(raw_dir, "manifest.json"), 'w') as f:
        json.dump({"sim": "sim05", "B0": B0, "alpha": ALPHA,
                    "eps0": EPS0, "K_mr_ref": K_MR}, f, indent=2)

    # Plot
    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.42)
    f_ghz = f_saw_values * 1e-9

    colors = {'MEL only': ps.BLUE, 'MEL+MR': ps.VERMILION, 'MEL+MR+B': ps.TEAL}
    styles = {'MEL only': '-', 'MEL+MR': '-', 'MEL+MR+B': '--'}

    for name, arr in spectra.items():
        maxv = np.max(arr) if np.max(arr) > 0 else 1
        axes[0].plot(f_ghz, arr / maxv, styles[name],
                     color=colors[name], linewidth=1.2, label=name)

    axes[0].axvline(f_kittel*1e-9, color='gray', linewidth=0.5,
                    linestyle=':', label=r'$f_\mathrm{Kittel}$')
    axes[0].set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    axes[0].set_ylabel(r'$|\langle m_\perp \rangle|$ (norm.)')
    axes[0].legend(fontsize=6, loc='upper right')
    ps.add_panel_label(axes[0], '(a)')

    axes[1].plot(K_mr_values*1e-6, kmr_responses*1e3, 'o-',
                 color=ps.BLUE, markersize=5,
                 markeredgecolor='white', markeredgewidth=0.4)
    axes[1].set_xlabel(r'$K_\mathrm{mr}$ (MJ/m$^3$)')
    axes[1].set_ylabel(r'$|\langle m_\perp \rangle|$ ($\times 10^{-3}$)')
    axes[1].set_xlim(left=-0.5)
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, "fig_magneto_rotation_validation.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_magneto_rotation_validation.pdf"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_magneto_rotation_validation.pdf")

    return spectra, K_mr_values, kmr_responses


# ===================================================================
# SIM 06: Nonreciprocal SAW-magnon coupling
# ===================================================================

def run_sim06():
    """Nonreciprocal SAW-magnon coupling (+k vs -k)."""
    print("\n" + "="*60)
    print("SIM 06: Nonreciprocal SAW-magnon coupling")
    print("="*60)

    ALPHA = 0.005
    K_MR = 5e6
    EPS0 = 1e-3
    B0 = 50e-3

    # (a) Spectral sweep for +k, -k, and MEL-only at B0=50mT
    f_saw_values = np.linspace(1e9, 10e9, 20)

    print("\n  +k sweep...")
    t0 = time.time()
    plus_k = np.zeros(len(f_saw_values))
    for j, f in enumerate(f_saw_values):
        plus_k[j] = make_fmr_point(f, B0, ALPHA, K_mr=K_MR,
                                    enable_barnett=True, eps0=EPS0)
    print(f"    Done in {time.time()-t0:.0f} s")

    print("  -k sweep...")
    t0 = time.time()
    minus_k = np.zeros(len(f_saw_values))
    for j, f in enumerate(f_saw_values):
        minus_k[j] = make_fmr_point(f, B0, ALPHA, K_mr=-K_MR,
                                     enable_barnett=True, eps0=EPS0,
                                     phase=np.pi)
    print(f"    Done in {time.time()-t0:.0f} s")

    print("  MEL-only reference...")
    t0 = time.time()
    mel_only = np.zeros(len(f_saw_values))
    for j, f in enumerate(f_saw_values):
        mel_only[j] = make_fmr_point(f, B0, ALPHA, K_mr=0.0,
                                      enable_barnett=False, eps0=EPS0)
    print(f"    Done in {time.time()-t0:.0f} s")

    # (b) Field-dependent nonreciprocity at Kittel frequency
    print("\n  Field-dependent nonreciprocity...")
    B0_values = np.array([20, 50, 100, 150, 200]) * 1e-3
    nr_data = []
    for B0_val in B0_values:
        f_k = GAMMA * np.sqrt(B0_val*(B0_val+MU0*MSAT)) / (2*np.pi)
        mp = make_fmr_point(f_k, B0_val, ALPHA, K_mr=K_MR,
                             enable_barnett=True, eps0=EPS0)
        mm = make_fmr_point(f_k, B0_val, ALPHA, K_mr=-K_MR,
                             enable_barnett=True, eps0=EPS0, phase=np.pi)
        mref = make_fmr_point(f_k, B0_val, ALPHA, K_mr=0.0,
                               enable_barnett=False, eps0=EPS0)
        delta = mp - mm
        ratio = delta / max(mp, mm, 1e-20)
        nr_data.append({'B0': B0_val, 'f_k': f_k, 'm+': mp, 'm-': mm,
                         'ref': mref, 'delta': delta, 'ratio': ratio})
        print(f"    B0={B0_val*1e3:.0f}mT: +k={mp:.2e}, -k={mm:.2e}, "
              f"ratio={ratio:.4f}")

    # Save
    raw_dir = os.path.join(DATA_DIR, "raw", "sim06_nonreciprocal")
    np.savez(os.path.join(raw_dir, "spectra.npz"),
             f_saw=f_saw_values, plus_k=plus_k,
             minus_k=minus_k, mel_only=mel_only)
    np.savez(os.path.join(raw_dir, "nonreciprocity.npz"),
             B0=B0_values,
             m_plus=[d['m+'] for d in nr_data],
             m_minus=[d['m-'] for d in nr_data],
             ratio=[d['ratio'] for d in nr_data])
    with open(os.path.join(raw_dir, "manifest.json"), 'w') as f:
        json.dump({"sim": "sim06", "K_mr": K_MR, "eps0": EPS0}, f, indent=2)

    # Plot
    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.42)
    f_ghz = f_saw_values * 1e-9

    max_pk = max(np.max(plus_k), 1e-20)
    max_mk = max(np.max(minus_k), 1e-20)
    max_all = max(max_pk, max_mk)

    axes[0].plot(f_ghz, plus_k/max_all, color=ps.BLUE, linewidth=1.2,
                 label=r'$+k$ (forward)')
    axes[0].plot(f_ghz, minus_k/max_all, '--', color=ps.VERMILION,
                 linewidth=1.2, label=r'$-k$ (backward)')
    if np.max(mel_only) > 0:
        axes[0].plot(f_ghz, mel_only/np.max(mel_only)*0.3, ':',
                     color='gray', linewidth=0.8, label='MEL only')

    axes[0].fill_between(f_ghz, plus_k/max_all, minus_k/max_all,
                         alpha=0.12, color=ps.TEAL)
    axes[0].set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    axes[0].set_ylabel(r'$|\langle m_\perp \rangle|$ (norm.)')
    axes[0].legend(fontsize=6, loc='upper right')
    ps.add_panel_label(axes[0], '(a)')

    ratios = [abs(d['ratio'])*100 for d in nr_data]
    axes[1].plot(B0_values*1e3, ratios, 'o-', color=ps.BLUE,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.4)
    axes[1].set_xlabel(r'$B_0$ (mT)')
    axes[1].set_ylabel(r'$|\Delta m / m_\mathrm{max}|$ (%)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, "fig_nonreciprocal_saw.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_nonreciprocal_saw.pdf"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_nonreciprocal_saw.pdf")

    return plus_k, minus_k, mel_only, nr_data


# ===================================================================
# SIM 07: Barnett field validation
# ===================================================================

def run_sim07():
    """Barnett effective field validation."""
    print("\n" + "="*60)
    print("SIM 07: Barnett field validation")
    print("="*60)

    B0 = 50e-3
    ALPHA = 0.005
    EPS0 = 5e-3  # larger strain for Barnett visibility

    f_kittel = GAMMA * np.sqrt(B0 * (B0 + MU0 * MSAT)) / (2*np.pi)
    print(f"  Kittel freq: {f_kittel*1e-9:.2f} GHz")

    # Barnett field scaling
    print("\n  Barnett field scaling:")
    f_scaling = np.array([1, 2, 3, 5, 8, 10]) * 1e9
    H_B = XI * EPS0 * 2*np.pi*f_scaling / (2*GAMMA)
    for f, h in zip(f_scaling, H_B):
        print(f"    f={f*1e-9:.0f} GHz: H_B = {h*1e6:.1f} uT")

    # Spectral comparison: MR-only vs MR+Barnett
    print("\n  Spectral sweep...")
    f_saw_values = np.linspace(1e9, 10e9, 20)

    print("    MR only...")
    mr_only = np.zeros(len(f_saw_values))
    for j, f in enumerate(f_saw_values):
        mr_only[j] = make_fmr_point(f, B0, ALPHA, K_mr=5e6,
                                     enable_barnett=False, eps0=EPS0)

    print("    MR + Barnett...")
    mr_barnett = np.zeros(len(f_saw_values))
    for j, f in enumerate(f_saw_values):
        mr_barnett[j] = make_fmr_point(f, B0, ALPHA, K_mr=5e6,
                                        enable_barnett=True, eps0=EPS0)

    # Save
    raw_dir = os.path.join(DATA_DIR, "raw", "sim07_barnett")
    np.savez(os.path.join(raw_dir, "spectra.npz"),
             f_saw=f_saw_values, mr_only=mr_only, mr_barnett=mr_barnett)
    np.savez(os.path.join(raw_dir, "scaling.npz"),
             f_scaling=f_scaling, H_B=H_B)
    with open(os.path.join(raw_dir, "manifest.json"), 'w') as f:
        json.dump({"sim": "sim07", "eps0": EPS0, "B0": B0}, f, indent=2)

    # Plot
    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.42)
    f_ghz = f_saw_values * 1e-9

    max_v = max(np.max(mr_only), np.max(mr_barnett), 1e-20)
    axes[0].plot(f_ghz, mr_only/max_v, color=ps.BLUE, linewidth=1.2,
                 label='MR only')
    axes[0].plot(f_ghz, mr_barnett/max_v, '--', color=ps.VERMILION,
                 linewidth=1.2, label='MR + Barnett')
    axes[0].set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    axes[0].set_ylabel(r'$|\langle m_\perp \rangle|$ (norm.)')
    axes[0].legend(fontsize=7)
    ps.add_panel_label(axes[0], '(a)')

    axes[1].plot(f_scaling*1e-9, H_B*1e6, 'o-', color=ps.BLUE,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.4)
    f_line = np.linspace(0, f_scaling[-1], 100)
    H_line = XI * EPS0 * 2*np.pi*f_line / (2*GAMMA)
    axes[1].plot(f_line*1e-9, H_line*1e6, '--', color=ps.VERMILION,
                 linewidth=0.9, label=r'$\xi\varepsilon_0\omega/2\gamma$')
    axes[1].set_xlabel(r'$f_\mathrm{SAW}$ (GHz)')
    axes[1].set_ylabel(r'$H_\mathrm{Barnett}$ ($\mu$T)')
    axes[1].legend(fontsize=7)
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, "fig_barnett_validation.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_barnett_validation.pdf"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_barnett_validation.pdf")

    return mr_only, mr_barnett


# ===================================================================
# SIM 08: Standing SAW - spatially resolved EP search
# ===================================================================

def run_sim08():
    """Standing SAW cavity with spatially resolved coupling analysis."""
    print("\n" + "="*60)
    print("SIM 08: Standing SAW - Spatially resolved EP search")
    print("="*60)

    ALPHA = 0.01
    K_MR = 8e6
    EPS0 = 3e-3
    B0 = 50e-3
    F_SAW = 2.0e9
    LAMBDA_SAW = V_SAW / F_SAW

    # Grid: 1D strip, ~2 wavelengths
    CX = 20e-9
    NX = int(2 * LAMBDA_SAW / CX)
    NX = min(NX, 256)  # cap for speed
    CY, CZ = 30e-9, 1e-9

    L_total = NX * CX
    print(f"  Grid: {NX}x1x1, CX={CX*1e9:.0f} nm")
    print(f"  L = {L_total*1e6:.1f} um = {L_total/LAMBDA_SAW:.1f} lambda")
    print(f"  SAW: f={F_SAW*1e-9:.1f} GHz, lambda={LAMBDA_SAW*1e6:.1f} um")

    # Analytical coupling map
    x_pos = np.arange(NX) * CX
    k = 2*np.pi / LAMBDA_SAW

    saw_obj = StandingSAW(F_SAW, LAMBDA_SAW, EPS0, ellipticity=XI,
                          K_mr=K_MR, enable_barnett=True)
    coupling = saw_obj.local_coupling_map(x_pos, B1, MSAT)
    print(f"  g_mel range: {np.min(coupling['g_mel'])*1e-9:.2f} - "
          f"{np.max(coupling['g_mel'])*1e-9:.2f} GHz")
    print(f"  g_mr range:  {np.min(coupling['g_mr'])*1e-9:.2f} - "
          f"{np.max(coupling['g_mr'])*1e-9:.2f} GHz")

    T_TOTAL = 15e-9
    DT_SAVE = 0.05e-9
    N_SAVE = int(T_TOTAL / DT_SAVE) + 1

    def run_1d_sim(label, K_mr_val, enable_barnett):
        """Run a 1D strip simulation and return m_y(t,x)."""
        print(f"\n  Running: {label}...")
        t0 = time.time()

        cellsize = (CX, CY, CZ)
        world = World(cellsize)
        magnet = Ferromagnet(world, Grid((NX, 1, 1)))

        magnet.msat = MSAT
        magnet.aex = AEX
        magnet.alpha = ALPHA
        magnet.magnetization = (1, 0, 0)
        magnet.bias_magnetic_field = (B0, 0, 0)
        magnet.enable_demag = False
        magnet.B1 = B1
        magnet.B2 = 0.0

        saw = StandingSAW(F_SAW, LAMBDA_SAW, EPS0,
                          direction='x', ellipticity=XI,
                          K_mr=K_mr_val, enable_barnett=enable_barnett)
        saw.apply(magnet, Msat=MSAT)

        m_y_tx = np.zeros((N_SAVE, NX))
        m_z_tx = np.zeros((N_SAVE, NX))

        m_init = magnet.magnetization.eval()
        m_y_tx[0, :] = m_init[1, 0, 0, :]
        m_z_tx[0, :] = m_init[2, 0, 0, :]

        for i in range(1, N_SAVE):
            world.timesolver.run(DT_SAVE)
            m = magnet.magnetization.eval()
            m_y_tx[i, :] = m[1, 0, 0, :]
            m_z_tx[i, :] = m[2, 0, 0, :]

            if (i+1) % (N_SAVE//3) == 0:
                max_my = np.max(np.abs(m_y_tx[i, :]))
                print(f"    [{100*(i+1)/N_SAVE:5.1f}%] max|m_y|={max_my:.2e}")

        dt = T_TOTAL / (N_SAVE - 1)
        print(f"    Done in {time.time()-t0:.0f} s")
        return np.linspace(0, T_TOTAL, N_SAVE), m_y_tx, m_z_tx

    # Run simulations
    t_full, my_full, mz_full = run_1d_sim("MEL+MR+B", K_MR, True)
    t_mel, my_mel, mz_mel = run_1d_sim("MEL only", 0.0, False)
    t_mr, my_mr, mz_mr = run_1d_sim("MR only", K_MR, False)

    # Local spectral analysis
    print("\n  Computing local spectra...")
    dt = t_full[1] - t_full[0]
    win_cells = min(32, NX // 4)

    def local_analysis(my_tx, label):
        x_centers, freqs, spectra = windowed_local_spectrum(
            my_tx, dt, x_pos, CX,
            window_width_cells=win_cells, overlap=0.5)

        freq_map = np.full((len(x_centers), 2), np.nan)
        lw_map = np.full((len(x_centers), 2), np.nan)
        regimes = []

        for i in range(len(x_centers)):
            peaks = extract_peaks(freqs, spectra[i], n_peaks=2,
                                  prominence=0.05)
            for j, p in enumerate(peaks[:2]):
                if p['success']:
                    freq_map[i, j] = p['f0']
                    lw_map[i, j] = p['fwhm']

            if len(peaks) >= 2:
                sp = abs(peaks[1]['f0'] - peaks[0]['f0'])
                alw = (peaks[0]['fwhm'] + peaks[1]['fwhm']) / 2
                r, C = classify_coupling_regime(sp, alw)
            else:
                r, C = 'single', 0.0
            regimes.append((r, C))

        regime_counts = {}
        for r, _ in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        print(f"    {label} regimes: {regime_counts}")

        return {
            'x_centers': x_centers,
            'freqs': freqs,
            'spectra': spectra,
            'freq_map': freq_map,
            'linewidth_map': lw_map,
            'regimes': regimes,
        }

    data_full = local_analysis(my_full, "Full")
    data_mel = local_analysis(my_mel, "MEL")
    data_mr = local_analysis(my_mr, "MR")

    # Save data
    raw_dir = os.path.join(DATA_DIR, "raw", "sim08_standing_ep")
    np.savez(os.path.join(raw_dir, "raw_full.npz"),
             time=t_full, m_y=my_full, m_z=mz_full)
    np.savez(os.path.join(raw_dir, "raw_mel.npz"),
             time=t_mel, m_y=my_mel, m_z=mz_mel)
    np.savez(os.path.join(raw_dir, "raw_mr.npz"),
             time=t_mr, m_y=my_mr, m_z=mz_mr)

    proc_dir = os.path.join(DATA_DIR, "processed", "maps")
    for name, data in [("full", data_full), ("mel", data_mel), ("mr", data_mr)]:
        np.savez(os.path.join(proc_dir, f"spatial_map_{name}.npz"),
                 x_centers=data['x_centers'], freqs=data['freqs'],
                 spectra=data['spectra'], freq_map=data['freq_map'],
                 linewidth_map=data['linewidth_map'])

    with open(os.path.join(raw_dir, "manifest.json"), 'w') as f:
        json.dump({"sim": "sim08", "K_mr": K_MR, "eps0": EPS0,
                    "f_SAW": F_SAW, "NX": NX, "CX": CX,
                    "T_total": T_TOTAL}, f, indent=2)

    # Plot: 4-panel figure
    fig, axes = ps.quad_panel(height_cm=13.0, wspace=0.40, hspace=0.50)

    f_kittel = GAMMA * np.sqrt(B0*(B0+MU0*MSAT)) / (2*np.pi)

    def plot_spectrum_map(ax, data, title):
        if data['spectra'].size == 0:
            return
        x_um = data['x_centers'] * 1e6
        f_ghz = data['freqs'] * 1e-9
        f_max = min(3*f_kittel*1e-9, f_ghz[-1])
        f_min = max(0.1, f_ghz[1])
        f_mask = (f_ghz >= f_min) & (f_ghz <= f_max)

        S = data['spectra'][:, f_mask].T
        max_S = np.max(S)
        if max_S > 0:
            S = S / max_S
        ext = [x_um[0], x_um[-1], f_ghz[f_mask][0], f_ghz[f_mask][-1]]
        ax.imshow(S, aspect='auto', origin='lower', extent=ext,
                  cmap='inferno', vmin=0, vmax=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r'$f$ (GHz)')

    plot_spectrum_map(axes[0, 0], data_full, 'MEL + MR + Barnett')
    ps.add_panel_label(axes[0, 0], '(a)')

    plot_spectrum_map(axes[0, 1], data_mr, 'MR only')
    ps.add_panel_label(axes[0, 1], '(b)')

    # (c) Analytical coupling map
    x_um = x_pos * 1e6
    g_mel_norm = np.abs(np.cos(k * x_pos))
    g_mr_norm = np.abs(np.sin(k * x_pos))
    g_total = np.sqrt(g_mel_norm**2 + (K_MR/abs(B1))**2 * g_mr_norm**2)

    axes[1, 0].plot(x_um, g_mel_norm, color=ps.BLUE, linewidth=1.0,
                    label=r'$|g_\mathrm{MEL}|$')
    axes[1, 0].plot(x_um, g_mr_norm * K_MR / abs(B1),
                    color=ps.VERMILION, linewidth=1.0,
                    label=r'$|g_\mathrm{MR}|$')
    axes[1, 0].plot(x_um, g_total, '--', color='black', linewidth=0.8,
                    label=r'$g_\mathrm{total}$')

    # Mark strain nodes and antinodes
    for n in range(5):
        x_node = n * LAMBDA_SAW / 2  # strain antinodes at kx = n*pi
        if x_node <= L_total:
            axes[1, 0].axvline(x_node*1e6, color=ps.BLUE,
                               linewidth=0.3, linestyle=':')
        x_rot = (n + 0.5) * LAMBDA_SAW / 2  # rotation antinodes
        if x_rot <= L_total:
            axes[1, 0].axvline(x_rot*1e6, color=ps.VERMILION,
                               linewidth=0.3, linestyle=':')

    axes[1, 0].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 0].set_ylabel('Coupling (norm.)')
    axes[1, 0].legend(fontsize=5.5, loc='upper right')
    ps.add_panel_label(axes[1, 0], '(c)')

    # (d) Position-resolved max amplitude
    max_my_full = np.max(np.abs(my_full), axis=0)
    max_my_mel = np.max(np.abs(my_mel), axis=0)
    max_my_mr = np.max(np.abs(my_mr), axis=0)

    axes[1, 1].plot(x_um, max_my_full, color=ps.TEAL, linewidth=0.8,
                    label='Full')
    axes[1, 1].plot(x_um, max_my_mel, ':', color=ps.BLUE, linewidth=0.8,
                    label='MEL')
    axes[1, 1].plot(x_um, max_my_mr, '--', color=ps.VERMILION,
                    linewidth=0.8, label='MR')
    axes[1, 1].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 1].set_ylabel(r'max$|m_y|$')
    axes[1, 1].legend(fontsize=6, loc='upper right')
    ps.add_panel_label(axes[1, 1], '(d)')

    fig.savefig(os.path.join(FIG_DIR, "fig_standing_saw_ep.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_standing_saw_ep.pdf"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_standing_saw_ep.pdf")

    return data_full, data_mel, data_mr


# ===================================================================
# Coupling channel decomposition figure
# ===================================================================

def make_coupling_decomposition_figure():
    """Create a schematic figure showing the three coupling channels."""
    print("\n  Creating coupling decomposition figure...")

    fig, axes = ps.double_panel_h(height_cm=6.5, wspace=0.42)

    # (a) Phase relationships for traveling SAW
    t = np.linspace(0, 2*np.pi, 200)

    # eps_xx ~ sin(kx - wt), plotted at fixed x=0 vs wt
    axes[0].plot(t/(2*np.pi), np.sin(t), color=ps.BLUE, linewidth=1.2,
                 label=r'$\varepsilon_{xx} \sim \sin(\omega t)$')
    axes[0].plot(t/(2*np.pi), np.cos(t), color=ps.VERMILION, linewidth=1.2,
                 label=r'$\Omega_y \sim \cos(\omega t)$')
    axes[0].plot(t/(2*np.pi), 0.1*np.sin(t), ':', color=ps.TEAL,
                 linewidth=1.0,
                 label=r'$\omega_y \sim \sin(\omega t)$ ($\times 0.1$)')

    axes[0].set_xlabel(r'$\omega t / 2\pi$')
    axes[0].set_ylabel('Amplitude (norm.)')
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=6, loc='upper right')
    axes[0].axhline(0, color='gray', linewidth=0.3)
    ps.add_panel_label(axes[0], '(a)')

    # (b) Standing wave: coupling vs position
    kx = np.linspace(0, 2*np.pi, 200)
    g_mel = np.abs(np.cos(kx))
    g_mr = np.abs(np.sin(kx))
    g_total = np.sqrt(g_mel**2 + (0.5)**2 * g_mr**2)  # K_mr/B1 ratio

    axes[1].fill_between(kx/np.pi, 0, g_mel, alpha=0.2, color=ps.BLUE)
    axes[1].fill_between(kx/np.pi, 0, 0.5*g_mr, alpha=0.2,
                         color=ps.VERMILION)
    axes[1].plot(kx/np.pi, g_mel, color=ps.BLUE, linewidth=1.2,
                 label=r'$g_\mathrm{MEL} \sim |\cos(kx)|$')
    axes[1].plot(kx/np.pi, 0.5*g_mr, color=ps.VERMILION, linewidth=1.2,
                 label=r'$g_\mathrm{MR} \sim |\sin(kx)|$')
    axes[1].plot(kx/np.pi, g_total, '--', color='black', linewidth=0.8,
                 label=r'$g_\mathrm{total}$')

    # Mark interesting positions
    axes[1].annotate('MEL\nmax', xy=(0, 1.02), fontsize=6, ha='center',
                     color=ps.BLUE)
    axes[1].annotate('MR\nmax', xy=(0.5, 0.52), fontsize=6, ha='center',
                     color=ps.VERMILION)

    axes[1].set_xlabel(r'$kx / \pi$')
    axes[1].set_ylabel('Coupling strength (norm.)')
    axes[1].set_xlim(0, 2)
    axes[1].legend(fontsize=6, loc='center right')
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, "fig_coupling_decomposition.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_coupling_decomposition.pdf"))
    plt.close(fig)
    print(f"  Saved: fig_coupling_decomposition.pdf")


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()

    print("="*60)
    print("RUNNING ALL NEW SAW-MAGNON SIMULATIONS")
    print("="*60)

    # Coupling decomposition figure (no simulation needed)
    make_coupling_decomposition_figure()

    # Run simulations
    sim05_data = run_sim05()
    sim06_data = run_sim06()
    sim07_data = run_sim07()
    sim08_data = run_sim08()

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL SIMULATIONS COMPLETE in {t_total:.0f} s "
          f"({t_total/60:.1f} min)")
    print(f"{'='*60}")

    # Save summary log
    log_file = os.path.join(DATA_DIR, "logs", "run_all_summary.txt")
    with open(log_file, 'w') as f:
        f.write(f"Run completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {t_total:.0f} s\n\n")
        f.write("Simulations:\n")
        f.write("  sim05: Magneto-rotation validation\n")
        f.write("  sim06: Nonreciprocal SAW-magnon coupling\n")
        f.write("  sim07: Barnett field validation\n")
        f.write("  sim08: Standing SAW EP search\n")
    print(f"\nSummary log: {log_file}")


if __name__ == "__main__":
    main()
