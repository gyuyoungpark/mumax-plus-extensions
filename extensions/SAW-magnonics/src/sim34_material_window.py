"""Simulation 34 (Paper 2): Material accessibility window for EP encircling.

Computes, for a panel of common ferromagnetic materials, the strain
amplitude at which the SAW-magnon Hamiltonian crosses the exceptional
point and identifies which materials place that EP within the
experimentally accessible strain window
(eps_EP < 5e-3, the typical limit of high-power IDT-driven SAWs).

Materials surveyed:
    YIG       (Y3Fe5O12)
    CoFeB
    Permalloy (Ni81Fe19)
    Ni
    Co (hcp)
    Fe (bcc)
    Galfenol  (FeGa)
    Terfenol-D (TbDyFe)

Outputs:
  - data/sim34_material_window.npz: per-material parameters + EP coords
  - figures/fig_sim34_material_window.pdf: bar chart + table
  - stdout: latex-ready table for SI

Estimated runtime: < 2 minutes (analytic).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim34_material_window.npz")

GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7
F_REF = 3.0e9
OMEGA_P = 2 * np.pi * F_REF
KAPPA_P = 2 * np.pi * 5e6
XI = 0.68
EPS_ACCESSIBLE = 5e-3            # high-power IDT-driven SAW upper bound

# ---- Materials: Ms (A/m), B1 (J/m^3), Ku (J/m^3), alpha, Aex (J/m) ----
MATERIALS = {
    'YIG':       dict(Ms=140e3,  B1=-8.8e6,   Ku=0.0,   alpha=5e-4,  Aex=3.65e-12),
    'CoFeB':     dict(Ms=1.0e6,  B1=-8.0e6,   Ku=0.0,   alpha=5e-3,  Aex=19e-12),
    'Permalloy': dict(Ms=860e3,  B1=-7.0e6,   Ku=0.0,   alpha=8e-3,  Aex=13e-12),
    'Ni':        dict(Ms=480e3,  B1=9.2e6,    Ku=0.0,   alpha=4.5e-2, Aex=8e-12),
    'Co':        dict(Ms=1.4e6,  B1=-8.1e6,   Ku=4.5e5, alpha=1.1e-2, Aex=20e-12),
    'Fe':        dict(Ms=1.7e6,  B1=-3.4e6,   Ku=4.8e4, alpha=1.9e-3, Aex=21e-12),
    'Galfenol':  dict(Ms=1.4e6,  B1=-1.5e7,   Ku=2.0e5, alpha=1.0e-2, Aex=18e-12),
    'TerfenolD': dict(Ms=8.0e5,  B1=-1.5e8,   Ku=8.0e4, alpha=2.5e-2, Aex=9e-12),
}


def Kmr(par):
    return par['Ku'] + 0.5 * MU0 * par['Ms'] ** 2


def kittel_field(omega, Ms):
    a = 1.0
    b = MU0 * Ms
    c = -(omega / GAMMA) ** 2
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2


def main():
    print("=" * 78)
    print("Sim 34 (Paper 2): Material window for EP accessibility")
    print(f"  omega_p/2pi = {F_REF*1e-9:.2f} GHz")
    print(f"  kappa_p/2pi = {KAPPA_P/(2*np.pi)*1e-6:.2f} MHz")
    print(f"  Accessible strain bound = {EPS_ACCESSIBLE:.0e}")
    print("=" * 78)

    rows = []
    for name, par in MATERIALS.items():
        kmr = Kmr(par)
        kappa_m = par['alpha'] * OMEGA_P
        # EP condition: g_mr * eps_EP = (kappa_p - kappa_m)/2  (assumes >0)
        gap = abs(KAPPA_P - kappa_m) / 2
        g_per_eps = GAMMA * kmr * XI / (2 * par['Ms'])
        eps_EP = gap / g_per_eps
        B_EP = kittel_field(OMEGA_P, par['Ms'])
        accessible = eps_EP < EPS_ACCESSIBLE
        rows.append((name, par['Ms'], par['alpha'], kmr,
                     kappa_m / (2 * np.pi) * 1e-6, eps_EP, B_EP * 1e3,
                     accessible))

    # Print table
    print(f"  {'Material':<12s} {'Ms (kA/m)':>10s} {'alpha':>8s} "
          f"{'Kmr (MJ/m^3)':>14s} {'kappa_m (MHz)':>14s} "
          f"{'eps_EP':>10s} {'B_EP (mT)':>10s} {'Accessible':>10s}")
    for name, Ms, a, kmr, km, eps_EP, B_EP, acc in rows:
        flag = 'YES' if acc else 'no'
        print(f"  {name:<12s} {Ms*1e-3:>10.0f} {a:>8.1e} "
              f"{kmr*1e-6:>14.2f} {km:>14.1f} "
              f"{eps_EP:>10.2e} {B_EP:>10.1f} {flag:>10s}")

    # Save data
    save_dict = {
        'names': np.array([r[0] for r in rows]),
        'Ms': np.array([r[1] for r in rows]),
        'alpha': np.array([r[2] for r in rows]),
        'Kmr': np.array([r[3] for r in rows]),
        'kappa_m_MHz': np.array([r[4] for r in rows]),
        'eps_EP': np.array([r[5] for r in rows]),
        'B_EP_mT': np.array([r[6] for r in rows]),
        'accessible': np.array([r[7] for r in rows]),
    }
    np.savez(CACHE, **save_dict)
    print(f"\n  Saved: {CACHE}")

    plot(save_dict)
    latex_table(save_dict)


def plot(d):
    try:
        from plot_style import (apply_style, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        DOUBLE_COL = 7.0

    names = list(d['names'])
    eps = d['eps_EP']
    acc = d['accessible'].astype(bool)
    order = np.argsort(eps)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL / 3.0))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.92)
    colors = [TEAL if acc[i] else VERMILION for i in order]
    bars = ax.bar(range(len(order)), eps[order], color=colors,
                  edgecolor='k', linewidth=0.4)
    ax.axhline(EPS_ACCESSIBLE, color=BLACK, ls='--', lw=0.8,
               label=fr'IDT bound $\varepsilon_0={EPS_ACCESSIBLE:.0e}$')
    ax.set_yscale('log')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel(r'$\varepsilon_\mathrm{EP}$')
    ax.legend(fontsize=7, loc='upper left')

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim34_material_window.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim34_material_window.pdf/png")


def latex_table(d):
    print("\n  LaTeX table for SI (paste into supplemental.tex):")
    print(r"\begin{tabular}{lccccc}")
    print(r"Material & $M_s$ (kA/m) & $\alpha$ & $K_\mathrm{mr}$ (MJ/m$^3$) "
          r"& $\varepsilon_\mathrm{EP}$ & $B_\mathrm{EP}$ (mT) \\ \hline")
    for i, name in enumerate(d['names']):
        Ms = d['Ms'][i] * 1e-3
        a = d['alpha'][i]
        kmr = d['Kmr'][i] * 1e-6
        e = d['eps_EP'][i]
        B = d['B_EP_mT'][i]
        acc = '\\checkmark' if d['accessible'][i] else '$\\times$'
        print(f"{name} & {Ms:.0f} & {a:.1e} & {kmr:.2f} & {e:.1e} & {B:.1f} \\\\")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
