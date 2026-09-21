"""Generate Fig 4 for PRL: dynamical protection of the magnon-phonon EP.

3-panel figure showing:
  (a) Parameter-space loops inside/outside EP with marked EP position
  (b) Floquet monodromy matrix eigenvalues in complex plane
      for CCW vs CW, inside vs outside EP: demonstrates that CCW=CW
      (dynamical protection) in both cases
  (c) Figure of merit P_swap(inside)/P_swap(outside) versus loop period,
      showing that no parameter regime gives large ratio -> protection
      is not a fine-tuning artifact
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

try:
    import plot_style as ps
    HAS_STYLE = True
except ImportError:
    HAS_STYLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR_PAPER = os.path.join(SCRIPT_DIR, "..", "paper", "prl")
FIG_DIR_MAIN = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# =====================================================================
# Reuse sim13b physical constants
# =====================================================================
GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7
MS = 140e3
XI = 0.68
F_SAW = 3.0e9
OMEGA_SAW = 2 * np.pi * F_SAW
Q_SAW = 5000
KMR = 2e5


def kittel(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS))


def find_B0_res():
    omega = OMEGA_SAW
    a = 1.0
    b = MU0 * MS
    c = -(omega / GAMMA) ** 2
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def g_mr(eps0):
    return GAMMA * KMR * XI * eps0 / (2.0 * MS)


def H(eps0, B0, alpha, direction='+k'):
    om_m = kittel(B0)
    km = alpha * om_m
    kp = OMEGA_SAW / (2 * Q_SAW)
    gmr = g_mr(eps0)
    g = 1j * gmr if direction == '+k' else -1j * gmr
    return np.array([[om_m - 1j * km, g],
                     [np.conj(g), OMEGA_SAW - 1j * kp]], dtype=complex)


def find_ep(alpha):
    B0_res = find_B0_res()

    def obj(x):
        eps0, B0 = x
        if eps0 <= 0 or B0 <= 0:
            return 1e30
        HH = H(eps0, B0, alpha)
        w = np.linalg.eigvals(HH)
        return abs(w[0] - w[1])

    res = minimize(obj, [5e-5, B0_res], method='Nelder-Mead',
                   options={'xatol': 1e-12, 'fatol': 1e-3, 'maxiter': 800})
    return res.x[0], res.x[1]


def loop_trajectory(t, T_loop, center, radius, direction='CCW'):
    phase = 2 * np.pi * t / T_loop
    if direction == 'CW':
        phase = -phase
    eps0 = center[0] + radius[0] * np.cos(phase)
    B0 = center[1] + radius[1] * np.sin(phase)
    return eps0, B0


def compute_monodromy(T_loop, center, radius, alpha, direction='CCW'):
    def rhs(t, U_flat):
        U = U_flat.reshape(2, 2)
        eps0, B0 = loop_trajectory(t, T_loop, center, radius, direction)
        HH = H(eps0, B0, alpha)
        return (-1j * (HH @ U)).flatten()

    U0 = np.eye(2, dtype=complex).flatten()
    sol = solve_ivp(rhs, (0, T_loop), U0, method='DOP853',
                    t_eval=[T_loop], rtol=1e-11, atol=1e-13)
    return sol.y[:, -1].reshape(2, 2)


def compute_P_swap(U, H0):
    _, V_H = np.linalg.eig(H0)
    V_inv = np.linalg.inv(V_H)
    U_eig = V_inv @ U @ V_H
    diag_n = abs(U_eig[0, 0]) ** 2 + abs(U_eig[1, 1]) ** 2
    off_n = abs(U_eig[0, 1]) ** 2 + abs(U_eig[1, 0]) ** 2
    return off_n / (diag_n + 1e-30)


# =====================================================================
# Main figure generation
# =====================================================================
def main():
    # Use a representative alpha in the strong-coupling regime
    alpha = 1e-3
    eps0_ep, B0_ep = find_ep(alpha)
    print(f"EP at eps0 = {eps0_ep:.3e}, B0 = {B0_ep*1e3:.2f} mT, alpha = {alpha}")

    # Loop geometry
    radius_rel = 0.20
    d_eps = radius_rel * eps0_ep
    d_B = radius_rel * B0_ep

    center_in = (eps0_ep, B0_ep)
    center_out = (eps0_ep + 3 * d_eps, B0_ep)

    # ==== Panel (a): parameter-space loops ====
    if HAS_STYLE:
        ps.apply_style()

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.48, left=0.08, right=0.98,
                          bottom=0.20, top=0.93)

    ax = fig.add_subplot(gs[0, 0])
    phi = np.linspace(0, 2 * np.pi, 200)
    # inside
    eps_in = center_in[0] + d_eps * np.cos(phi)
    B_in = center_in[1] + d_B * np.sin(phi)
    ax.plot(eps_in * 1e5, B_in * 1e3, '-', color='C0', lw=1.4, label='inside')
    ax.fill(eps_in * 1e5, B_in * 1e3, color='C0', alpha=0.12)
    # outside
    eps_out = center_out[0] + d_eps * np.cos(phi)
    B_out = center_out[1] + d_B * np.sin(phi)
    ax.plot(eps_out * 1e5, B_out * 1e3, '-', color='C3', lw=1.4,
            label='outside')
    ax.fill(eps_out * 1e5, B_out * 1e3, color='C3', alpha=0.12)
    # EP
    ax.plot(eps0_ep * 1e5, B0_ep * 1e3, '*', color='gold',
            ms=13, mec='black', mew=0.5, zorder=10, label='EP')
    # Arrows (CCW)
    for loop_eps, loop_B, col in [(eps_in, B_in, 'C0'),
                                    (eps_out, B_out, 'C3')]:
        idx = 30
        ax.annotate('', xy=(loop_eps[idx + 8] * 1e5, loop_B[idx + 8] * 1e3),
                    xytext=(loop_eps[idx] * 1e5, loop_B[idx] * 1e3),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

    ax.set_xlabel(r'$\varepsilon_0\;(\times 10^{-5})$', fontsize=8)
    ax.set_ylabel(r'$B_0$ (mT)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='lower right', framealpha=0.9)
    ax.text(-0.25, 1.05, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold')

    # ==== Panel (b): Floquet eigenvalues in complex plane ====
    T_loop = 100e-9
    U_in_ccw = compute_monodromy(T_loop, center_in, (d_eps, d_B), alpha, 'CCW')
    U_in_cw = compute_monodromy(T_loop, center_in, (d_eps, d_B), alpha, 'CW')
    U_out_ccw = compute_monodromy(T_loop, center_out, (d_eps, d_B), alpha, 'CCW')
    U_out_cw = compute_monodromy(T_loop, center_out, (d_eps, d_B), alpha, 'CW')

    w_in_ccw = np.linalg.eigvals(U_in_ccw)
    w_in_cw = np.linalg.eigvals(U_in_cw)
    w_out_ccw = np.linalg.eigvals(U_out_ccw)
    w_out_cw = np.linalg.eigvals(U_out_cw)

    ax = fig.add_subplot(gs[0, 1])
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.5, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.3, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.3, alpha=0.5)

    # Plot with slight offset to show overlap
    ax.plot(w_in_ccw.real, w_in_ccw.imag, 'o', color='C0',
            ms=9, mec='black', mew=0.4, label='inside CCW')
    ax.plot(w_in_cw.real, w_in_cw.imag, 'x', color='navy',
            ms=9, mew=1.5, label='inside CW')
    ax.plot(w_out_ccw.real, w_out_ccw.imag, 's', color='C3',
            ms=8, mec='black', mew=0.4, label='outside CCW')
    ax.plot(w_out_cw.real, w_out_cw.imag, '+', color='darkred',
            ms=10, mew=1.5, label='outside CW')

    ax.set_xlabel(r'Re $\mu$', fontsize=8)
    ax.set_ylabel(r'Im $\mu$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc='best', framealpha=0.85)
    ax.set_aspect('equal')
    ax.text(-0.28, 1.05, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold')
    ax.set_title(r'Floquet multipliers', fontsize=8)

    # ==== Panel (c): P_swap sweep over loop parameters ====
    # Scan T_loop for fixed radius
    T_list = np.logspace(np.log10(10e-9), np.log10(1000e-9), 20)
    P_in_list = []
    P_out_list = []

    H0_in = H(center_in[0] + d_eps, center_in[1], alpha)
    H0_out = H(center_out[0] + d_eps, center_out[1], alpha)

    for T in T_list:
        U_in = compute_monodromy(T, center_in, (d_eps, d_B), alpha, 'CCW')
        U_out = compute_monodromy(T, center_out, (d_eps, d_B), alpha, 'CCW')
        P_in_list.append(compute_P_swap(U_in, H0_in))
        P_out_list.append(compute_P_swap(U_out, H0_out))

    P_in_arr = np.array(P_in_list)
    P_out_arr = np.array(P_out_list)

    ax = fig.add_subplot(gs[0, 2])
    ax.semilogx(T_list * 1e9, P_in_arr, 'o-', color='C0',
                ms=4, lw=1.2, mec='black', mew=0.3, label='inside')
    ax.semilogx(T_list * 1e9, P_out_arr, 's-', color='C3',
                ms=4, lw=1.2, mec='black', mew=0.3, label='outside')
    ax.axhline(1, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'$T_\mathrm{loop}$ (ns)', fontsize=8)
    ax.set_ylabel(r'$P_\mathrm{swap}$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='best', framealpha=0.9)
    ax.text(-0.30, 1.05, '(c)', transform=ax.transAxes,
            fontsize=10, fontweight='bold')
    ax.set_title(r'Dynamical protection', fontsize=8)

    # Save to paper dir
    out_paper = os.path.join(FIG_DIR_PAPER, "fig_prl_dynamical_protection")
    out_main = os.path.join(FIG_DIR_MAIN, "fig_prl_dynamical_protection")
    for out_base in [out_paper, out_main]:
        for ext in ['pdf', 'png']:
            fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_paper}.pdf")
    print(f"Saved: {out_main}.pdf")

    # Also save the panel-c data for the text
    print("\nPanel (c) data sample:")
    print(f"  T_loop ratio max: {np.max(P_in_arr/(P_out_arr+1e-30)):.3f}")
    print(f"  T_loop ratio min: {np.min(P_in_arr/(P_out_arr+1e-30)):.3f}")
    print(f"  Mean in: {np.mean(P_in_arr):.3f}, out: {np.mean(P_out_arr):.3f}")


if __name__ == "__main__":
    main()
