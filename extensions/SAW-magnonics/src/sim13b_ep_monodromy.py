"""Simulation 13b: Monodromy (Floquet) analysis of EP encircling.

Root-cause diagnosis for sim13: why does the "outside EP" loop also show
apparent chirality? This script computes the monodromy matrix U(T_loop)
numerically for each loop configuration and decomposes it in the
instantaneous eigenbasis of H(t=0). The clean topological signatures are:

  * Outside EP (adiabatic): U is DIAGONAL in the instantaneous eigenbasis
                            eigenvalues = exp(-i ∫ w_n(t) dt)
                            => state returns to itself (up to phase+decay)
  * Inside EP (topological encircling): U is ANTI-DIAGONAL
                            state |n⟩ -> phase × |m≠n⟩
                            monodromy eigenvalues ≈ ±i relative

Observable: P_swap = |U_{12} U_{21}| / |U_{11} U_{22}|.
    P_swap >> 1 : topological swap (chiral state transfer)
    P_swap << 1 : adiabatic (no swap)

We scan two control parameters:
    (i) the Gilbert damping alpha — smaller alpha reduces the passive
        mode-selection that masks the topological signature
    (ii) the loop period T_loop — balance between adiabaticity and decay

The goal: find a (alpha, T_loop, loop_radius) regime where P_swap(inside)
>> P_swap(outside) convincingly, proving the topology is accessible and
identifying the "sweet spot" for sim13 re-run with parameter C.
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
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim13b_monodromy.npz")


# ==========================================================================
# Physical constants & material (YIG, matching sim13)
# ==========================================================================
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


def hamiltonian(eps0, B0, alpha, Q=Q_SAW):
    """2x2 non-Hermitian H; +k direction (g = +i*g_mr)."""
    om_m = kittel(B0)
    km = alpha * om_m
    kp = OMEGA_SAW / (2 * Q)
    g = 1j * g_mr(eps0)
    return np.array([[om_m - 1j * km, g],
                     [np.conj(g), OMEGA_SAW - 1j * kp]], dtype=complex)


def find_ep(alpha):
    """EP where |g| = (kappa_m - kappa_p)/2 at resonance."""
    B0_res = find_B0_res()
    kappa_p = OMEGA_SAW / (2 * Q_SAW)

    def obj(x):
        eps0, B0 = x
        if eps0 <= 0 or B0 <= 0:
            return 1e30
        H = hamiltonian(eps0, B0, alpha)
        w = np.linalg.eigvals(H)
        return abs(w[0] - w[1])

    res = minimize(obj, [5e-5, B0_res], method='Nelder-Mead',
                   options={'xatol': 1e-12, 'fatol': 1e-2, 'maxiter': 800})
    return res.x[0], res.x[1], res.fun


# ==========================================================================
# Monodromy matrix computation
# ==========================================================================
def loop_trajectory(t, T_loop, center, radius, direction='CCW'):
    phase = 2 * np.pi * t / T_loop
    if direction == 'CW':
        phase = -phase
    eps0 = center[0] + radius[0] * np.cos(phase)
    B0 = center[1] + radius[1] * np.sin(phase)
    return eps0, B0


def compute_monodromy(T_loop, center, radius, alpha,
                      direction='CCW', n_samples=400):
    """Integrate dU/dt = -i H(t) U, U(0) = I, to get U(T_loop).

    Returns the 2x2 complex monodromy matrix.
    """
    def rhs(t, U_flat):
        U = U_flat.reshape(2, 2)
        eps0, B0 = loop_trajectory(t, T_loop, center, radius, direction)
        H = hamiltonian(eps0, B0, alpha)
        dU = -1j * (H @ U)
        return dU.flatten()

    U0 = np.eye(2, dtype=complex).flatten()
    sol = solve_ivp(rhs, (0, T_loop), U0, method='DOP853',
                    t_eval=[T_loop], rtol=1e-11, atol=1e-13)
    U_final = sol.y[:, -1].reshape(2, 2)
    return U_final


def analyze_monodromy(U, H0):
    """Express U in instantaneous eigenbasis of H0.

    Returns:
        U_eig     : U represented in H0-eigenbasis
        diag_norm : |U_eig[0,0]|^2 + |U_eig[1,1]|^2
        off_norm  : |U_eig[0,1]|^2 + |U_eig[1,0]|^2
        P_swap    : off_norm / diag_norm (topology figure of merit)
        w_U       : eigenvalues of U (Floquet multipliers)
    """
    w_H, V_H = np.linalg.eig(H0)
    # Normalize columns (V_H may not be orthogonal for non-Hermitian)
    V_H_inv = np.linalg.inv(V_H)
    U_eig = V_H_inv @ U @ V_H

    diag_norm = abs(U_eig[0, 0]) ** 2 + abs(U_eig[1, 1]) ** 2
    off_norm = abs(U_eig[0, 1]) ** 2 + abs(U_eig[1, 0]) ** 2

    # P_swap: ratio of off-diagonal to diagonal weight
    P_swap = off_norm / (diag_norm + 1e-30)

    w_U = np.linalg.eigvals(U)
    return dict(U_eig=U_eig, diag_norm=diag_norm, off_norm=off_norm,
                P_swap=P_swap, w_U=w_U, w_H=w_H)


# ==========================================================================
# Main diagnostic sweep
# ==========================================================================
def run_diagnostic():
    print("=" * 72)
    print("Sim 13b: Monodromy/Floquet diagnostic for EP encircling")
    print("=" * 72)

    # Scan 2 controls: Gilbert damping and loop period
    alpha_list = [5e-4, 1e-3, 2.37e-3, 5e-3]   # 4 values
    T_loop_list = [20e-9, 50e-9, 100e-9, 200e-9, 500e-9]  # 5 values
    radius_rel_list = [0.10, 0.20, 0.30]  # 3 loop radii (relative to EP)

    results = []

    print(f"\nScanning alpha x T_loop x radius = "
          f"{len(alpha_list)} x {len(T_loop_list)} x {len(radius_rel_list)} "
          f"= {len(alpha_list)*len(T_loop_list)*len(radius_rel_list)} configs")

    for alpha in alpha_list:
        eps0_ep, B0_ep, res = find_ep(alpha)
        print(f"\nalpha = {alpha:.1e}: "
              f"EP at (eps0={eps0_ep:.3e}, B0={B0_ep*1e3:.2f} mT) "
              f"| residue={res:.2e}")

        for T_loop in T_loop_list:
            for radius_rel in radius_rel_list:
                d_eps = radius_rel * eps0_ep
                d_B = radius_rel * B0_ep

                center_inside = (eps0_ep, B0_ep)
                center_outside = (eps0_ep + 3 * d_eps, B0_ep)

                H0_in = hamiltonian(center_inside[0] + d_eps,
                                    center_inside[1], alpha)
                H0_out = hamiltonian(center_outside[0] + d_eps,
                                     center_outside[1], alpha)

                # Inside loop CCW + CW
                U_in_ccw = compute_monodromy(T_loop, center_inside,
                                             (d_eps, d_B), alpha, 'CCW')
                U_in_cw = compute_monodromy(T_loop, center_inside,
                                            (d_eps, d_B), alpha, 'CW')
                U_out_ccw = compute_monodromy(T_loop, center_outside,
                                              (d_eps, d_B), alpha, 'CCW')

                a_in_ccw = analyze_monodromy(U_in_ccw, H0_in)
                a_in_cw = analyze_monodromy(U_in_cw, H0_in)
                a_out = analyze_monodromy(U_out_ccw, H0_out)

                results.append(dict(
                    alpha=alpha, T_loop=T_loop, radius_rel=radius_rel,
                    eps0_ep=eps0_ep, B0_ep=B0_ep,
                    P_swap_in_ccw=a_in_ccw['P_swap'],
                    P_swap_in_cw=a_in_cw['P_swap'],
                    P_swap_out=a_out['P_swap'],
                    U_in_ccw=U_in_ccw, U_out=U_out_ccw,
                    U_eig_in_ccw=a_in_ccw['U_eig'],
                    U_eig_out=a_out['U_eig'],
                    w_U_in_ccw=a_in_ccw['w_U'],
                    w_U_out=a_out['w_U'],
                ))

                print(f"  T={T_loop*1e9:4.0f}ns r={radius_rel:.2f}  "
                      f"P_swap: in(CCW)={a_in_ccw['P_swap']:6.2f}  "
                      f"in(CW)={a_in_cw['P_swap']:6.2f}  "
                      f"out={a_out['P_swap']:6.2f}  "
                      f"ratio={a_in_ccw['P_swap']/max(a_out['P_swap'], 1e-6):6.2f}")

    # Find best config: max (P_swap_inside / P_swap_outside)
    def figure_of_merit(r):
        return min(r['P_swap_in_ccw'], r['P_swap_in_cw']) / max(r['P_swap_out'], 1e-6)

    best = max(results, key=figure_of_merit)
    print(f"\n{'='*72}")
    print(f"BEST CONFIG:")
    print(f"  alpha    = {best['alpha']:.1e}")
    print(f"  T_loop   = {best['T_loop']*1e9:.0f} ns")
    print(f"  radius   = {best['radius_rel']:.2f} × EP")
    print(f"  eps0_EP  = {best['eps0_ep']:.3e}")
    print(f"  B0_EP    = {best['B0_ep']*1e3:.2f} mT")
    print(f"  P_swap(inside CCW) = {best['P_swap_in_ccw']:.3f}")
    print(f"  P_swap(inside CW)  = {best['P_swap_in_cw']:.3f}")
    print(f"  P_swap(outside)    = {best['P_swap_out']:.3f}")
    print(f"  Figure of merit    = "
          f"{figure_of_merit(best):.3f}")
    print(f"{'='*72}")

    return results, best


# ==========================================================================
# Plotting
# ==========================================================================
def plot_diagnostic(results, best):
    if HAS_STYLE:
        ps.apply_style()

    fig = plt.figure(figsize=(12, 8))

    alphas = sorted(set(r['alpha'] for r in results))
    T_loops = sorted(set(r['T_loop'] for r in results))
    radii = sorted(set(r['radius_rel'] for r in results))

    # ---- Panel grid: P_swap(in) / P_swap(out) heat map for each radius ----
    for ridx, radius in enumerate(radii):
        ax = fig.add_subplot(2, 3, ridx + 1)
        Z = np.zeros((len(alphas), len(T_loops)))
        for r in results:
            if r['radius_rel'] != radius:
                continue
            i = alphas.index(r['alpha'])
            j = T_loops.index(r['T_loop'])
            Z[i, j] = np.log10(
                min(r['P_swap_in_ccw'], r['P_swap_in_cw'])
                / max(r['P_swap_out'], 1e-6)
            )
        im = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-2, vmax=2,
                       extent=[min(T_loops) * 1e9, max(T_loops) * 1e9,
                               0, len(alphas) - 1])
        ax.set_xticks([T * 1e9 for T in T_loops])
        ax.set_xticklabels([f'{T*1e9:.0f}' for T in T_loops])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f'{a:.1e}' for a in alphas])
        ax.set_xlabel('$T_\\mathrm{loop}$ (ns)')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(f'radius = {radius:.2f} × EP')
        plt.colorbar(im, ax=ax, label=r'$\log_{10}(P_\mathrm{in}/P_\mathrm{out})$')

        # Mark best
        if best['radius_rel'] == radius:
            bi = alphas.index(best['alpha'])
            bj = T_loops.index(best['T_loop'])
            ax.plot(best['T_loop'] * 1e9, bi, 'k*', markersize=15,
                    markeredgecolor='yellow')

    # ---- Best config monodromy matrices ----
    ax = fig.add_subplot(2, 3, 4)
    U_in = np.abs(best['U_eig_in_ccw'])
    im = ax.imshow(U_in, cmap='viridis', vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(f'|U(inside CCW)| in eigenbasis\nalpha={best["alpha"]:.1e}, '
                 f'T={best["T_loop"]*1e9:.0f}ns')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{U_in[i, j]:.3f}', ha='center', va='center',
                    color='white' if U_in[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(2, 3, 5)
    U_out = np.abs(best['U_eig_out'])
    im = ax.imshow(U_out, cmap='viridis', vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title('|U(outside)| in eigenbasis')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{U_out[i, j]:.3f}', ha='center', va='center',
                    color='white' if U_out[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)

    # ---- Floquet eigenvalues in complex plane ----
    ax = fig.add_subplot(2, 3, 6)
    w_in = best['w_U_in_ccw']
    w_out = best['w_U_out']
    ax.plot(w_in.real, w_in.imag, 'bo', markersize=12, label='inside')
    ax.plot(w_out.real, w_out.imag, 'rs', markersize=12, label='outside')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Floquet multipliers (U eigenvalues)')
    ax.legend()
    ax.set_aspect('equal')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim13b_monodromy.{ext}'),
                    dpi=300)
    plt.close(fig)
    print(f"\n  Saved: fig_sim13b_monodromy.pdf/png")


def main():
    results, best = run_diagnostic()
    plot_diagnostic(results, best)

    # Save
    np.savez(CACHE_FILE,
             alpha=[r['alpha'] for r in results],
             T_loop=[r['T_loop'] for r in results],
             radius=[r['radius_rel'] for r in results],
             P_swap_in_ccw=[r['P_swap_in_ccw'] for r in results],
             P_swap_in_cw=[r['P_swap_in_cw'] for r in results],
             P_swap_out=[r['P_swap_out'] for r in results],
             best_alpha=best['alpha'],
             best_T_loop=best['T_loop'],
             best_radius=best['radius_rel'],
             best_eps0_ep=best['eps0_ep'],
             best_B0_ep=best['B0_ep'],
             )
    print(f"  Saved: {CACHE_FILE}")


if __name__ == "__main__":
    main()
