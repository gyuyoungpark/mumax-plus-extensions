"""Simulation 13: Dynamic encircling of exceptional point in SAW-magnon system.

Builds on the non-Hermitian 2-mode framework from sim09. The effective 2x2
Hamiltonian

    H(eps0, B0) = [ omega_m(B0) - i*kappa_m,    g(eps0)            ]
                  [ g*(eps0),                    omega_p - i*kappa_p ]

has an exceptional point (EP) at (eps0_EP, B0_EP) where the two eigenvalues
and eigenvectors coalesce. When the parameters (eps0, B0) are adiabatically
varied along a closed loop in parameter space:

  * Loop does NOT encircle EP  -> adiabatic, final state = initial state
  * Loop DOES encircle EP       -> chiral state transfer: the final state is
                                    the same eigenmode regardless of the
                                    initial state, selected by loop direction
                                    (CW vs CCW).

Reference: Doppler, Mailybaev, Bohm et al. Nature 537, 76 (2016);
           Xu, Mason, Jiang, Harris, Nature 537, 80 (2016);
           Hassan, Zhen, Soljacic, Khajavikhan, Christodoulides,
                  PRL 118, 093002 (2017).

This is a lightweight analytical/semi-analytical calculation (scipy ODE),
not a full mumax+ co-simulation. It predicts the chiral mode switching
signature that a subsequent micromagnetic validation would confirm.

Protocol:
  1. Locate the EP in (eps0, B0) plane (sim09 result + finer scan).
  2. Define two loops: one encircling EP ("inside"), one not ("outside").
  3. For each loop direction (CW, CCW), evolve 4 initial states through
     one full period T_loop.
  4. Plot parameter-space loops, eigenvalue Riemann sheets, Bloch-sphere
     trajectories, and final-state polarization summary.

Outputs:
    data/sim13_ep_encircling.npz
    figures/fig_ep_encircling.{pdf,png}
    figures/fig_ep_bloch_trajectories.{pdf,png}
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.integrate import solve_ivp

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

CACHE_FILE = os.path.join(DATA_DIR, "sim13_ep_encircling.npz")


# ==========================================================================
# Physical constants
# ==========================================================================
GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7

# ==========================================================================
# Material (YIG, matching sim09)
# ==========================================================================
MS = 140e3
XI = 0.68
F_SAW = 3.0e9
OMEGA_SAW = 2 * np.pi * F_SAW
Q_SAW = 5000

# Choose Kmr so that at reference eps0, EP is accessible
KMR = 2e5                 # J/m^3
EPS0_REF = 5e-5           # reference strain amplitude
ALPHA = 2.37e-3           # from sim09: alpha_EP_plus for current parameters


# ==========================================================================
# Kittel and coupling
# ==========================================================================
def kittel_frequency(B0, Ms=MS):
    """Kittel FMR frequency (rad/s) for in-plane film."""
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * Ms))


def find_resonance_field(f_target=F_SAW):
    """B0 that satisfies omega_m(B0) = 2*pi*f_target."""
    omega = 2 * np.pi * f_target
    a = 1.0
    b = MU0 * MS
    c = -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def coupling_mr(eps0, Kmr=KMR):
    """Magneto-rotation coupling (rad/s), assuming m0 || k_SAW (g_mel=0)."""
    return GAMMA * Kmr * XI * eps0 / (2.0 * MS)


def hamiltonian(eps0, B0, direction='+k', alpha=ALPHA, Q=Q_SAW):
    """Return the 2x2 non-Hermitian Hamiltonian in (mag, phon) basis.

    H = [[ omega_m - i*kappa_m,   g   ],
         [         g*,           omega_p - i*kappa_p ]]

    For m0 || k_SAW:  g(+k) = +i*g_mr,  g(-k) = -i*g_mr
    """
    omega_m = kittel_frequency(B0)
    kappa_m = alpha * omega_m
    kappa_p = OMEGA_SAW / (2 * Q)
    g_mr = coupling_mr(eps0)
    if direction == '+k':
        g = 1j * g_mr
    else:
        g = -1j * g_mr
    H = np.array([[omega_m - 1j * kappa_m, g],
                  [np.conj(g), OMEGA_SAW - 1j * kappa_p]], dtype=complex)
    return H


def eigvals_sorted(H):
    """Return eigenvalues sorted by real part."""
    w = np.linalg.eigvals(H)
    order = np.argsort(w.real)
    return w[order]


# ==========================================================================
# Locate EP in (eps0, B0) plane
# ==========================================================================
def find_ep(direction='+k', alpha=ALPHA, Q=Q_SAW):
    """Find the EP location in (eps0, B0) space.

    Uses a 2D Nelder-Mead minimization of the eigenvalue separation.
    """
    from scipy.optimize import minimize

    B0_res = find_resonance_field()

    def objective(x):
        eps0, B0 = x
        if eps0 <= 0 or B0 <= 0:
            return 1e30
        H = hamiltonian(eps0, B0, direction=direction, alpha=alpha, Q=Q)
        w = np.linalg.eigvals(H)
        return abs(w[0] - w[1])

    x0 = np.array([EPS0_REF, B0_res])
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-3, 'maxiter': 500})
    eps0_ep, B0_ep = res.x
    return eps0_ep, B0_ep, res.fun


# ==========================================================================
# Time-dependent parameter loop
# ==========================================================================
def parameter_loop(t, T_loop, center, radius, direction='CW'):
    """Elliptical loop in (eps0, B0) space centered at `center`.

    Parameters
    ----------
    t : float or array
        Time (s).
    T_loop : float
        Loop period (s).
    center : tuple (eps0_c, B0_c)
    radius : tuple (delta_eps, delta_B)
    direction : 'CW' or 'CCW'
    """
    phase = 2 * np.pi * t / T_loop
    if direction == 'CCW':
        phase = -phase
    eps0_c, B0_c = center
    d_eps, d_B = radius
    eps0 = eps0_c + d_eps * np.cos(phase)
    B0 = B0_c + d_B * np.sin(phase)
    return eps0, B0


# ==========================================================================
# Time evolution: Schrodinger-like equation with non-Hermitian H
# ==========================================================================
def evolve(psi0, t_span, loop_args, direction='+k', n_samples=400):
    """Evolve the 2-component state under time-varying H(t).

    Uses scipy.integrate.solve_ivp (DOP853, high accuracy).
    Returns time array and state trajectory (normalized).
    """
    def rhs(t, psi):
        eps0, B0 = parameter_loop(t, **loop_args)
        H = hamiltonian(eps0, B0, direction=direction)
        return -1j * (H @ psi)

    t_eval = np.linspace(*t_span, n_samples)
    sol = solve_ivp(rhs, t_span, psi0, method='DOP853',
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    psi_t = sol.y  # shape (2, n_samples)

    # Post-select: normalize each column (remove overall loss)
    norms = np.linalg.norm(psi_t, axis=0)
    psi_t_normalized = psi_t / norms[np.newaxis, :]
    return sol.t, psi_t, psi_t_normalized


def bloch_coords(psi):
    """Convert 2-component spinor to Bloch sphere (x, y, z).

    Using |0> as north pole, |1> as south pole.
    """
    a, b = psi[0], psi[1]
    norm = abs(a)**2 + abs(b)**2
    if norm < 1e-30:
        return 0.0, 0.0, 0.0
    sx = 2 * np.real(np.conj(a) * b) / norm
    sy = 2 * np.imag(np.conj(a) * b) / norm
    sz = (abs(a)**2 - abs(b)**2) / norm
    return sx, sy, sz


# ==========================================================================
# Eigenvalue Riemann sheets along loop
# ==========================================================================
def eigenvalue_trajectory(T_loop, loop_args, direction='+k', n_pts=400):
    """Compute eigenvalues of H(t) along the loop."""
    t_arr = np.linspace(0, T_loop, n_pts)
    w1 = np.zeros(n_pts, dtype=complex)
    w2 = np.zeros(n_pts, dtype=complex)
    for i, t in enumerate(t_arr):
        eps0, B0 = parameter_loop(t, **loop_args)
        H = hamiltonian(eps0, B0, direction=direction)
        w = np.linalg.eigvals(H)
        # Keep consistent ordering by minimizing jumps
        if i == 0:
            order = np.argsort(w.real)
        else:
            # Match to previous
            prev = np.array([w1[i - 1], w2[i - 1]])
            if abs(w[0] - prev[0]) + abs(w[1] - prev[1]) \
               <= abs(w[0] - prev[1]) + abs(w[1] - prev[0]):
                order = [0, 1]
            else:
                order = [1, 0]
        w1[i] = w[order[0]]
        w2[i] = w[order[1]]
    return t_arr, w1, w2


# ==========================================================================
# Main sweep
# ==========================================================================
def run_full_sweep():
    """Run all initial-state × direction × loop-type combinations."""
    print("=" * 72)
    print("Sim 13: Dynamic EP encircling")
    print("=" * 72)

    # ------- 1. Locate EP -------
    eps0_ep, B0_ep, residue = find_ep(direction='+k')
    print(f"\n  EP location (+k):")
    print(f"    eps0_EP = {eps0_ep:.4e}")
    print(f"    B0_EP   = {B0_ep * 1e3:.3f} mT")
    print(f"    |d(omega)| residue = {residue:.3e} rad/s")

    # Reference scales
    g_at_ep = coupling_mr(eps0_ep)
    omega_m_ep = kittel_frequency(B0_ep)
    kappa_m_ep = ALPHA * omega_m_ep
    kappa_p = OMEGA_SAW / (2 * Q_SAW)
    print(f"    g_mr(EP)/(2pi) = {g_at_ep / (2 * np.pi) * 1e-6:.3f} MHz")
    print(f"    |kappa_m - kappa_p|/2/(2pi) = "
          f"{abs(kappa_m_ep - kappa_p) / 2 / (2 * np.pi) * 1e-6:.3f} MHz")

    # ------- 2. Define two loops -------
    # Loop radius: ~20% relative deviation (large enough to enclose EP
    # but small enough to stay adiabatic)
    delta_eps_rel = 0.30
    delta_B_rel = 0.10

    loop_inside = dict(
        T_loop=200e-9,  # 200 ns, much longer than 1/g ~ 7 ns => adiabatic
        center=(eps0_ep, B0_ep),
        radius=(delta_eps_rel * eps0_ep, delta_B_rel * B0_ep),
    )
    # Outside loop: offset the center so EP is NOT enclosed
    offset = 2.5 * delta_eps_rel * eps0_ep
    loop_outside = dict(
        T_loop=200e-9,
        center=(eps0_ep + offset, B0_ep),
        radius=(delta_eps_rel * eps0_ep, delta_B_rel * B0_ep),
    )

    # ------- 3. Initial states (normalized) -------
    initial_states = {
        '|m>': np.array([1.0, 0.0], dtype=complex),
        '|p>': np.array([0.0, 1.0], dtype=complex),
        '|+>': np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
        '|-i>': np.array([1.0, -1j], dtype=complex) / np.sqrt(2),
    }

    results = {}
    for loop_name, loop_args in [('inside', loop_inside),
                                  ('outside', loop_outside)]:
        for direction_label in ['CW', 'CCW']:
            full_loop = dict(loop_args)
            full_loop['direction'] = direction_label
            T_loop = full_loop['T_loop']

            # Eigenvalue trajectory (does not depend on initial state)
            t_eig, w1, w2 = eigenvalue_trajectory(
                T_loop, full_loop, direction='+k', n_pts=600)

            for is_name, psi0 in initial_states.items():
                key = f"{loop_name}_{direction_label}_{is_name}"
                t_arr, psi_raw, psi_norm = evolve(
                    psi0, (0, T_loop), full_loop, direction='+k', n_samples=500)
                final = psi_norm[:, -1]
                sx, sy, sz = bloch_coords(final)
                results[key] = {
                    'time': t_arr,
                    'psi_norm': psi_norm,
                    'psi_raw': psi_raw,
                    'final': final,
                    'bloch_final': (sx, sy, sz),
                    'loop_name': loop_name,
                    'direction_label': direction_label,
                    'initial': is_name,
                    'w1': w1,
                    'w2': w2,
                    't_eig': t_eig,
                }
                print(f"  [{key}] final = "
                      f"({final[0].real:+.3f}{final[0].imag:+.3f}j, "
                      f"{final[1].real:+.3f}{final[1].imag:+.3f}j)  "
                      f"Bloch=({sx:+.2f}, {sy:+.2f}, {sz:+.2f})")

    # ------- 4. Save -------
    save_dict = {
        'eps0_EP': eps0_ep,
        'B0_EP': B0_ep,
        'g_mr_EP': g_at_ep,
        'kappa_m_EP': kappa_m_ep,
        'kappa_p': kappa_p,
        'loop_inside_center': np.array(loop_inside['center']),
        'loop_inside_radius': np.array(loop_inside['radius']),
        'loop_outside_center': np.array(loop_outside['center']),
        'loop_outside_radius': np.array(loop_outside['radius']),
        'T_loop': loop_inside['T_loop'],
    }
    for key, r in results.items():
        save_dict[f'{key}_time'] = r['time']
        save_dict[f'{key}_psi0_real'] = r['psi_norm'].real
        save_dict[f'{key}_psi0_imag'] = r['psi_norm'].imag
        save_dict[f'{key}_final_real'] = r['final'].real
        save_dict[f'{key}_final_imag'] = r['final'].imag
        save_dict[f'{key}_bloch'] = np.array(r['bloch_final'])
        save_dict[f'{key}_w1_real'] = r['w1'].real
        save_dict[f'{key}_w1_imag'] = r['w1'].imag
        save_dict[f'{key}_w2_real'] = r['w2'].real
        save_dict[f'{key}_w2_imag'] = r['w2'].imag
    np.savez(CACHE_FILE, **save_dict)
    print(f"\n  Data saved: {CACHE_FILE}")

    return results, (eps0_ep, B0_ep), (loop_inside, loop_outside)


# ==========================================================================
# Plotting
# ==========================================================================
def plot_parameter_space(results, ep_pos, loops):
    """Figure 1: parameter space loops + eigenvalue Riemann sheets."""
    if HAS_STYLE:
        ps.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    eps0_ep, B0_ep = ep_pos
    loop_inside, loop_outside = loops

    # ---- Panel (a): Parameter-space loops ----
    ax = axes[0]
    # EP marker
    ax.plot(eps0_ep * 1e5, B0_ep * 1e3, 'r*', ms=14, zorder=10,
            label='EP')

    for loop_name, loop, color in [('inside', loop_inside, 'C0'),
                                    ('outside', loop_outside, 'C1')]:
        phi = np.linspace(0, 2 * np.pi, 200)
        eps = loop['center'][0] + loop['radius'][0] * np.cos(phi)
        B = loop['center'][1] + loop['radius'][1] * np.sin(phi)
        ax.plot(eps * 1e5, B * 1e3, '-', color=color, lw=1.4,
                label=loop_name)
        # Arrow showing CCW direction
        idx = 30
        dx = -loop['radius'][0] * np.sin(phi[idx])
        dy = loop['radius'][1] * np.cos(phi[idx])
        ax.annotate('', xy=(eps[idx + 5] * 1e5, B[idx + 5] * 1e3),
                    xytext=(eps[idx] * 1e5, B[idx] * 1e3),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax.set_xlabel(r'$\varepsilon_0 \, (\times 10^{-5})$')
    ax.set_ylabel(r'$B_0$ (mT)')
    ax.set_title('(a) Parameter loops')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ---- Panel (b): Eigenvalue real-part (Riemann sheets) inside loop ----
    ax = axes[1]
    r_inside_CW = [r for k, r in results.items()
                   if 'inside_CW' in k][0]
    t = r_inside_CW['t_eig'] * 1e9
    ax.plot(t, r_inside_CW['w1'].real / (2 * np.pi * 1e9),
            '-', color='C0', lw=1.3, label=r'Re $\omega_1$')
    ax.plot(t, r_inside_CW['w2'].real / (2 * np.pi * 1e9),
            '-', color='C3', lw=1.3, label=r'Re $\omega_2$')
    ax.set_xlabel('$t$ (ns)')
    ax.set_ylabel(r'$\mathrm{Re}\,\omega / 2\pi$ (GHz)')
    ax.set_title('(b) Eigenvalue real part (inside loop)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- Panel (c): Riemann sheet in complex plane ----
    ax = axes[2]
    ax.plot(r_inside_CW['w1'].real / (2 * np.pi * 1e9),
            r_inside_CW['w1'].imag / (2 * np.pi * 1e6),
            '-', color='C0', lw=1.2, label=r'$\omega_1$')
    ax.plot(r_inside_CW['w2'].real / (2 * np.pi * 1e9),
            r_inside_CW['w2'].imag / (2 * np.pi * 1e6),
            '-', color='C3', lw=1.2, label=r'$\omega_2$')
    ax.set_xlabel(r'$\mathrm{Re}\,\omega / 2\pi$ (GHz)')
    ax.set_ylabel(r'$\mathrm{Im}\,\omega / 2\pi$ (MHz)')
    ax.set_title('(c) Riemann surface (inside)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f"fig_ep_encircling.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_ep_encircling.pdf/png")


def plot_bloch_sphere(results):
    """Figure 2: Bloch-sphere trajectories for the 4 initial states."""
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  Skipping Bloch plot (mpl_toolkits not available)")
        return

    if HAS_STYLE:
        ps.apply_style()

    fig = plt.figure(figsize=(12, 6))

    configs = [('inside', 'CW'), ('inside', 'CCW'),
               ('outside', 'CW'), ('outside', 'CCW')]

    for idx, (loop_name, dir_name) in enumerate(configs):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        # Sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(u.size), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.06, color='gray',
                        edgecolor='none')

        colors_map = {'|m>': 'C0', '|p>': 'C3', '|+>': 'C2', '|-i>': 'C4'}
        for is_name, color in colors_map.items():
            key = f"{loop_name}_{dir_name}_{is_name}"
            if key not in results:
                continue
            r = results[key]
            # Bloch trajectory
            traj = np.zeros((3, r['psi_norm'].shape[1]))
            for i in range(r['psi_norm'].shape[1]):
                traj[:, i] = bloch_coords(r['psi_norm'][:, i])
            ax.plot(traj[0], traj[1], traj[2], '-',
                    color=color, lw=1.0, alpha=0.7, label=is_name)
            ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0],
                       color=color, s=30, marker='o', edgecolors='k', lw=0.3)
            ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1],
                       color=color, s=70, marker='*', edgecolors='k', lw=0.3)

        ax.set_title(f"{loop_name}  {dir_name}", fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle('Bloch sphere trajectories (dots=start, stars=end)',
                 fontsize=10)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f"fig_ep_bloch_trajectories.{ext}"),
                    dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_ep_bloch_trajectories.pdf/png")


def plot_final_state_summary(results):
    """Figure 3: final state |<m|psi>|^2 for every configuration.

    KEY PHYSICS FIGURE: shows chiral state transfer when EP is encircled.
    """
    if HAS_STYLE:
        ps.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    initial_order = ['|m>', '|p>', '|+>', '|-i>']
    for panel_idx, loop_name in enumerate(['outside', 'inside']):
        ax = axes[panel_idx]
        cw_vals = []
        ccw_vals = []
        for is_name in initial_order:
            key_cw = f"{loop_name}_CW_{is_name}"
            key_ccw = f"{loop_name}_CCW_{is_name}"
            # Population in |m> at final time
            cw_vals.append(abs(results[key_cw]['final'][0])**2)
            ccw_vals.append(abs(results[key_ccw]['final'][0])**2)

        x = np.arange(len(initial_order))
        w = 0.35
        ax.bar(x - w / 2, cw_vals, w, color='C0', label='CW')
        ax.bar(x + w / 2, ccw_vals, w, color='C3', label='CCW')
        ax.set_xticks(x)
        ax.set_xticklabels(initial_order)
        ax.set_ylabel(r'$|\langle m | \psi_\mathrm{final} \rangle|^2$')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{loop_name} EP')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Final state magnon population vs loop direction',
                 fontsize=10)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f"fig_ep_chiral_transfer.{ext}"),
                    dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_ep_chiral_transfer.pdf/png")


# ==========================================================================
# Main
# ==========================================================================
def main():
    results, ep_pos, loops = run_full_sweep()
    plot_parameter_space(results, ep_pos, loops)
    plot_bloch_sphere(results)
    plot_final_state_summary(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
