"""Simulation 9: Analytic Exceptional Point calculation for chiral SAW-magnon coupling.

Computes the eigenvalue structure of the 2x2 non-Hermitian Hamiltonian
for magnon-phonon hybridization with direction-dependent coupling.

    H(±k) = [ omega_m - i*kappa_m,     g(±k)     ]
             [ g*(±k),              omega_p - i*kappa_p ]

The chiral SAW coupling:
    g(+k) = g_mel + i*g_mr     (constructive rotation)
    g(-k) = g_mel - i*g_mr     (destructive rotation)

For m0 || k_SAW (in-plane, along SAW propagation):
    g_mel -> 0 (zero torque, H_mel || m0)
    g -> ±i * g_mr  (pure imaginary, from magneto-rotation)

This means the coupling matrix becomes non-symmetric:
    H(+k) has off-diagonal = +i*g_mr  (drives right-circular precession)
    H(-k) has off-diagonal = -i*g_mr  (drives left-circular precession)

Since the FMR mode has a definite chirality, the driving efficiency differs.
In the non-Hermitian framework, this leads to direction-dependent EP locations.

Material: YIG thin film on LiNbO3
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

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

# ==========================================================================
# Physical constants
# ==========================================================================
GAMMA = 1.76e11        # gyromagnetic ratio (rad/s/T)
MU0 = 4 * np.pi * 1e-7  # vacuum permeability (T*m/A)

# ==========================================================================
# Material parameters: YIG
# ==========================================================================
MS = 140e3             # saturation magnetization (A/m)
AEX = 3.65e-12         # exchange stiffness (J/m)
B1 = 3.48e6            # magnetoelastic coupling (J/m^3)
B2 = 6.96e6            # magnetoelastic coupling (J/m^3)
XI = 0.68              # Rayleigh wave ellipticity

# SAW parameters (LiNbO3 substrate)
F_SAW = 3.0e9          # SAW frequency (Hz)
OMEGA_SAW = 2 * np.pi * F_SAW
EPS0 = 5e-5            # peak strain amplitude


def kittel_frequency(B0, Ms=MS):
    """Kittel FMR frequency for in-plane magnetized thin film (rad/s).

    omega_FMR = gamma * sqrt(B0 * (B0 + mu0*Ms))
    """
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * Ms))


def coupling_rates(B0, Kmr, eps0=EPS0, xi=XI, Ms=MS):
    """Compute MEL and MR coupling rates (rad/s).

    For in-plane m0 || k_SAW:
        g_mel = 0  (zero torque: H_mel || m0, only normal strain eps_xx)
        g_mr = gamma * Kmr * xi * eps0 / (2 * Ms)

    H_mr [Tesla] = Kmr * xi * eps0 / (2 * Msat)
    g_mr = gamma * H_mr = gamma * Kmr * xi * eps0 / (2 * Ms)

    Returns (g_mel, g_mr) in rad/s.
    """
    # MEL: zero for m0 || k_SAW geometry (only normal strain, B1 term)
    g_mel = 0.0

    # MR: finite coupling
    # H_mr [T] = Kmr * xi * eps0 / (2 * Msat)
    # g_mr = gamma [rad/(s*T)] * H_mr [T]
    g_mr = GAMMA * Kmr * xi * eps0 / (2.0 * Ms)

    return g_mel, g_mr


def eigenvalues_2x2(omega_m, kappa_m, omega_p, kappa_p, g_complex):
    """Eigenvalues of the 2x2 non-Hermitian Hamiltonian.

    H = [ omega_m - i*kappa_m,     g     ]
        [ g*,              omega_p - i*kappa_p ]

    Returns omega_plus, omega_minus (complex).
    """
    avg = 0.5 * ((omega_m + omega_p) - 1j * (kappa_m + kappa_p))
    delta = 0.5 * ((omega_m - omega_p) - 1j * (kappa_m - kappa_p))
    discriminant = delta**2 + g_complex * np.conj(g_complex)  # = delta^2 + |g|^2

    sqrt_disc = np.sqrt(discriminant.astype(complex))

    return avg + sqrt_disc, avg - sqrt_disc


# ==========================================================================
# Calculation 1: Eigenvalue trajectories vs B0
# ==========================================================================
def calc_eigenvalue_trajectories(Kmr, alpha, Q_saw, eps0=EPS0):
    """Compute eigenvalue trajectories as B0 sweeps through FMR resonance.

    Parameters
    ----------
    Kmr : float
        Magneto-rotation coupling constant (J/m^3).
    alpha : float
        Gilbert damping parameter.
    Q_saw : float
        SAW cavity quality factor.
    eps0 : float
        Peak strain amplitude.

    Returns
    -------
    dict with B0 values and eigenvalues for +k and -k.
    """
    omega_p = OMEGA_SAW
    kappa_p = omega_p / (2 * Q_saw)  # phonon half-linewidth

    g_mel, g_mr = coupling_rates(0, Kmr, eps0)

    # B0 range: find B0 that gives omega_m = omega_p
    # omega_p = gamma * sqrt(B0_res * (B0_res + mu0*Ms))
    # Solving: B0_res^2 + mu0*Ms*B0_res - (omega_p/gamma)^2 = 0
    a_coeff = 1.0
    b_coeff = MU0 * MS
    c_coeff = -(omega_p / GAMMA)**2
    B0_res = (-b_coeff + np.sqrt(b_coeff**2 - 4*a_coeff*c_coeff)) / (2*a_coeff)

    print(f"  Resonance field: B0_res = {B0_res*1e3:.2f} mT")
    print(f"  FMR frequency at resonance: {kittel_frequency(B0_res)/(2*np.pi)*1e-9:.3f} GHz")
    print(f"  SAW frequency: {F_SAW*1e-9:.3f} GHz")

    # Sweep B0 around resonance
    B0_range = 0.3 * B0_res  # ±30% around resonance
    B0_values = np.linspace(B0_res - B0_range, B0_res + B0_range, 500)
    B0_values = B0_values[B0_values > 0]  # ensure positive

    # Coupling constants
    g_plus = g_mel + 1j * g_mr    # +k direction
    g_minus = g_mel - 1j * g_mr   # -k direction

    print(f"  |g(+k)|/(2pi) = {np.abs(g_plus)/(2*np.pi)*1e-6:.3f} MHz")
    print(f"  |g(-k)|/(2pi) = {np.abs(g_minus)/(2*np.pi)*1e-6:.3f} MHz")

    # Storage
    omega_plus_pk = np.zeros(len(B0_values), dtype=complex)
    omega_minus_pk = np.zeros(len(B0_values), dtype=complex)
    omega_plus_mk = np.zeros(len(B0_values), dtype=complex)
    omega_minus_mk = np.zeros(len(B0_values), dtype=complex)

    for i, B0 in enumerate(B0_values):
        omega_m = kittel_frequency(B0)
        kappa_m = alpha * omega_m  # magnon half-linewidth

        # +k eigenvalues
        w1, w2 = eigenvalues_2x2(omega_m, kappa_m, omega_p, kappa_p, g_plus)
        omega_plus_pk[i] = w1
        omega_minus_pk[i] = w2

        # -k eigenvalues
        w1, w2 = eigenvalues_2x2(omega_m, kappa_m, omega_p, kappa_p, g_minus)
        omega_plus_mk[i] = w1
        omega_minus_mk[i] = w2

    # Loss rates
    kappa_m_res = alpha * omega_p
    print(f"  kappa_m/(2pi) = {kappa_m_res/(2*np.pi)*1e-6:.3f} MHz (at resonance)")
    print(f"  kappa_p/(2pi) = {kappa_p/(2*np.pi)*1e-6:.3f} MHz")
    print(f"  |kappa_m - kappa_p|/(2*2pi) = {abs(kappa_m_res - kappa_p)/(2*2*np.pi)*1e-6:.3f} MHz")
    print(f"  Cooperativity C = |g|^2/(kappa_m*kappa_p) = "
          f"{np.abs(g_plus)**2/(kappa_m_res*kappa_p):.1f}")

    # EP condition: |g| = |kappa_m - kappa_p|/2
    delta_kappa = abs(kappa_m_res - kappa_p)
    g_ep = delta_kappa / 2
    print(f"  EP coupling: g_EP/(2pi) = {g_ep/(2*np.pi)*1e-6:.3f} MHz")

    regime = "STRONG" if np.abs(g_plus) > delta_kappa/2 else "WEAK"
    print(f"  Coupling regime: {regime}")

    return {
        'B0': B0_values,
        'B0_res': B0_res,
        'omega_plus_pk': omega_plus_pk,
        'omega_minus_pk': omega_minus_pk,
        'omega_plus_mk': omega_plus_mk,
        'omega_minus_mk': omega_minus_mk,
        'g_plus': g_plus,
        'g_minus': g_minus,
        'kappa_m_res': kappa_m_res,
        'kappa_p': kappa_p,
        'omega_p': omega_p,
        'alpha': alpha,
        'Kmr': Kmr,
        'Q_saw': Q_saw,
    }


# ==========================================================================
# Calculation 2: EP phase diagram in (alpha, Kmr) space
# ==========================================================================
def calc_ep_phase_diagram(Q_saw=5000, eps0=EPS0):
    """Compute EP phase diagram: coupling regime in (alpha, Kmr) space.

    EP condition: |g| = |kappa_m - kappa_p| / 2
    Strong coupling: |g| > |kappa_m - kappa_p| / 2
    Weak coupling:   |g| < |kappa_m - kappa_p| / 2

    For the chiral case, |g(+k)| = |g(-k)| = g_mr when g_mel = 0.
    So the EP boundary is the same for both directions in the (alpha, Kmr) plane.

    However, when g_mel != 0 (oblique m0), |g(+k)| != |g(-k)|, and the
    EP boundaries split.
    """
    omega_p = OMEGA_SAW
    kappa_p = omega_p / (2 * Q_saw)

    alpha_values = np.logspace(-5, -1, 200)
    Kmr_values = np.logspace(4, 7, 200)  # 10 kJ/m^3 to 10 MJ/m^3

    # For each (alpha, Kmr), compute cooperativity
    C_map = np.zeros((len(alpha_values), len(Kmr_values)))
    regime_map = np.zeros((len(alpha_values), len(Kmr_values)))  # 0=weak, 1=strong, 2=EP

    for i, alpha in enumerate(alpha_values):
        kappa_m = alpha * omega_p  # at resonance
        for j, Kmr in enumerate(Kmr_values):
            _, g_mr = coupling_rates(0, Kmr, eps0)
            g_abs = g_mr  # |g(+k)| = |g(-k)| = g_mr
            delta_kappa = abs(kappa_m - kappa_p)

            C_map[i, j] = g_abs**2 / (kappa_m * kappa_p) if kappa_m > 0 else 0

            if g_abs > delta_kappa / 2:
                regime_map[i, j] = 1  # strong coupling
            else:
                regime_map[i, j] = 0  # weak coupling

    # EP boundary: g_mr = |kappa_m - kappa_p| / 2
    # gamma * Kmr * xi * eps0 / (2 * Ms) = |alpha * omega_p - omega_p/(2Q)| / 2
    # Kmr_EP(alpha) = |alpha - 1/(2Q)| * omega_p * Ms / (gamma * xi * eps0)
    Kmr_EP = np.abs(alpha_values - 1/(2*Q_saw)) * omega_p * MS / (GAMMA * XI * eps0)

    return {
        'alpha': alpha_values,
        'Kmr': Kmr_values,
        'C_map': C_map,
        'regime_map': regime_map,
        'Kmr_EP': Kmr_EP,
    }


# ==========================================================================
# Calculation 3: Chiral EP with oblique magnetization
# ==========================================================================
def calc_chiral_ep_oblique(theta_deg, Kmr, alpha, Q_saw, eps0=EPS0):
    """When m0 is at angle theta to k_SAW, both g_mel and g_mr are active.

    g_mel(theta) = gamma * |B1| * eps0 * |sin(theta)| / (mu0 * Ms)
    g_mr(theta) = gamma * Kmr * xi * eps0 / (mu0 * Ms)  (independent of theta for PMA anisU=z)

    Actually for the general case with anisU = z:
    H_mr depends on m.z component, which varies with theta.

    Simplified model (linearized):
    g_mel ~ gamma * 2|B1| * eps0 * |sin(theta)| / (mu0 * Ms)
       (MEL torque proportional to transverse component of H_mel)
    g_mr ~ gamma * Kmr * xi * eps0 * |cos(theta)| / (mu0 * Ms)
       (MR field proportional to m.x component ≈ cos(theta))

    Actually let me derive more carefully for m0 = (cos_theta, 0, sin_theta):
    For in-plane rotation in x-z plane (theta from x-axis):

    MEL effective field for epsilon_xx:
    H_mel = -(2B1/mu0*Ms) * eps_xx * m_x * x_hat
    |H_mel| = (2|B1|/mu0*Ms) * eps0 * |cos(theta)|
    Torque = m x H_mel, component perp to m0:
    |tau| = |H_mel| * |sin(angle between m0 and H_mel)|
    H_mel is along x, m0 = (cos_theta, 0, sin_theta)
    sin(angle) = |sin(theta)|
    Effective coupling: g_mel ~ gamma * (2|B1|*eps0*|cos(theta)*sin(theta)|) / (mu0*Ms)
                      = gamma * |B1|*eps0*|sin(2*theta)| / (mu0*Ms)

    MR: H_mr is along z (for anisU=z, SAW along x)
    |H_mr| = (2Kmr/mu0*Ms) * Omega_y * |m_x| = (Kmr*xi*eps0/mu0*Ms) * |cos(theta)|
    Torque perp to m0: |tau| = |H_mr| * |sin(angle)|
    H_mr is along z, angle from m0 to z is (pi/2 - theta)
    sin(pi/2 - theta) = |cos(theta)|
    Effective coupling: g_mr ~ gamma * Kmr*xi*eps0*cos^2(theta) / (mu0*Ms)

    Hmm, this is getting complicated. Let me use a simpler model.
    """
    theta = np.deg2rad(theta_deg)

    # Effective coupling rates (simplified model)
    # H_mel [T] = 2|B1|*eps0*|cos(theta)|/Ms (field along x)
    # Torque perp to m0: H_perp = H_mel * |sin(theta)| = |B1|*eps0*|sin(2*theta)|/Ms
    # MEL: maximum at theta = 45°, zero at theta = 0° and 90°
    g_mel = GAMMA * abs(B1) * eps0 * abs(np.sin(2*theta)) / MS

    # MR: H_mr [T] = Kmr*xi*eps0/(2*Ms) * f(theta)
    # For anisU=z, m0 at angle theta from x in x-z plane:
    # MR field depends on m_x component ~ cos(theta)
    g_mr = GAMMA * Kmr * XI * eps0 * np.cos(theta)**2 / (2.0 * MS)

    # Chiral coupling
    g_plus = g_mel + 1j * g_mr
    g_minus = g_mel - 1j * g_mr

    return g_mel, g_mr, g_plus, g_minus


def calc_angle_dependent_ep(Kmr, alpha, Q_saw, eps0=EPS0):
    """Compute |g(+k)| and |g(-k)| vs field angle theta.

    Find the angle range where chiral EP appears.
    """
    omega_p = OMEGA_SAW
    kappa_p = omega_p / (2 * Q_saw)
    kappa_m = alpha * omega_p  # at resonance

    theta_values = np.linspace(0, 90, 181)

    g_mel_arr = np.zeros(len(theta_values))
    g_mr_arr = np.zeros(len(theta_values))
    g_plus_abs = np.zeros(len(theta_values))
    g_minus_abs = np.zeros(len(theta_values))
    nonrecip = np.zeros(len(theta_values))

    for i, theta in enumerate(theta_values):
        g_mel, g_mr, g_plus, g_minus = calc_chiral_ep_oblique(
            theta, Kmr, alpha, Q_saw, eps0)
        g_mel_arr[i] = g_mel
        g_mr_arr[i] = g_mr
        g_plus_abs[i] = np.abs(g_plus)
        g_minus_abs[i] = np.abs(g_minus)
        if g_plus_abs[i] + g_minus_abs[i] > 0:
            nonrecip[i] = (g_plus_abs[i] - g_minus_abs[i]) / (g_plus_abs[i] + g_minus_abs[i])

    ep_threshold = abs(kappa_m - kappa_p) / 2

    return {
        'theta': theta_values,
        'g_mel': g_mel_arr,
        'g_mr': g_mr_arr,
        'g_plus': g_plus_abs,
        'g_minus': g_minus_abs,
        'nonrecip': nonrecip,
        'ep_threshold': ep_threshold,
    }


# ==========================================================================
# Plotting
# ==========================================================================
def plot_eigenvalue_trajectories(result, filename="fig_ep_eigenvalues.pdf"):
    """Plot eigenvalue trajectories for +k and -k directions."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    B0_mT = result['B0'] * 1e3
    B0_res_mT = result['B0_res'] * 1e3
    f_saw_GHz = F_SAW * 1e-9

    # Convert to GHz
    re_plus_pk = result['omega_plus_pk'].real / (2*np.pi*1e9)
    re_minus_pk = result['omega_minus_pk'].real / (2*np.pi*1e9)
    im_plus_pk = -result['omega_plus_pk'].imag / (2*np.pi*1e6)  # MHz, positive = damping
    im_minus_pk = -result['omega_minus_pk'].imag / (2*np.pi*1e6)

    re_plus_mk = result['omega_plus_mk'].real / (2*np.pi*1e9)
    re_minus_mk = result['omega_minus_mk'].real / (2*np.pi*1e9)
    im_plus_mk = -result['omega_plus_mk'].imag / (2*np.pi*1e6)
    im_minus_mk = -result['omega_minus_mk'].imag / (2*np.pi*1e6)

    # Uncoupled dispersions
    omega_m_arr = np.array([kittel_frequency(B) for B in result['B0']])
    f_m = omega_m_arr / (2*np.pi*1e9)

    # (a) Re(omega) for +k
    axes[0, 0].plot(B0_mT, re_plus_pk, '-', color='C0', lw=1.5, label=r'$\omega_+$')
    axes[0, 0].plot(B0_mT, re_minus_pk, '-', color='C1', lw=1.5, label=r'$\omega_-$')
    axes[0, 0].plot(B0_mT, f_m, '--', color='gray', lw=0.8, alpha=0.5, label='uncoupled')
    axes[0, 0].axhline(f_saw_GHz, ls='--', color='gray', lw=0.8, alpha=0.5)
    axes[0, 0].set_ylabel(r'Re($\omega$) (GHz)')
    axes[0, 0].set_title(r'$+k$ (forward SAW)', fontsize=9)
    axes[0, 0].legend(fontsize=7)

    # (b) Re(omega) for -k
    axes[0, 1].plot(B0_mT, re_plus_mk, '-', color='C0', lw=1.5, label=r'$\omega_+$')
    axes[0, 1].plot(B0_mT, re_minus_mk, '-', color='C1', lw=1.5, label=r'$\omega_-$')
    axes[0, 1].plot(B0_mT, f_m, '--', color='gray', lw=0.8, alpha=0.5)
    axes[0, 1].axhline(f_saw_GHz, ls='--', color='gray', lw=0.8, alpha=0.5)
    axes[0, 1].set_title(r'$-k$ (backward SAW)', fontsize=9)
    axes[0, 1].legend(fontsize=7)

    # (c) Im(omega) for +k
    axes[1, 0].plot(B0_mT, im_plus_pk, '-', color='C0', lw=1.5)
    axes[1, 0].plot(B0_mT, im_minus_pk, '-', color='C1', lw=1.5)
    axes[1, 0].set_xlabel(r'$B_0$ (mT)')
    axes[1, 0].set_ylabel(r'$-$Im($\omega$)$/2\pi$ (MHz)')

    # (d) Im(omega) for -k
    axes[1, 1].plot(B0_mT, im_plus_mk, '-', color='C0', lw=1.5)
    axes[1, 1].plot(B0_mT, im_minus_mk, '-', color='C1', lw=1.5)
    axes[1, 1].set_xlabel(r'$B_0$ (mT)')

    # Mark resonance
    for ax in axes.flat:
        ax.axvline(B0_res_mT, ls=':', color='k', lw=0.5, alpha=0.3)

    fig.suptitle(
        rf'Chiral magnon-phonon hybridization ($K_{{mr}}$={result["Kmr"]*1e-6:.1f} MJ/m³, '
        rf'$\alpha$={result["alpha"]:.0e}, Q={result["Q_saw"]:.0f})',
        fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_ep_phase_diagram(result, filename="fig_ep_phase_diagram.pdf"):
    """Plot EP phase diagram in (alpha, Kmr) space."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    alpha = result['alpha']
    Kmr = result['Kmr'] * 1e-6  # to MJ/m^3
    C_map = result['C_map']

    # Cooperativity colormap
    im = ax.pcolormesh(alpha, Kmr, np.log10(C_map.T + 1e-10),
                       cmap='RdYlBu_r', shading='auto',
                       vmin=-2, vmax=4)
    cb = fig.colorbar(im, ax=ax, label=r'$\log_{10}(C)$')

    # EP boundary
    Kmr_EP = result['Kmr_EP'] * 1e-6
    ax.plot(alpha, Kmr_EP, 'k-', lw=2, label='EP boundary')

    # C=1 contour
    ax.contour(alpha, Kmr, C_map.T, levels=[1], colors='white',
               linewidths=1.5, linestyles='--')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Gilbert damping $\alpha$')
    ax.set_ylabel(r'$K_{mr}$ (MJ/m³)')
    ax.set_title('Coupling regime phase diagram', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    # Mark typical YIG parameters
    ax.plot(2e-4, 1.0, 'w*', ms=12, mec='k', mew=1.0, label='YIG')
    ax.annotate('YIG', (2e-4, 1.0), textcoords="offset points",
                xytext=(10, -5), fontsize=8, color='white',
                fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_angle_dependence(result, filename="fig_ep_angle_dependence.pdf"):
    """Plot coupling rates and nonreciprocity vs field angle."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    theta = result['theta']

    # (a) Coupling rates vs angle
    axes[0].plot(theta, result['g_mel'] / (2*np.pi*1e6), '-', color='C0',
                 lw=1.5, label=r'$g_{mel}$')
    axes[0].plot(theta, result['g_mr'] / (2*np.pi*1e6), '-', color='C1',
                 lw=1.5, label=r'$g_{mr}$')
    axes[0].axhline(result['ep_threshold'] / (2*np.pi*1e6), ls='--',
                     color='gray', lw=1, label='EP threshold')
    axes[0].set_xlabel(r'Field angle $\theta$ (°)')
    axes[0].set_ylabel(r'Coupling rate $g/2\pi$ (MHz)')
    axes[0].legend(fontsize=7)
    axes[0].set_title('(a) Coupling decomposition', fontsize=9)

    # (b) |g(+k)| and |g(-k)| vs angle
    axes[1].plot(theta, result['g_plus'] / (2*np.pi*1e6), '-', color='C0',
                 lw=1.5, label=r'$|g(+k)|$')
    axes[1].plot(theta, result['g_minus'] / (2*np.pi*1e6), '-', color='C1',
                 lw=1.5, label=r'$|g(-k)|$')
    axes[1].axhline(result['ep_threshold'] / (2*np.pi*1e6), ls='--',
                     color='gray', lw=1, label='EP threshold')
    axes[1].set_xlabel(r'Field angle $\theta$ (°)')
    axes[1].set_ylabel(r'$|g(\pm k)|/2\pi$ (MHz)')
    axes[1].legend(fontsize=7)
    axes[1].set_title('(b) Direction-dependent coupling', fontsize=9)

    # (c) Nonreciprocity ratio
    axes[2].plot(theta, result['nonrecip'] * 100, '-', color='C2', lw=1.5)
    axes[2].set_xlabel(r'Field angle $\theta$ (°)')
    axes[2].set_ylabel('Nonreciprocity (%)')
    axes[2].set_title('(c) Nonreciprocity', fontsize=9)
    axes[2].axhline(0, ls='-', color='gray', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_riemann_surface(result, filename="fig_ep_riemann.pdf"):
    """Plot Riemann surface structure around EP."""
    fig = plt.figure(figsize=(7, 3))

    B0_mT = result['B0'] * 1e3

    # Frequency splitting
    split_pk = np.abs(result['omega_plus_pk'] - result['omega_minus_pk']) / (2*np.pi*1e6)
    split_mk = np.abs(result['omega_plus_mk'] - result['omega_minus_mk']) / (2*np.pi*1e6)

    ax1 = fig.add_subplot(121)
    ax1.plot(B0_mT, split_pk, '-', color='C0', lw=1.5, label=r'$+k$')
    ax1.plot(B0_mT, split_mk, '--', color='C1', lw=1.5, label=r'$-k$')
    ax1.set_xlabel(r'$B_0$ (mT)')
    ax1.set_ylabel(r'$|\omega_+ - \omega_-|/2\pi$ (MHz)')
    ax1.set_title('(a) Mode splitting', fontsize=9)
    ax1.legend(fontsize=7)

    # Minimum splitting (proxy for EP proximity)
    min_split_pk = np.min(split_pk)
    min_split_mk = np.min(split_mk)
    print(f"  Min splitting +k: {min_split_pk:.3f} MHz")
    print(f"  Min splitting -k: {min_split_mk:.3f} MHz")

    # Complex plane trajectory
    ax2 = fig.add_subplot(122)
    re_pk = (result['omega_plus_pk'].real - result['omega_minus_pk'].real) / (2*np.pi*1e6)
    im_pk = (result['omega_plus_pk'].imag - result['omega_minus_pk'].imag) / (2*np.pi*1e6)
    re_mk = (result['omega_plus_mk'].real - result['omega_minus_mk'].real) / (2*np.pi*1e6)
    im_mk = (result['omega_plus_mk'].imag - result['omega_minus_mk'].imag) / (2*np.pi*1e6)

    ax2.plot(re_pk, im_pk, '-', color='C0', lw=1.5, label=r'$+k$')
    ax2.plot(re_mk, im_mk, '--', color='C1', lw=1.5, label=r'$-k$')
    ax2.plot(0, 0, 'k+', ms=10, mew=2)  # EP location
    ax2.set_xlabel(r'Re($\omega_+ - \omega_-$)$/2\pi$ (MHz)')
    ax2.set_ylabel(r'Im($\omega_+ - \omega_-$)$/2\pi$ (MHz)')
    ax2.set_title('(b) Complex plane', fontsize=9)
    ax2.legend(fontsize=7)
    ax2.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 70)
    print("Sim 09: Analytic Exceptional Point Calculation")
    print("         Chiral SAW-Magnon Hybrid System (YIG)")
    print("=" * 70)

    # --- Case 1: Strong coupling (YIG, low damping) ---
    print("\n--- Case 1: YIG strong coupling ---")
    Kmr = 1.0e6    # 1 MJ/m^3
    alpha = 2e-4   # YIG thin film
    Q_saw = 5000   # SAW cavity Q

    result1 = calc_eigenvalue_trajectories(Kmr, alpha, Q_saw)
    plot_eigenvalue_trajectories(result1, "fig_ep_eigenvalues_strong.pdf")
    plot_riemann_surface(result1, "fig_ep_riemann_strong.pdf")

    # --- Case 2: Near EP (tuned damping) ---
    print("\n--- Case 2: Near EP (tuned damping) ---")
    # EP condition: g = |kappa_m - kappa_p|/2
    # Find alpha_EP such that g_mr = |alpha*omega_p - omega_p/(2Q)|/2
    _, g_mr = coupling_rates(0, Kmr)
    # alpha_EP * omega_p - omega_p/(2Q) = ±2*g_mr
    # alpha_EP = 1/(2Q) ± 2*g_mr/omega_p
    alpha_EP_plus = 1/(2*Q_saw) + 2*g_mr/OMEGA_SAW
    alpha_EP_minus = 1/(2*Q_saw) - 2*g_mr/OMEGA_SAW
    print(f"  g_mr/(2pi) = {g_mr/(2*np.pi)*1e-6:.3f} MHz")
    print(f"  EP damping values: alpha_EP = {alpha_EP_plus:.2e} or {alpha_EP_minus:.2e}")

    # Use alpha slightly above EP
    alpha_ep = max(alpha_EP_plus, alpha_EP_minus) * 1.0
    if alpha_ep > 0 and alpha_ep < 1:
        result2 = calc_eigenvalue_trajectories(Kmr, alpha_ep, Q_saw)
        plot_eigenvalue_trajectories(result2, "fig_ep_eigenvalues_ep.pdf")
        plot_riemann_surface(result2, "fig_ep_riemann_ep.pdf")

    # --- Case 3: Multiple Kmr values comparison ---
    print("\n--- Case 3: Kmr comparison ---")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    Kmr_values = [0.3e6, 1.0e6, 5.0e6]
    alpha = 2e-4

    for col, Kmr in enumerate(Kmr_values):
        result = calc_eigenvalue_trajectories(Kmr, alpha, Q_saw)
        B0_mT = result['B0'] * 1e3

        # Real part
        axes[0, col].plot(B0_mT, result['omega_plus_pk'].real/(2*np.pi*1e9),
                          '-', color='C0', lw=1.2, label=r'$+k$, $\omega_+$')
        axes[0, col].plot(B0_mT, result['omega_minus_pk'].real/(2*np.pi*1e9),
                          '-', color='C1', lw=1.2, label=r'$+k$, $\omega_-$')
        axes[0, col].plot(B0_mT, result['omega_plus_mk'].real/(2*np.pi*1e9),
                          '--', color='C0', lw=1.2, label=r'$-k$, $\omega_+$')
        axes[0, col].plot(B0_mT, result['omega_minus_mk'].real/(2*np.pi*1e9),
                          '--', color='C1', lw=1.2, label=r'$-k$, $\omega_-$')

        f_m = np.array([kittel_frequency(B)/(2*np.pi*1e9) for B in result['B0']])
        axes[0, col].plot(B0_mT, f_m, ':', color='gray', lw=0.8, alpha=0.5)
        axes[0, col].axhline(F_SAW*1e-9, ls=':', color='gray', lw=0.8, alpha=0.5)
        axes[0, col].set_title(rf'$K_{{mr}}$={Kmr*1e-6:.1f} MJ/m³', fontsize=9)

        if col == 0:
            axes[0, col].set_ylabel(r'Re($\omega/2\pi$) (GHz)')
            axes[0, col].legend(fontsize=5.5, ncol=2)

        # Imaginary part
        axes[1, col].plot(B0_mT, -result['omega_plus_pk'].imag/(2*np.pi*1e6),
                          '-', color='C0', lw=1.2)
        axes[1, col].plot(B0_mT, -result['omega_minus_pk'].imag/(2*np.pi*1e6),
                          '-', color='C1', lw=1.2)
        axes[1, col].plot(B0_mT, -result['omega_plus_mk'].imag/(2*np.pi*1e6),
                          '--', color='C0', lw=1.2)
        axes[1, col].plot(B0_mT, -result['omega_minus_mk'].imag/(2*np.pi*1e6),
                          '--', color='C1', lw=1.2)
        axes[1, col].set_xlabel(r'$B_0$ (mT)')
        if col == 0:
            axes[1, col].set_ylabel(r'$-$Im($\omega/2\pi$) (MHz)')

    fig.suptitle(rf'YIG: $\alpha$={alpha:.0e}, Q={Q_saw}, $f_{{SAW}}$={F_SAW*1e-9:.0f} GHz',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ep_kmr_comparison.pdf"), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "fig_ep_kmr_comparison.png"), dpi=200)
    plt.close(fig)
    print("  Saved: fig_ep_kmr_comparison.pdf")

    # --- Phase diagram ---
    print("\n--- EP Phase diagram ---")
    pd_result = calc_ep_phase_diagram(Q_saw=Q_saw)
    plot_ep_phase_diagram(pd_result)

    # --- Angle dependence ---
    print("\n--- Angle dependence ---")
    angle_result = calc_angle_dependent_ep(Kmr=1.0e6, alpha=2e-4, Q_saw=Q_saw)
    plot_angle_dependence(angle_result)

    # --- Save all data ---
    np.savez(os.path.join(DATA_DIR, "sim09_ep_data.npz"),
             B0_res=result1['B0_res'],
             g_plus=result1['g_plus'],
             g_minus=result1['g_minus'],
             kappa_m=result1['kappa_m_res'],
             kappa_p=result1['kappa_p'],
             alpha_EP_plus=alpha_EP_plus,
             alpha_EP_minus=alpha_EP_minus,
             )
    print(f"\n  Data saved to {DATA_DIR}/sim09_ep_data.npz")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Material: YIG (Ms={MS*1e-3:.0f} kA/m, alpha={2e-4:.0e})")
    print(f"  SAW: f={F_SAW*1e-9:.1f} GHz, eps0={EPS0:.0e}, Q={Q_saw}")
    print(f"  Resonance field: B0_res = {result1['B0_res']*1e3:.2f} mT")
    print(f"  Coupling: |g|/(2pi) = {np.abs(result1['g_plus'])/(2*np.pi)*1e-6:.2f} MHz")
    print(f"  Damping: kappa_m/(2pi) = {result1['kappa_m_res']/(2*np.pi)*1e-6:.3f} MHz")
    print(f"  Phonon loss: kappa_p/(2pi) = {result1['kappa_p']/(2*np.pi)*1e-6:.3f} MHz")
    C = np.abs(result1['g_plus'])**2 / (result1['kappa_m_res'] * result1['kappa_p'])
    print(f"  Cooperativity: C = {C:.1f}")
    if C > 1:
        print("  -> STRONG COUPLING REGIME (C >> 1)")
    print(f"  EP damping: alpha_EP = {alpha_EP_plus:.2e}")


if __name__ == "__main__":
    main()
