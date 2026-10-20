"""Analytical FM/RWA coordinate map. No numerical values are evaluated on import.

This map is for one circular FM near +z with gamma>0, B0>0 and small angle.
It must NOT be applied to an AFM without its own mode/ellipticity derivation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .energy import EnergyMode, MagneticSource, finite_real


def moment_from_homogeneous_material(*, ms_a_per_m: float, occupied_volume_m3: float) -> float:
    """Ms*occupied magnetic volume, not bounding-box volume, in J/T."""
    ms = finite_real(ms_a_per_m, "ms_a_per_m")
    volume = finite_real(occupied_volume_m3, "occupied_volume_m3")
    if ms <= 0 or volume <= 0:
        raise ValueError("Ms and occupied magnetic volume must be positive")
    return ms*volume


@dataclass(frozen=True)
class FmRwaMapping:
    """Scales c=s_c*(q+i p), b=s_b*(mx-i my), each in sqrt(J*s).

    Free energy=omega_c*|c|^2+omega_m*|b|^2.
    RWA interaction=-g*(c*conj(b)+conj(c)*b).
    Thus cdot=-i*omega_c*c+i*g*b and bdot=-i*omega_m*b+i*g*c.
    This classical action normalization makes no quantum-state claim.
    """

    cavity_action_scale_sqrt_j_s: float
    spin_action_scale_sqrt_j_s: float
    coupling_rad_s: float
    raw_cavity_from_mminus_rad_s: float
    raw_mminus_from_cavity_rad_s: float


def fm_rwa_mapping(mode: EnergyMode, source: MagneticSource,
                   *, gamma_rad_per_s_t: float) -> FmRwaMapping:
    """Return the declared analytical coefficients when explicitly requested.

    For z=q+i p, raw zdot contains +i*omega_c*mu*h/(4E0)*mminus;
    raw mminus_dot contains +i*gamma*h/2*z. The corresponding g is
    h*sqrt(mu*gamma*omega_c/(8E0)). Viscous p damping becomes amplitude
    damping kappa only after an additional weak-loss RWA.
    """
    if mode.profile_id != "uniform":
        raise ValueError("FM RWA map here requires a uniform circular mode")
    gamma = finite_real(gamma_rad_per_s_t, "gamma_rad_per_s_t")
    if gamma <= 0:
        raise ValueError("gamma must be positive for the declared LLG convention")
    mu, h, wc, e0 = (source.moment_j_per_t, mode.field_t_per_q,
                      mode.omega_rad_s, mode.energy_scale_j)
    return FmRwaMapping(
        sqrt(e0/wc), sqrt(mu/(2*gamma)),
        h*sqrt(mu*gamma*wc/(8*e0)), wc*mu*h/(4*e0), gamma*h/2,
    )
