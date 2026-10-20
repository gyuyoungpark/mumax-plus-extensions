"""Declared classical spin/cavity energy; no engine or optional-library imports.

Coordinates ``q,p`` are real, dimensionless energy coordinates, not photon
annihilation amplitudes. Fields are B in tesla; angular frequencies are rad/s.
The model contains no legacy phase-only or symmetric dissipative coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def finite_real(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


class UnsupportedCoupling(ValueError):
    """A requested channel is outside the declared energy model."""


def reject_legacy_coupling(*, g_diss: float = 0.0, phase_rad: float = 0.0) -> None:
    """Reject old channels instead of silently dropping their parameters."""
    if finite_real(g_diss, "g_diss") != 0.0:
        raise UnsupportedCoupling(
            "Legacy g_diss is unsupported: a physical bath and both reciprocal "
            "directions must be specified separately. The supported bath is "
            "viscous -2*kappa*p only."
        )
    if finite_real(phase_rad, "phase_rad") != 0.0:
        raise UnsupportedCoupling("Legacy coupling phase is not an energy-derived channel")


@dataclass(frozen=True)
class MagneticSource:
    """One FM or AFM sublattice; moment=Ms*occupied volume in J/T.

    This scalar-moment formulation requires homogeneous Ms in that source.
    A heterogeneous source needs a moment-weighted reduction, not a plain mean.
    Each AFM sublattice is a separate source: no factor 1/2 is inserted.
    """

    name: str
    moment_j_per_t: float

    def __post_init__(self) -> None:
        _name(self.name, "source name")
        mu = finite_real(self.moment_j_per_t, "moment_j_per_t")
        if mu <= 0:
            raise ValueError("moment_j_per_t must be positive")
        object.__setattr__(self, "moment_j_per_t", mu)


@dataclass(frozen=True)
class EnergyMode:
    """An oscillator with E0*(q*q+p*p) and Bx=h*q*u.

    ``profile_id`` identifies the exact dimensionless profile/convention used
    in BOTH the overlap and field. Only ``uniform`` is supported by the adapter.
    Nonuniform profiles must be identified by their recorded content hash and
    normalization convention by the future spatial implementation.
    """

    name: str
    omega_rad_s: float
    kappa_rad_s: float
    energy_scale_j: float
    field_t_per_q: float
    profile_id: str = "uniform"

    def __post_init__(self) -> None:
        _name(self.name, "mode name")
        _name(self.profile_id, "profile_id")
        for name in ("omega_rad_s", "kappa_rad_s", "energy_scale_j", "field_t_per_q"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if self.omega_rad_s <= 0 or self.energy_scale_j <= 0:
            raise ValueError("omega_rad_s and energy_scale_j must be positive")
        if self.kappa_rad_s < 0:
            raise ValueError("kappa_rad_s must be nonnegative")


@dataclass(frozen=True)
class Quadratures:
    q: float
    p: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", finite_real(self.q, "q"))
        object.__setattr__(self, "p", finite_real(self.p, "p"))


@dataclass(frozen=True)
class QuadratureRate:
    """Coordinate derivatives or additive drive, both in 1/s."""

    dq_dt: float
    dp_dt: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "dq_dt", finite_real(self.dq_dt, "dq_dt"))
        object.__setattr__(self, "dp_dt", finite_real(self.dp_dt, "dp_dt"))


@dataclass(frozen=True)
class OverlapSnapshot:
    """values[mode][source]=<u_j*m_ix> over occupied magnetic volume.

    All source/mode pairs and matching profile identities are required.
    This object accepts measured overlaps; it does not invent spatial dynamics.
    """

    values: Mapping[str, Mapping[str, float]]
    profile_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        values = {
            mode: MappingProxyType({source: finite_real(x, "overlap")
                                    for source, x in row.items()})
            for mode, row in self.values.items()
        }
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "profile_ids", MappingProxyType(dict(self.profile_ids)))


@dataclass(frozen=True)
class EnergyCavityModel:
    """Both interaction directions derived from the same specified energy.

    E = E_spin + sum_j E0_j(q_j^2+p_j^2)
        - sum_ij mu_i*h_j*q_j*<u_j*m_ix>.
    dq_j=omega_j*p_j,
    dp_j=-omega_j*q_j-2*kappa_j*p_j
         +omega_j/(2*E0_j)*sum_i mu_i*h_j*<u_j*m_ix>.

    Positive oscillator scales and losses do not alone prove positivity of the
    TOTAL coupled energy: that also depends on the declared spin energy/state.
    Spin Gilbert damping belongs to the magnetic engine, not a second bath here.
    """

    modes: tuple[EnergyMode, ...]
    sources: tuple[MagneticSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.modes or not self.sources:
            raise ValueError("At least one mode and source are required")
        for items, kind in ((self.modes, "mode"), (self.sources, "source")):
            names = [x.name for x in items]
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {kind} names")

    def validate_state(self, state: Mapping[str, Quadratures]) -> None:
        if set(state) != {m.name for m in self.modes}:
            raise ValueError("State must contain exactly the declared modes")
        if not all(isinstance(x, Quadratures) for x in state.values()):
            raise TypeError("Each state value must be Quadratures(q,p)")

    def validate_overlaps(self, overlaps: OverlapSnapshot) -> None:
        if set(overlaps.values) != {m.name for m in self.modes}:
            raise ValueError("Overlap modes do not match the model")
        expected_profiles = {m.name: m.profile_id for m in self.modes}
        if dict(overlaps.profile_ids) != expected_profiles:
            raise ValueError("Overlap and feedback profile identities differ")
        for row in overlaps.values.values():
            if set(row) != {s.name for s in self.sources}:
                raise ValueError("Every overlap needs exactly the declared sources")

    def rhs(self, state: Mapping[str, Quadratures], overlaps: OverlapSnapshot,
            *, drive: Mapping[str, QuadratureRate] | None = None
            ) -> dict[str, QuadratureRate]:
        self.validate_state(state)
        self.validate_overlaps(overlaps)
        drive = {} if drive is None else drive
        if not set(drive).issubset(state):
            raise ValueError("Drive refers to an undeclared mode")
        result = {}
        for mode in self.modes:
            x = state[mode.name]
            source = sum(s.moment_j_per_t * mode.field_t_per_q
                         * overlaps.values[mode.name][s.name] for s in self.sources)
            forcing = drive.get(mode.name, QuadratureRate(0.0, 0.0))
            if not isinstance(forcing, QuadratureRate):
                raise TypeError("Drive values must be QuadratureRate")
            result[mode.name] = QuadratureRate(
                mode.omega_rad_s*x.p + forcing.dq_dt,
                -mode.omega_rad_s*x.q - 2*mode.kappa_rad_s*x.p
                + mode.omega_rad_s*source/(2*mode.energy_scale_j) + forcing.dp_dt,
            )
        return result

    def oscillator_energy_j(self, state: Mapping[str, Quadratures]) -> float:
        self.validate_state(state)
        return sum(m.energy_scale_j*(state[m.name].q**2+state[m.name].p**2)
                   for m in self.modes)

    def interaction_energy_j(self, state: Mapping[str, Quadratures],
                             overlaps: OverlapSnapshot) -> float:
        self.validate_state(state)
        self.validate_overlaps(overlaps)
        return -sum(s.moment_j_per_t*m.field_t_per_q*state[m.name].q
                    * overlaps.values[m.name][s.name]
                    for m in self.modes for s in self.sources)

    def total_energy_j(self, state: Mapping[str, Quadratures], overlaps: OverlapSnapshot,
                       *, spin_energy_j: float) -> float:
        """E_spin must EXCLUDE the cavity Zeeman cross term already added here.

        A magnetic engine that receives feedback through its bias-field handle
        may include that term in its reported Zeeman energy. That reported total
        cannot be passed unchanged without double counting the interaction.
        """
        return (finite_real(spin_energy_j, "spin_energy_j")
                + self.oscillator_energy_j(state)
                + self.interaction_energy_j(state, overlaps))

    def viscous_bath_power_w(self, state: Mapping[str, Quadratures]) -> float:
        """Cavity bath contribution only: -4*sum(E0*kappa*p^2)."""
        self.validate_state(state)
        return -4*sum(m.energy_scale_j*m.kappa_rad_s*state[m.name].p**2
                      for m in self.modes)

    def feedback_x_t(self, state: Mapping[str, Quadratures],
                     local_profiles: Mapping[str, float]) -> float:
        """Same Bx on every coupled source at a given physical location.

        Spatial callers must evaluate the EXACT recorded profile, with the same
        convention used for OverlapSnapshot; no spatial engine is supplied here.
        """
        self.validate_state(state)
        if set(local_profiles) != set(state):
            raise ValueError("Local profiles must contain exactly the modes")
        return sum(m.field_t_per_q*state[m.name].q
                   * finite_real(local_profiles[m.name], "local profile")
                   for m in self.modes)

    def uniform_feedback_x_t(self, state: Mapping[str, Quadratures]) -> float:
        if any(m.profile_id != "uniform" for m in self.modes):
            raise UnsupportedCoupling("Uniform feedback cannot represent a spatial profile")
        return self.feedback_x_t(state, {m.name: 1.0 for m in self.modes})

    def manifest(self) -> dict:
        """JSON-compatible model identity; this is not a run/binary manifest."""
        return {
            "model": "classical_energy_qp_v1", "field_unit": "T",
            "energy_unit": "J", "coordinates": "q,p (not photon amplitudes)",
            "bath": "dp/dt += -2*kappa*p",
            "modes": [dict(name=m.name, omega_rad_s=m.omega_rad_s,
                           kappa_rad_s=m.kappa_rad_s, energy_scale_j=m.energy_scale_j,
                           field_t_per_q=m.field_t_per_q, profile_id=m.profile_id)
                      for m in self.modes],
            "sources": [dict(name=s.name, moment_j_per_t=s.moment_j_per_t)
                        for s in self.sources],
        }
