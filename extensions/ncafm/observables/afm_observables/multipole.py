"""Suzuki et al. Eq. (6), rank 3, Condon--Shortley convention.

Positions are relative to an explicitly selected cluster origin in meters.
Moments are Cartesian magnetic moments in A m^2, not unit spin directions.
Output order is q=-3,...,3; units are A m^4. Cluster weights specify counting,
not a statistical average. Origins and cluster membership are physical inputs.
"""
import numpy as np


def _vectors(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim < 2 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite (..., sites, 3) vectors")
    return value


def solid_gradients(positions):
    """Gradient of sqrt(4 pi/7) r^3 Y_3q^*, including all seven components."""
    r = _vectors(positions, "positions")
    x, y, z = np.moveaxis(r, -1, 0)
    u = x - 1j * y
    v = 4 * z*z - x*x - y*y
    positive = np.stack([
        np.stack([-3*z*x, -3*z*y, 3*z*z-1.5*(x*x+y*y)], axis=-1),
        -np.sqrt(3)/4*np.stack([v-2*x*u, -1j*v-2*y*u, 8*z*u], axis=-1),
        np.sqrt(15/8)*np.stack([2*z*u, -2j*z*u, u*u], axis=-1),
        -np.sqrt(5)/4*np.stack([3*u*u, -3j*u*u, np.zeros_like(u)], axis=-1),
    ], axis=-2)
    return np.concatenate([
        np.stack([-positive[..., 3, :].conj(), positive[..., 2, :].conj(),
                  -positive[..., 1, :].conj()], axis=-2), positive], axis=-2)


def cluster_octupole(positions, moments, weights):
    """Sum explicit clusters, allowing independent moments at every site.

    Arrays have shape (clusters, sites, 3). Repeated physical sites must have
    consistent moments; the caller supplies their counting weights explicitly.
    No inversion pairing, rigid-spin condition or crystal symmetry is assumed.
    """
    r = _vectors(positions, "positions")
    m = _vectors(moments, "moments")
    w = np.asarray(weights, dtype=float)
    if r.ndim != 3 or m.shape != r.shape or w.shape != (r.shape[0],):
        raise ValueError("Expected (clusters, sites, 3) and one weight per cluster")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Cluster counting weights must be finite, nonnegative and nonzero")
    return np.einsum("c,csqj,csj->q", w, solid_gradients(r), m)


def gamma_pair(multipoles):
    """Real T_x^gamma,T_y^gamma; not a three-component Hall vector."""
    value = np.asarray(multipoles, dtype=complex)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected seven finite rank-3 spherical components")
    result = np.array([(-value[4]+value[2])/np.sqrt(2),
                       -1j*(value[4]+value[2])/np.sqrt(2)])
    if np.max(np.abs(result.imag)) > 1e-12 * max(np.max(np.abs(value)), 1e-300):
        raise ValueError("Multipoles do not have the reality symmetry of real moments")
    return result.real


def registered_proxy(spins, weights):
    """Continuous fixed-slot proxy. Its z component is NOT a rank-3 z moment."""
    spins = _vectors(spins, "spins")
    weights = np.asarray(weights, dtype=float)
    if spins.shape != (3, 3) or weights.shape != (3,):
        raise ValueError("Three ordered sublattice spins and weights are required")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    if weights.sum() == 0:
        return np.zeros(3)
    angles = -2*np.pi*np.arange(3)/3
    transformed = spins.copy()
    transformed[:, 0] = np.cos(angles)*spins[:, 0]-np.sin(angles)*spins[:, 1]
    transformed[:, 1] = np.sin(angles)*spins[:, 0]+np.cos(angles)*spins[:, 1]
    return np.einsum("i,ij->j", weights, transformed)/weights.sum()
