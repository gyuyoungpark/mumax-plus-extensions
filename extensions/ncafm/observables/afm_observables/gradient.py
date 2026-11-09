"""Coercivity gate for a continuum sublattice gradient model."""
import numpy as np


def require_coercive(stiffness):
    """G[a,s,t] multiplies 1/2 (d_a m_s).(d_a m_t), SI J/m.

    This is a sufficient continuum PSD requirement, not a test of stability of
    every finite-cutoff lattice Hamiltonian. Atomic bond signs do not label
    A/C/G magnetic order in this gradient parametrization.
    """
    g = np.asarray(stiffness, dtype=float)
    if g.ndim != 3 or g.shape[0] != 3 or g.shape[1] != g.shape[2] or not np.all(np.isfinite(g)):
        raise ValueError("Expected finite (3, sublattices, sublattices) stiffness")
    if not np.allclose(g, g.transpose(0, 2, 1), atol=0, rtol=1e-12):
        raise ValueError("Reciprocal stiffness must be symmetric in sublattice indices")
    eigenvalues = np.linalg.eigvalsh(g)
    if eigenvalues.min() < -1e-12*max(np.max(np.abs(g)), 1e-300):
        raise ValueError("Indefinite continuum gradient energy; specify a lattice cutoff or additional stabilizing terms")
    return eigenvalues
