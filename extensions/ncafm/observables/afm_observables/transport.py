"""Explicit Hall packing h=(sigma_yz,sigma_zx,sigma_xy), sigmaA_ij=eps_ijk h_k."""
import numpy as np


def conductivity(longitudinal, hall):
    s = np.asarray(longitudinal, dtype=float)
    if s.ndim == 0:
        s = np.eye(3)*s
    h = np.asarray(hall, dtype=float)
    if s.shape != (3, 3) or h.shape != (3,) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(h)):
        raise ValueError("Finite symmetric 3x3 longitudinal tensor and Hall vector required")
    if not np.allclose(s, s.T, rtol=1e-12, atol=0) or np.min(np.linalg.eigvalsh(s)) <= 0:
        raise ValueError("Longitudinal conductivity must be symmetric positive definite")
    x, y, z = h
    return s + np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]])


def resistivity(longitudinal, hall):
    """Exact inversion, including Hall-angle and anisotropic longitudinal terms."""
    return np.linalg.inv(conductivity(longitudinal, hall))


def hall_resistivity_isotropic(sigma_xx, hall):
    """Antisymmetric resistivity vector, in the SAME (yz,zx,xy) packing."""
    h = np.asarray(hall, dtype=float)
    s = float(sigma_xx)
    if h.shape != (3,) or not np.all(np.isfinite(h)) or not np.isfinite(s) or s <= 0:
        raise ValueError("Positive finite sigma_xx and finite Hall vector required")
    scale = max(s, float(np.max(np.abs(h))))
    return -(h/scale)/(scale*((s/scale)**2+np.dot(h/scale, h/scale)))


def assigned_vector_response(proxy, coefficient):
    """Optional phenomenological vector response, NOT a rank-3 spin Hall tensor.

    This coefficient is independent of any anomalous Hall coefficient.
    A physical spin-current response requires its own tensor and calibration.
    """
    if not np.isfinite(coefficient):
        raise ValueError("Finite independent response coefficient required")
    return np.asarray(proxy, dtype=float)*coefficient
