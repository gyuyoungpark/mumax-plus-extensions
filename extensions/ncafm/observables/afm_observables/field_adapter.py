"""Readout adapter for declared solver fields; never changes a solver state."""
import numpy as np
from .multipole import solid_gradients


def site_coefficients(positions, site_indices, moment_sizes, weights, nspins):
    """Build C[q,solver_slot,xyz] for unit-direction solver fields.

    Geometry is (clusters, sites, 3) in meters. Site indices explicitly register
    every physical site to a solver slot. Equal slots impose equal directions;
    use independent slots for independently evolving sites. Moment magnitudes
    in A m^2 and cluster counting weights are explicit, not inferred from Ms.
    """
    r=np.asarray(positions,dtype=float)
    ids=np.asarray(site_indices)
    mu=np.asarray(moment_sizes,dtype=float)
    w=np.asarray(weights,dtype=float)
    if r.ndim!=3 or r.shape[-1]!=3 or ids.shape!=r.shape[:2] or mu.shape!=ids.shape:
        raise ValueError('Cluster geometry, registration and moment shapes differ')
    if not isinstance(nspins,int) or nspins<=0 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=nspins):
        raise ValueError('Site-to-solver indices must be integers in range')
    if w.shape!=(r.shape[0],) or not np.all(np.isfinite(w)) or np.any(w<0) or not np.any(w>0):
        raise ValueError('Explicit finite nonnegative counting weights required')
    if not np.all(np.isfinite(mu)) or np.any(mu<0):
        raise ValueError('Moment magnitudes must be finite and nonnegative')
    gradients=solid_gradients(r)
    c=np.zeros((7,nspins,3),complex)
    for cluster in range(r.shape[0]):
        for site in range(r.shape[1]):
            c[:,ids[cluster,site],:]+=w[cluster]*mu[cluster,site]*gradients[cluster,site]
    return c


def evaluate_fields(coefficients, fields):
    """Fields are arrays (3,...) or mumax+ quantities exposing .eval().

    Evaluating GPU-backed quantities reads their current values to the host;
    this adapter performs the multipole contraction on CPU. It does not imply
    a native CUDA binding or an independently calibrated material model.
    """
    c=np.asarray(coefficients,dtype=complex)
    values=np.stack([x.eval() if hasattr(x,'eval') else x for x in fields])
    if c.shape!=(7,len(fields),3) or values.shape[:2]!=(len(fields),3):
        raise ValueError('Expected coefficients (7,slots,3) and fields (slots,3,...)')
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite coefficient or solver field')
    return np.einsum('qsc,sc...->q...',c,values)


def cuda_real_coefficients(coefficients):
    """14 real outputs: q=-3..3 real parts, followed by imaginary parts."""
    c=np.asarray(coefficients,dtype=complex)
    if c.ndim!=3 or c.shape[0]!=7 or c.shape[2]!=3 or not np.all(np.isfinite(c)):
        raise ValueError('Expected finite (7,slots,3) coefficients')
    return np.ascontiguousarray(np.concatenate([c.real,c.imag],axis=0),dtype=np.float64)
