"""Explicit graph-face energy and its EXACT Cartesian derivatives.

For oriented face (a,b), delta_s=m[b,s]-m[a,s]:
E_face = 1/2 sum_st G_st delta_s.delta_t
         + sum_st D_st . (m[a,s] x m[b,t]).

G and D are face ENERGY coefficients in joules, including geometric factors.
No continuum coefficient averaging or ghost-spin rule is inferred. A free
boundary is a missing face, so both energy and field use the same graph.
"""
import numpy as np


def energy_field(spins, moments, edges, stiffness, dmi):
    """Return energy (J), dE/dm (J), and effective B=-dE/dm/moment (T).

    Arrays: spins[cells,slots,3], moment magnitudes[cells,slots] (A m^2),
    edges[faces,2], G[faces,slots,slots], D[faces,slots,slots,3].
    A zero moment is inactive. A cross-gradient face term requires all four
    endpoints; a directed DMI term requires its two endpoints. These masking
    rules are part of the specified discrete model, not a universal interface
    prescription. Onsite energies and their derivatives must be added separately.
    """
    m=np.asarray(spins,dtype=float); mu=np.asarray(moments,dtype=float)
    e=np.asarray(edges); g=np.asarray(stiffness,dtype=float); d=np.asarray(dmi,dtype=float)
    if m.ndim!=3 or m.shape[-1]!=3 or mu.shape!=m.shape[:2]:
        raise ValueError('Expected spins (cells,slots,3) and matching moment magnitudes')
    if e.ndim!=2 or e.shape[1]!=2 or not np.issubdtype(e.dtype,np.integer) or np.any(e<0) or np.any(e>=len(m)):
        raise ValueError('Face endpoints must be integer cell indices in range')
    slots=m.shape[1]
    if g.shape!=(len(e),slots,slots) or d.shape!=g.shape+(3,):
        raise ValueError('Face coefficient shapes do not match graph and sublattices')
    if not all(np.all(np.isfinite(x)) for x in (m,mu,g,d)) or np.any(mu<0):
        raise ValueError('Finite coefficients and nonnegative moments required')
    if not np.allclose(g,g.transpose(0,2,1),rtol=1e-12,atol=0):
        raise ValueError('Cross-gradient face stiffness must be reciprocal')
    energy=0.; gradient=np.zeros_like(m)
    for face,(a,b) in enumerate(e):
        active=(mu[a]>0)&(mu[b]>0)
        gg=g[face]*active[:,None]*active[None,:]
        dd=d[face]*(mu[a]>0)[:,None,None]*(mu[b]>0)[None,:,None]
        delta=m[b]-m[a]
        derivative=gg@delta
        energy+=.5*np.sum(delta*derivative)
        gradient[a]-=derivative; gradient[b]+=derivative
        energy+=np.sum(dd*np.cross(m[a,:,None,:],m[b,None,:,:]))
        gradient[a]+=np.sum(np.cross(m[b,None,:,:],dd),axis=1)
        gradient[b]+=np.sum(np.cross(dd,m[a,:,None,:]),axis=0)
    field=np.zeros_like(m)
    np.divide(-gradient,mu[:,:,None],out=field,where=mu[:,:,None]>0)
    return float(energy),gradient,field
