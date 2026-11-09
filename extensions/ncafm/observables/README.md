# Site-based rank-three magnetic multipoles and Hall tensor conventions

This additive module computes **position-dependent cluster magnetic multipoles**.
It does not replace `NcAfm.octupole_vector`, which is a fixed-sublattice
registered proxy in this repository. A three-vector proxy is not a general
rank-three magnetic multipole or a calibrated electronic response.

`afm_observables.cluster_octupole(positions, moments, weights)` takes explicit
arrays `(clusters, sites, 3)` and one counting weight per cluster. Positions
are relative to the selected cluster origin in meters. Cartesian magnetic
moments are in A m^2. The seven complex outputs, ordered q=-3,...,3, are in
A m^4 and include the `sqrt(4 pi/7)` prefactor and complex conjugation of
Suzuki et al., Eq. (6). `gamma_pair` returns the real `(Tx^gamma,Ty^gamma)`
pair. Neither an unspecified third vector component nor a Hall coefficient
is inferred. Clusters, periodic images, origins, multiplicities and crystal
axes must be selected from the actual structural model. Independent moments
at every site are supported. Positions and axial magnetic moments must be
transformed consistently when changing the Cartesian frame.

The registered proxy remains useful on a declared rigid planar submanifold.
Outside that submanifold, the site-based operator must be evaluated directly.
A fixed reference normalization need not bound its norm by one.

`field_adapter.site_coefficients` explicitly registers physical sites to solver
slots and includes the per-site moment magnitudes and cluster weights.
`evaluate_fields` accepts NumPy arrays or mumax+ field quantities exposing
`.eval()`, without mutating them. This is CPU postprocessing of solver fields.
Repeated slots are a declared layer-pairing constraint; use six independent
slots when all six moments must evolve independently. The native contraction
layout is provided by `cuda_real_coefficients` (seven real parts, then seven
imaginary parts). No untested native engine binding is advertised.

## Electrical conventions

`h=(sigma_yz,sigma_zx,sigma_xy)` and `sigmaA_ij=epsilon_ijk h_k`.
For isotropic positive longitudinal conductivity `s`, the exact antisymmetric
resistivity vector in the **same index order** is `-h/(s*s+dot(h,h))`.
An anisotropic positive definite longitudinal tensor is inverted as a full
matrix by `resistivity`. Zero, negative and nonfinite longitudinal inputs
are rejected. `assigned_vector_response` has an independent coefficient;
it is an assigned vector law, not the rank-three spin-current conductivity.

## Gradient model

`require_coercive` checks the symmetric gradient stiffness matrices for a
continuum model with energy `1/2 sum G[a,s,t] d_a m_s . d_a m_t`, in J/m.
Atomic exchange-bond signs and A/C/G atomic ordering are different inputs.
An indefinite second-order gradient expansion requires an explicit cutoff
or additional stabilizing terms. A failed sufficient continuum coercivity
test alone does not prove instability of a finite-cutoff lattice model.

## Checks and native scope

Requires Python and NumPy. From this folder:

```
python tests/verify.py --out /path/outside/source/continuous
python tests/verify.py --out /path/outside/source/resumed --stop-after 1
python tests/verify.py --out /path/outside/source/resumed
```

Compare the two `summary.json` files for exact restart agreement. Checkpoints
record the progress index, generator state and code/runtime identities.
Use distinct output directories for simultaneous runs. The optional independent
Windows C++ oracle is in `native/observables.cpp`. `native/site_multipole.cu`
is a device-array contraction kernel; its coefficients must include geometry,
counting weights and the declared output basis. Compiling it is not evidence
of GPU execution or integration into the complete mumax+ engine. This module
does not install or modify the user's active mumax+ binary.

`patches/` contains **version-specific legacy compatibility patches**, not
patches to the current public engine. Inspect their source identities in
`patches/BASES.json` before use; never apply them blindly to another version.
The constructor-default patch concerns the isolated variational implementation,
which still needs a rebuilt native-module integration test.

## Primary references

- Suzuki et al., Phys. Rev. B 95, 094406 (2017),
  https://doi.org/10.1103/PhysRevB.95.094406
- Liu and Balents, Phys. Rev. Lett. 119, 087202 (2017),
  https://doi.org/10.1103/PhysRevLett.119.087202
- Yoo, Winkler and Hoffmann, Applied Physics Reviews 13, 021423 (2026),
  https://doi.org/10.1063/5.0302867
- Moreels et al., npj Computational Materials 12, 71 (2026),
  https://doi.org/10.1038/s41524-025-01893-y

These definitions and verification checks do not establish an independent
material calibration, an absolute Hall coefficient, or novelty over the cited
micromagnetic and multipole frameworks.
