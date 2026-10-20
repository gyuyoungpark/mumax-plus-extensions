# Physics contract and validation status

This directory contains distinct implementations and evidence. A passing CPU
control does not validate a CUDA implementation, and a source audit does not
replace an independent trajectory comparison.

## Source reviewed on October 21, 2026

| Component | Physical scope | Evidence and remaining work |
|---|---|---|
| Native `src/physics/cavityafm.*` | Uniform real x-directed Zeeman field; equal, static, spatially uniform sublattice Ms; one cavity and an optional reciprocal auxiliary oscillator; native local Gilbert damping | Source/energy audit and host C++ input-contract checks. The updated integration and bindings still require a fresh linked DOUBLE build and independent GPU regression. |
| [`methods/src/cavity_method`](methods/src/cavity_method) | Explicit energy coordinates, physical source moments, reciprocal coupling and a uniform unforced callback adapter | The original frozen implementation passed finite controls and one lossless FM pilot against two independent reference trajectories. This public port includes disclosed software guards; its physical validation is not independently renewed. |
| Existing C1/C3 analytic workflows | Fixed-pole, oscillator-strength and response-shape comparisons in the declared symmetric model | Analytic synthetic and generator consistency controls, as described in the [guide](README.md). They are not measurements or fresh GPU verification. |
| Existing C2/C4 workflow | Independent free-spin calibration; deterministic numerical variation envelopes | Input admission has been tightened. New physical records and fits were not executed for this source update. Finite envelopes are not confidence intervals or global uncertainty bounds. |
| New signed-spectrum analysis | Declared complex raw trace, full signed frequency axis, physical peak separation and explicit branch coverage | Source/software controls only for the port. Historical FFT figures have not been reprocessed or repaired by adding this tool. |

The native input-contract test runs without CUDA:

```sh
c++ -std=c++17 -Isrc/physics test/cavity_config_host.cpp -o cavity_config_host
./cavity_config_host
```

This tests refusal of invalid inputs, not physical accuracy. The existing
`test/test_cavity_afm.py` requires the actual rebuilt DOUBLE engine and GPU;
it was not run for this update. Changing activation flags now registers the
oscillator equations automatically. Configure valid frequencies, losses,
normalization and sublattice Ms before enabling the primary cavity. Auxiliary
settings can be staged while the primary is disabled. Change configuration
only while the solver is idle.

## Energy, units and normalization

All magnetic fields are B in tesla. Frequencies, losses and couplings are
angular rates in rad/s. The gyromagnetic ratio convention is positive with
`dm/dt = -gamma m cross B + Gilbert torque`. A cavity stores real dimensionless
coordinates `(q,p,0)`, not a photon annihilation amplitude.

For one source with moment `mu=Ms*occupied_volume` in J/T and a uniform mode,

```text
E = E_spin + E0*(q*q+p*p) - mu*h*q*<mx>
dq/dt = omega*p
dp/dt = -omega*q - 2*kappa*p + omega*mu*h*<mx>/(2*E0)
B_cavity,x = h*q
```

Each AFM sublattice is a separate physical moment. The field has the same
physical sign on both sublattices. Opposite equilibrium torque signs follow
from antiparallel moments. They do not require a staggered applied field.
The native equal-moment implementation chooses
`E0=Ms*occupied_volume*cavity_energy_field` for one sublattice. Its source is
then `omega*h*(<m1x>+<m2x>)/(2*cavity_energy_field)`. The normalization field
is positive and is chosen equal to the exchange field in the examples; that
choice is not an additional law of reciprocity.

The same moment, occupied geometry and profile convention must enter energy,
field and source. Unequal, spatially varying or time-dependent sublattice Ms
is rejected by this native model. A general spatial implementation needs an
explicit cell-moment-weighted source; a sum of unweighted averages is not a
replacement. The standalone kernel design in the development work is not an
integrated or validated general GPU API.

The FM conversion `h=2*G/gamma` is conditional on a circular small-angle FM,
a rotating-wave approximation and `E0=mu*omega/(2*gamma)`. More generally the
reviewed FM mapping is `G=h*sqrt(mu*gamma*omega/(8*E0))`. Do not apply this FM
conversion to an AFM without the mode-overlap derivation.

## Damping, stability and energy reporting

The cavity bath is viscous: it acts as `-2*kappa*p`. It differs from isotropic
amplitude damping outside the weak-loss rotating-wave limit. Nonnegative
cavity losses and a passive Gilbert matrix make their dissipation contributions
nonpositive for the corresponding energy model. They do not guarantee a
positive transverse Hessian or a stable chosen equilibrium at arbitrary
coupling. The input checks do not certify coupled stability.

The native optional auxiliary has reciprocal position-position coupling, not
an independently tunable dissipative off-diagonal term. No legacy phase-only
projection or one-way `g_diss` channel is supplied as a passive model. A common
bath or active reservoir would require its own derivation and validation.

The library's spin `total_energy` omits the cavity oscillator energy and its
spin-cavity interaction. Construct the declared total coupled energy when
checking energy balance; avoid double counting when another adapter inserts
the feedback as a magnetic bias whose Zeeman energy is already reported.
Spin-only `minimize()` and `relax()` are refused while a native cavity is
enabled, including world and sublattice entry points. Disable cavities for
spin-only preparation. A general coupled equilibrium solver is not supplied.

## Open verification

- Fresh native build, binding activation/rollback checks and independent GPU
  trajectories for the updated source.
- General spatial profiles, unequal moments, multimode banks and native/Python
  equivalence; the new FM pilot does not establish these.
- Dissipative/common-bath coupling, driven restart and stochastic restart.
- Full-chain reproduction on a clean machine and public raw-data archive.
- Device feasibility, continuous statistical coverage and microscopic damping
  mechanism identification, which cannot be certified by source tests.

Prior failed and unexecuted work remains part of the research record. The
submission manuscripts and original records were not changed by this update.
New scientific runs require an explicit protocol, complete restart state,
source/runtime identities and prospective acceptance criteria.
