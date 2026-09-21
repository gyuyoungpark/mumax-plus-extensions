# Energy-consistent AFM cavity examples

This source-only package implements the declared zero-bias, uniform-Zeeman,
native-Gilbert model with a viscous cavity bath. Both interaction directions
derive from the same energy. It tests how the magnetic response numerator
affects a cavity exceptional point after matching the complete free magnetic
pole set and accounting for oscillator strength.

The model has an exact two-coordinate realization with derivative coupling.
The response-shape effect requires finite spin damping. These examples do not
establish a damping-independent material coordinate, arbitrary-bath equivalence
or experimental distinguishability.

No original simulation records, fitted calibration, frozen predictions,
manuscript, reviewer documents or machine paths are distributed. Every example
generates its own output. The provenance manifest contains source hashes only.

## Installation

Python **3.12** is the tested interpreter. Clone this public fork, including its
build submodule, and run the following commands from the new repository root:

```sh
git clone --recursive https://github.com/gyuyoungpark/mumax-plus-extensions.git
cd mumax-plus-extensions
python -m pip install -r extensions/photon-magnon/requirements-cpu.txt
```

The CPU examples below need no native build. For GPU generation, first install
the [CUDA, compiler and build dependencies](../../README.md#dependencies), then
build this checkout in DOUBLE precision. On a POSIX shell:

```sh
export MUMAXPLUS_FP_PRECISION=DOUBLE
python -m pip install -v .
```

On Windows, use a PowerShell session with the CUDA and MSVC build environment
available, as described in the repository's build instructions:

```powershell
$env:MUMAXPLUS_FP_PRECISION = 'DOUBLE'
python -m pip install -v .
```

Keep the DOUBLE setting when running GPU examples. They require a supported
NVIDIA GPU. Installing an unrelated upstream wheel does not install these
cavity changes. The [C2 guide](response_shape_validation/c2/README.md) describes
the complete calibration workflow.

The small [native coupled-ringdown example](../../examples/afm_cavity_ringdown.py)
uses the integrated physical coupling API. From the repository root:

```sh
python examples/afm_cavity_ringdown.py --mumaxplus-fp-precision DOUBLE
```

## Fast CPU identity demonstration

Run all commands below from the repository root, using a fresh output directory:

```sh
python extensions/photon-magnon/run_cpu.py demo --output extensions/photon-magnon/runs/demo
```

The helper generates **analytic synthetic** free-response coefficients from the
included fixed-pole model, freezes nine physical-field EP predictions, and
checks them with the original-sublattice six-dimensional generator. It needs no
GPU and distributes no precomputed data. This tests mathematical consistency;
it is **not** an independent free-time calibration or a reproduction of the
original GPU prediction errors. The predictor receives physical susceptibility
coefficients and declared cavity/port inputs, not material truth parameters.
The later validator receives the separate generated model parameters.

The three losses are the central value and half/double that value. The common
coupling response settings are explicitly prepared using nominal field-overlap
information; they are distinct from the material-input-free EP-field prediction.

## C1: complete free-pole and strength-matched comparison

```sh
python extensions/photon-magnon/run_cpu.py zero-field --output extensions/photon-magnon/runs/zero-field
```

This regenerates the 37-point zero-field input and common-point response from
source, then runs C1 and an independent quartic/EP audit. It can take longer than
the demo because of high-precision determinant and response checks. The scalar
control uses a passive reference with physical coupling rescaled to match the
target numerator strength. Static-normalized free susceptibility and absolute
port-normalized cavity transmission are separate comparisons. Generated plots,
JSON and NPZ files remain in the chosen run directory.

## C2 to C4: independent GPU calibration and prediction

Use the [C2 instructions](response_shape_validation/c2/README.md) to generate
new two-seed free-spin and known-field pulse records, fit the public observations,
and freeze the resulting calibration. For the commands below, place C2 output
under `extensions/photon-magnon/runs/gpu/response_shape_validation/c2/`.

```sh
python extensions/photon-magnon/run_cpu.py predict --output extensions/photon-magnon/runs/gpu
python extensions/photon-magnon/run_cpu.py bounds --output extensions/photon-magnon/runs/gpu
```

C3 copies the new frozen calibration, predicts the physical EP field and cavity
frequency, then checks all nine EP2 conditions and the complex response in the
full generator. It refuses to overwrite frozen predictions. The validator alone
uses `c2/private/truth_by_case.json`; "private" describes the analysis boundary,
not secret research data distributed with this package.

C4 consumes actual C2 step/amplitude/window envelopes, C3 predictions and their
validation. It evaluates finite envelope corners and separate scalar nuisance
fits. The CPU identity demo supplies no convergence envelopes and cannot be
used for this stage. Envelopes are deterministic numerical variations, not
experimental confidence intervals; sampled corners are not certified global
bounds. Diagnostic fits do not alter the frozen free-response prediction.

## Source layout and provenance

- `single_message_ep/primary/`: fixed-pole generator and bundled energy model.
- `single_message_ep/independent/`: separate real-quartic/EP verification.
- `response_shape_validation/c1/`: strength-matched reference and exact response.
- `response_shape_validation/c2/`: new GPU records, public-only fitting and freeze.
- `response_shape_validation/c3/`: public-coefficient prediction and later full validation.
- `response_shape_validation/c4/`: numerical envelopes and nuisance diagnostics.
- `source_map.json`: hashes of the validated code and its portable source copy.

Numerical equations and solver settings are retained from the validated source.
Portability changes replace local loaders and output paths, separate source from
generated data, and remove internal document references. Reproduction on another
BLAS/CUDA platform should be assessed by physical tolerances and convergence;
byte-for-byte output equality is not promised. The code uses the repository's
license. No legacy empirical normalization or phase-only cavity model is used
by these entry points.

## Portability checks

The public synthetic demo and the complete source-generated zero-field/C1
workflow were executed with Python 3.12. The synthetic demo passed nine EP2
and three common-response checks. C1 passed the physical strength-matching,
passivity and exact two-coordinate comparisons.

Separately, the portable C3/C4 scripts were tested against the original inputs
in a private directory outside the Git checkout. C3's scientific predictions
and full-generator validation matched exactly; C4's scientific payload also
matched exactly after excluding runtime/hash metadata. Those inputs and results
are not shipped. This verifies source portability; it does not replace a fresh
GPU convergence campaign on a different build or machine.
