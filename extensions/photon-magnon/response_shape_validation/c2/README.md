# C2: free-response calibration from fresh GPU records

This source package contains no recorded magnetization, fitted coefficients,
frozen predictions, manuscript files, or private machine paths. Run it with a
DOUBLE-precision build of this checkout and a CUDA-capable GPU. NumPy and SciPy
are needed for the public fitter; the fitter itself does not import mumaxplus.

From the repository root, with the built package available to Python:

```sh
python extensions/photon-magnon/response_shape_validation/c2/run_gpu.py --output extensions/photon-magnon/runs/gpu --check-runtime
python extensions/photon-magnon/response_shape_validation/c2/run_gpu.py --output extensions/photon-magnon/runs/gpu
python extensions/photon-magnon/response_shape_validation/c2/fit_public.py --output extensions/photon-magnon/runs/gpu
python extensions/photon-magnon/response_shape_validation/c2/validate_and_freeze.py --output extensions/photon-magnon/runs/gpu
python extensions/photon-magnon/run_cpu.py predict --output extensions/photon-magnon/runs/gpu
python extensions/photon-magnon/run_cpu.py bounds --output extensions/photon-magnon/runs/gpu
```

All commands use a common run root. C2 writes below
`extensions/photon-magnon/runs/gpu/response_shape_validation/c2/`. Choose a fresh output root for a new
build or source version. Runtime records include local module paths and hashes;
keep the whole generated run directory private. The repository ignores generated
data. The runner uses the standard installed import without changing Python's
search path or adding DLL directories. Configure any platform-specific library
loader requirements when installing/building mumaxplus, outside this script.
It requests DOUBLE precision when the environment has no explicit preference and
rejects an imported SINGLE build. `--expected-module-sha256 DIGEST` optionally
checks a newly built binary supplied by the caller; no old binary is assumed.

## Protocol and separation

The default final campaign generates 28 records for three declared theoretical
cases at zero static field. It keeps the free-pole target fixed while varying
the material tuple. Case C is a mathematical large-contrast example, not a claim
of accessible material tuning. Both sublattices receive the same physical pulse;
the native total field enters Gilbert precession and damping. The gyromagnetic
ratio is explicitly set to `1.7595e11 rad/(s T)` for both sublattices.

| Quantity | Cases A and B | Case C |
| --- | --- | --- |
| Free timesteps | 25, 12.5, 6.25 fs; half seed at 6.25 fs | 25, 12.5 fs |
| Pulse timesteps | 25, 12.5 fs; half drive at 12.5 fs | 25, 12.5 fs |
| Final pole selection | 6.25 fs, half seed | 12.5 fs |
| Final pulse selection | 12.5 fs | 12.5 fs |
| Record durations | Free 40 ps; pulse 80 ps | Free 40 ps; pulse 80 ps |

All records are sampled every 200 fs. Free seeds have amplitude `1e-6` (or half)
in one physical sublattice x component. The physical x-field pulse has amplitude
2 microtesla (or half), a sine-to-the-fourth envelope from 2 to 42 ps, and a
261 GHz carrier. Free fitting windows are 0–40, 5–32, and 10–40 ps. Pulse fitting
windows are 0–80, 2–64, and 10–80 ps. Optional `--group base` and `--group refine`
produce coarse 200 ps convergence records; use a separate output root if a
variant name was already generated. The finer optional `precision6` group also
generates 6.25 fs pulse and half-drive records. `--case A` and `--only free` or
`pulse` allow the campaign to be scheduled in pieces.

`public/` contains only time, physical spins, `Mx=mAx+mBx`, the known field and
waveform metadata. `private/` contains the generated model setup for later
validation. The public fitter reads only `public/`; no exchange, anisotropy,
damping parameter, squeezing ratio, analytic numerator, or coupled spectrum is
an input. An optional `--public DIR` supports fitting an isolated copy of public
observations. Its read manifest records paths relative to that directory. This
override is for isolated fitting; before freezing, copy the same observations
into the chosen run root's `response_shape_validation/c2/public/` directory.
The freeze step rechecks all hashes there.

Two independent free seeds identify a four-channel one-step propagator. The
fitter retains all four poles and averages their real and absolute imaginary
parts. A Green-function convolution fits the physical numerator without
differentiating observed magnetization. Column normalization avoids a unit-driven
conditioning artifact. Final timestep, amplitude and window variations are
combined with the four free-pole envelope corners. These deterministic
variations and the formal OLS residual covariance are not experimental error
bars, confidence intervals, or certified global bounds.

`validate_and_freeze.py` checks the full protocol and public read hashes, writes
`calibration_frozen.json` without overwriting existing bytes, and only then reads
the generated private model inputs. `--resume-validation` verifies the frozen
and fitter digests before repeating validation. Partial development fits made
with `fit_public.py --allow-partial` cannot be frozen. C3 receives the frozen
public coefficients; only post-prediction validation reads `private/truth_by_case.json`.
The numerical target is one tenth of each shape effect, which must be checked
separately for rho, EP location, coupling, and complex response. Passing a rho
check alone does not establish the accuracy of the other observables.

The model formulas and public example parameters are visible in the generator.
This is a reproducible computational separation of fitting and validation, not
an externally blinded experiment. The input/readout normalization and phase are
known; instrumental noise, unknown gain, and empirical calibration bounds are
not inferred here.
