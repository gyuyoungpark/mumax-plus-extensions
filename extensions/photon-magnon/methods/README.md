# Explicit spin-cavity energy methods

This is an experimental, source-only Python API beside the existing photon-magnon
examples. It does not replace their entry points or the repository's native AFM
cavity implementation. It is not a drop-in replacement for older `CavityMagnon`
or `CavityMagnonAFM` classes.

Both interaction directions follow one classical energy:

```text
E = E_spin + sum_j E0_j (q_j^2 + p_j^2)
    - sum_i,j mu_i h_j q_j <u_j m_ix>
dq_j/dt = omega_j p_j
dp_j/dt = -omega_j q_j - 2 kappa_j p_j
          + omega_j/(2 E0_j) sum_i mu_i h_j <u_j m_ix>
Bx = sum_j h_j q_j u_j
```

Fields are flux densities in tesla, angular frequencies are in rad/s, and
`mu_i = Ms_i * occupied_volume_i` is a physical magnetic moment in J/T. The real
`q,p` coordinates are dimensionless energy coordinates, not photon amplitudes.
For positive gamma the spin convention is `dm/dt = -gamma m x B`, with
`m_minus = mx - i*my`. The optional circular-FM rotating-wave map is conditional
on its stated small-angle assumptions; it is not an AFM normalization formula.

The supported cavity loss is the viscous term `-2*kappa*p`. Nonzero legacy
`g_diss`, a phase-only coupling, spatial profiles and claims of native equivalence
are explicitly rejected by the uniform adapter. Nonnegative losses alone do not
prove stability: the coupled energy about the chosen equilibrium must also have
the appropriate positive quadratic form.

## Evidence and limitations

The original frozen implementation had 31 finite mathematical/software controls
and one lossless uniform CPU FM pilot with two independent full-state reference
records. These observations apply to the original exact source hashes identified
in [source_map.json](source_map.json). They do not automatically transfer to the
edited public hashes. No physical trajectory, FFT, fit or native execution was
rerun for this port.

The public changes include a restore-failure invalidation fix, stronger AFM
partner-metadata checks, portable test temporary directories and documentation.
The local suites exercise only standard-library metadata/I/O and inert callbacks.
They are not evidence for coupled AFM dynamics or an actual native engine.
Neither the original finite pilot nor these local checks establish prototype
convergence order, damped power balance, spatial or multimode dynamics, passive
dissipative coupling, GPU performance, historical spectra or device precision.

`UniformCosimulationAdapter` requires caller-supplied engine and source hooks.
The engine owns magnetic integration and its complete solver/RNG/parameter state.
The caller must provide the actual homogeneous moment and an average over the
same occupied magnetic volume. Every coupled AFM sublattice must have a distinct
binding, the correct moment and a shared host declaration. Partner guards check
the consistency of these declarations; they cannot prove that callbacks refer
to the claimed physical objects or that all actual host sublattices were listed.
AFM use remains an **unvalidated caller contract**.

The adapter supports initial-condition excitation only. It records synchronous
pre-step/end-step samples and complete checkpoints. A failed restore invalidates
the adapter until a complete checkpoint is successfully restored. Saving or
auditing checkpoints does not establish an empirical crash/resume test of a
physical engine. A cavity-only `q,p` checkpoint is insufficient.

## Source layout

- `src/cavity_method/energy.py`: explicit moments, energy, reciprocal field/source.
- `src/cavity_method/normalization.py`: conditional circular-FM action/RWA map.
- `src/cavity_method/adapters.py`: uniform callback adapter and full-state checks.
- `src/cavity_method/spectrum.py`: full signed complex spectra and branch coverage.
- `src/cavity_method/provenance.py`: explicit file identities and atomic I/O.
- `analysis/analyze_raw.py`: an explicit stored-raw analysis CLI, with no simulator
  or fit command. See [the analysis guide](analysis/README.txt).
- `tests/`: metadata/I/O suites only; no dynamics or FFT tests.

Importing `cavity_method` uses the Python standard library only. NumPy is loaded
only inside explicitly called array/FFT operations; no magnetic binary is loaded
by the package. The two metadata suites need only Python 3.12. NumPy, available in
the parent extension's CPU requirements, is needed for a separately requested
raw-array inspection or Fourier operation.

## Metadata checks and help

From this `methods` directory, set the source import path and run only the
metadata suites. For a POSIX shell:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p '*metadata_io*.py' -v
python analysis/analyze_raw.py --help
python analysis/analyze_raw.py full-spectrum --help
python analysis/analyze_raw.py source-identities
```

For PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -m unittest discover -s tests -p '*metadata_io*.py' -v
python analysis/analyze_raw.py --help
python analysis/analyze_raw.py full-spectrum --help
python analysis/analyze_raw.py source-identities
```

The help and source-identity commands execute no arrays, FFTs or simulations.
The tests create opaque metadata fixtures in the operating system's temporary
directory; they do not require a repository `reports` folder. Generated analysis
outputs must be new files under `methods/analysis_outputs` and are ignored by Git.

Full Fourier analysis requires an exact operator-supplied review record and an
explicit execution flag. That record binds raw data, channel/sign declarations,
sampling, configuration and current source identities; it is not a cryptographic
proof of human identity. The CLI enforces this boundary. Low-level library
functions are callable utilities whose callers retain responsibility for their
own analysis authorization. No source-only publication authorizes new scientific
execution or repairs missing historical provenance.

No campaign runners, raw records, fitted results, manuscript files, private
approval records or binaries are distributed here. This code uses the repository
license.
