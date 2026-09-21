# Source update, 21 September 2026

## Scope

The public source tree was updated in an isolated checkout. The active research
workspace, its compiled engine, source hashes, and running checkpoints were not
modified. Unrelated SOT, interlayer-DMI, antiferromagnetic, and topology changes
from that workspace are not part of this update.

The corrected analysis lives in `revision_check/`. The historically published
`src/` analyses are not silently rewritten into different estimators. The old
sim39 phase sign and sim40 overlapping bands remain documented limitations of
those legacy entry points. New figure-generation sources are included as code,
not as validated figures or results.

Only the SAW-relevant engine changes were ported: host clock precision, thermal
RNG access, thermal Gaussian buffer/error handling, and the chiral-SAW field
diagnostic binding. This public build is a separate artifact, not the binary
that produced the private campaign data. Its commit cannot certify old runs.

## Fresh build and reference preparation

The `runs/build_forward.py` helper targets the author's Windows/Visual Studio
2022, CUDA sm_89, Python environment. Other systems need an equivalent documented
build and receipt, not disabled provenance checks. It must target a new empty
build directory. For a compatible environment:

```text
git submodule update --init src/bindings/pybind11
python extensions/SAW-magnonics/revision_check/runs/build_forward.py --build-dir <new-build-directory>
```

Select that build's `lib` directory through `PYTHONPATH` and its `CMakeCache.txt`
through `SAW_BUILD_CACHE`. Confirm that the imported Python package and extension
module belong to the intended checkout and build. Source/build identity checks
must pass before any simulation is authorized.

In `revision_check/claim_recovery_20260921`, regenerate the spatial reference with
`spatial_linear_check.py`. It writes identity-checked operator checkpoints and
`spatial_linear.json`. The latter is an input to `direct_check.py` and
`validation_campaign.py`, not a supplied or assumed calibration. Read
`VALIDATION_PLAN.md` before preflight and production. The private batch's old
reference files and checkpoints must not be stamped as belonging to this build.

## Verification scope

The data-independent portion of `test_saw_analysis.py` passes: 13 tests, with two
raw-data-dependent tests deselected. An initial attempt to run the full script
correctly raised `FileNotFoundError` for the absent `sim37_suhl_control.npz`;
that attempt is not reported as a passing full-suite run.

Before integrating the concurrent remote AFM update, the public checkout was
clean-built with Visual Studio 2022, CUDA 12.8, and
Python 3.14.2. Its forward build receipt records unchanged source hashes before
and after compilation; the provenance module returns `LINKED_VIA_BUILD_TREE`
with `blocks_downstream=false`. Tests explicitly imported this checkout's Python
package and its newly built binary, not the running research batch's engine.

- Clock precision and existing time-solver regression tests: 38 passed.
- Thermal stream tests: 6 passed on the final test run, covering odd/even grids,
  64-bit seed replay, stream changes, zero Ms, and zero temperature. The first
  run had one test-fixture error (a missing scalar-component array dimension);
  the fixture was corrected and all six thermal tests were rerun. No engine
  behavior or numerical tolerance was changed to make this test pass.
- Python compilation: succeeded. A pre-existing invalid-escape SyntaxWarning
  in the legacy report string in `step3_band_content.py` remains.
- Staged-diff whitespace and source-only scope checks: passed.

The remote AFM/NcAfm update was preserved by rebasing, without force-pushing.
Its modular source layout exposed incomplete discovery in the SAW build-receipt
helper: the independent object-source check correctly blocked three compiled
translation units outside `src/`. Discovery now covers CMake-backed modules in
both extension layouts, their headers and CMake includes, and new core source
subdirectories. It does not whitelist individual kernel names or disable the
independent coverage check. The new regression tests went from three failures
and one pass before the fix to four passes after it; the combined CPU suite
then passed 17 tests with the same two data-dependent exclusions.

A fresh clean build after that discovery fix records the expanded source set
before and after compilation. The combined checkout and its loaded binary now
return `LINKED_VIA_BUILD_TREE` with `blocks_downstream=false`. The incomplete
earlier receipt was preserved separately, not restamped as a valid build.
All 44 clock, thermal-stream, and existing time-solver tests were rerun against
this final combined checkout and fresh binary and passed (274.71 s). The import
paths and linked build status were asserted before starting those tests.

No historical raw-data reanalysis, full G4/G9 campaign, optical calibration, or
complete thermal validation is certified by this source update. Data-dependent
record/verdict tools need their referenced evidence files, which are not part
of a source-only snapshot. The copied `src/plot_style.py` supplies the plotting
helper used by historical drivers; this does not validate their figures.

## Current scientific limits

- Old sim40: `indistinguishable`; neither single-wave sufficiency nor an artifact
  cause is certified.
- The prospective threshold is confined to the specified q/2 sector.
- Initial-phase ensemble nulls are conditional on their specified distribution.
- Thermal phase statistics must be reported separately and require the actual
  completed thermal ensemble and preparation diagnostics.
- Required photon counts assume an optical transfer/noise model; no absolute BLS
  SNR is known without measured detector calibration.
- Source publication does not change the manuscript title, figures, or claims.
