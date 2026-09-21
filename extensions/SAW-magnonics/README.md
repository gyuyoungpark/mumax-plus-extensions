# SAW-magnonics extension for mumax+

Source snapshot for the ongoing revision of the submitted manuscript
*Finite-Momentum Parametric Magnon Pairing by Traveling Surface Acoustic Waves*.
The historical title identifies the project, not a newly certified result.

## September 2026 source update

This update publishes the corrected analysis modules, checkpoint and provenance
guards, and follow-up calculation drivers. It does not publish unfinished runs,
raw magnetization arrays, a manuscript revision, or an experimental detection
claim. The old sim40 model comparison remains **indistinguishable**; spatial
localization alone establishes neither two-mode pairing nor a boundary artifact.

| Location | Purpose |
|---|---|
| `src/` | Historical drivers and figure-generation sources, plus later simulations |
| `revision_check/saw_analysis.py` | Shared Fourier conventions, disjoint power bands, and conditional phase nulls |
| `revision_check/step1_ipair.py`, `step1b_curves.py` | Reanalysis of the overlapping legacy sim40 bands |
| `revision_check/step2_coherence.py` | Corrected pump-phase sign and explicit selection of the requested null row |
| `revision_check/model_comparison.py` | Spatial model comparison; held-out prediction is distinct from training fit |
| `revision_check/growth_interval.py` | Initial-growth measurand and refusal of unsuitable fitting intervals |
| `revision_check/runs/` | Condition-matched checkpoints, build receipts, and campaign prerequisites |
| `revision_check/claim_recovery_20260921/` | Spatial LLG reference, matched controls, and prospective validation drivers |

The current scope and fixed criteria are in
[`VALIDATION_PLAN.md`](revision_check/claim_recovery_20260921/VALIDATION_PLAN.md).
The planned checks are box/grid convergence, independent-seed MEL/OFF/MR phase
statistics, a **q/2-sector** threshold bracket, and 300 K on/off comparisons.
Conditional photon-count requirements are not an absolute BLS signal-to-noise
ratio. No measured detector calibration is available.

## Historical analyses are not the corrected estimators

The old `src/sim39_normalized_correlator.py` and `src/analyze_sim40.py` are retained
for provenance. Their legacy pump-phase correction and overlapping pair-band
definition must not be used to reinstate the original phase-discrimination or
finite-k-only claims. Use the `revision_check/` modules above. Some old scripts
can overwrite summaries when required raw inputs are absent; do not run them
against the only copy of a dataset. Not every historical script has a validated
checkpoint implementation or literature-material parameter set.

## Engine changes

The production SAW kernels are under the repository's `src/physics/`, not under
this extension's Python directory. The production chiral-SAW kernel is unchanged
in this update. The relevant shared-engine changes are:

- Double-precision host time accumulation, retaining the selected field precision.
- Explicit `Ferromagnet.thermal_seed` access for reproducible random streams.
- Checked Gaussian generation for odd cell counts and zero noise at zero Ms.
- Python binding for the chiral-SAW field diagnostic.

Resetting `thermal_seed` resets the random stream; it does not restore a solver's
already drawn noise or integration history. The validation driver instead saves
at declared block boundaries and replays independently seeded blocks.

## Reproduction and checks

Build this checkout using the top-level installation instructions. Python
analysis uses NumPy, SciPy, matplotlib, threadpoolctl, and pytest. The author's
current driver environment is Python 3.14; portability is not certified solely
by publishing the sources.

From this directory, the data-independent analysis regression command is:

```bash
python -m pytest revision_check/test_saw_analysis.py -q -k "not null_row_is_requested_target and not sim40_resolution"
```

The two deselected tests require historical raw data; they are not counted as
passing checks. The full script also requires those inputs.

Source publication is not a complete calibrated run archive. The guarded GPU
drivers require a build receipt and locally regenerated spatial-reference files.
See [`SOURCE_UPDATE.md`](SOURCE_UPDATE.md) for preparation and limitations.
Do not import a different build's checkpoints, change their stamps, or bypass a
missing-input guard. Raw data and live outputs are deliberately not included;
see [`DATA_ARCHIVE.md`](DATA_ARCHIVE.md).

## License

GNU General Public License v3.0, consistent with the parent mumax+ repository.
See the top-level `LICENSE` file.
