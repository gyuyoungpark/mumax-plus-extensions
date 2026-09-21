# PRE-REGISTRATION — sim40 eps_0 = 7e-5 one-component vs two-component test

Written and saved **before** any M1/M2 fit was run. Everything below was fixed
from (a) the run script `src/sim40_eps_kresolved.py`, (b) `CONVENTIONS.md`, and
(c) one exploratory look at the f_K-slice bin powers and the per-bin growth
rates (recorded in `EXPLORATORY_LOOK.txt`). No held-out error, no model fit and
no selection outcome had been computed when this file was written.

## 1. Analysis object

For time block b (contiguous, non-overlapping, index range [t0_b, t0_b+L_b)):

    u_b(x) = sum_t w_Hann(t) m_y(x,t) exp(+i 2 pi f_0 t),   f_0 = 3.000 GHz = f_SAW/2
    a_j^(b) = (1/N_x) sum_x u_b(x) exp(-i k_j x),  k_j = j*dk,  dk = 2 pi / L

f_0 is the exact Kittel/half-pump frequency, NOT a DFT bin centre: the sign
convention is `saw_analysis`'s (M(k,f) = sum m w exp(-ikx + i2 pi f t)), so a
wave exp(i(k_0 x - 2 pi f_0 t)) with k_0>0 lands at positive j.

Blocks: the published "second half", samples 375..750 (376 samples, 7.52 ns),
split into **4 non-overlapping blocks of 94 samples (1.88 ns)**.
TRAIN = blocks 0,1 (t = 7.50-11.26 ns). HELD-OUT = blocks 2,3 (t = 11.26-15.00 ns).

## 2. Background — identical for both models, fixed before fitting

1. `j = 0` (uniform FMR background) is **excluded**.
2. `j < 0` (counter-propagating content) is **excluded**.
3. **Pump background**: one wave at the FIXED physical pump wavenumber
   kappa_q = q/dk = 8.777143 (q = 2 pi * 6 GHz / 3500 m/s = 10.771175 um^-1),
   complex amplitude estimated per block by least squares **on bins {8,9,10}
   only**, then its exact periodic-Dirichlet profile subtracted from every bin.
   Bins {8,9,10} are then excluded from all fitting and all evaluation.
4. `j >= 11` excluded (measured content < 0.1% of slice power).

Analysis band **A = {1,2,3,4,5,6,7}**. This background enters M1 and M2 as the
same fixed pre-processing; neither model gets a background parameter the other
does not.

## 3. Models (fitted to the COMPLEX coefficients, never to power)

Basis: the exact periodic Dirichlet response of this transform,
`D_j(kappa) = (1/N) sum_n exp(i 2 pi (kappa - j) n / N)`, N = 1024.

    M1:  a_j^(b) = A_b D_j(kappa)                      kappa shared over blocks
    M2:  a_j^(b) = A_b D_j(kappa_a) + B_b D_j(kappa_b) kappa_a, kappa_b shared

kappa's continuous and off-grid-allowed; per-block complex amplitudes by
variable projection (exact linear solve at each kappa).
Parameter count, TRAIN (2 blocks): M1 = 1 + 2*2 = 5 real; M2 = 2 + 2*4 = 10 real;
data = 2 blocks * 7 bins * 2 = 28 real.

## 4. Held-out prediction error — the PRIMARY basis

Wavenumbers are frozen at their TRAIN values. On each HELD-OUT block:

* amplitudes re-estimated on the **calibration bins C = {4,5,6}** only;
* prediction error evaluated on the **disjoint evaluation bins E = {1,2,3,7}**;

        E_model = sum_{b in HELD-OUT} sum_{j in E} |a_j^(b) - ahat_j^(b)|^2
                  / sum_{b in HELD-OUT} sum_{j in E} |a_j^(b)|^2

Bins in E are never used to estimate anything for that block. A secondary split
C = {4,5}, E = {1,2,3,6,7} is reported as a robustness check with its own
controls; the primary is C = {4,5,6}.

## 5. TOLERANCES — FIXED HERE, IN ADVANCE

    tau_adeq = 0.25    "adequate" = held-out normalised prediction error <= 0.25
                       (the model predicts >= 75% of held-out tail power)
    rho      = 0.50    "M2 better" = E_M2 <= 0.50 * E_M1 (factor-2 error drop)

Decision rule, applied in this order:

1. If `E_M2 <= tau_adeq` AND `E_M2 <= rho * E_M1`  -> **two_components_required**
2. Else if `E_M1 <= tau_adeq`                      -> **single_wave_sufficient**
3. Else                                            -> **indistinguishable**,
   sub-labelled `both_models_inadequate`.

Overriding power gate, also fixed here: if the control **false-positive rate
> 0.05**, outcome 1 is downgraded to `indistinguishable / insufficient_power`;
if the control **detection rate at the observed amplitude ratio < 0.80**,
outcome 2 is downgraded to `indistinguishable / insufficient_power`.

## 6. Controls the rule must be calibrated on

Identical pipeline, identical rule, >= 200 realisations each, noise injected at
the measured residual level into bins 1..10 before pump estimation, so the whole
chain (pump removal, kappa search, M1, M2, selection) re-runs per realisation.

* (d) ONE off-grid wave -> **false-positive rate**.
* (e) TWO components, second-component relative amplitude r scanned
  -> **detection rate vs r**, for an exact-bin pair and for an off-grid pair
  at several separations.

## 7. Residual model for the parametric bootstrap

sim40 is deterministic; the residual is not a noise level. Three explicitly
stated residual models are run and compared:
R1 white (i.i.d. complex Gaussian, one common variance),
R2 per-bin white (i.i.d., measured per-bin variance),
R3 correlated (wild bootstrap of whole measured residual bin-vectors, phases
randomised per block: preserves the measured bin-to-bin correlation).
The result is reported as an exceedance rate UNDER THE ASSUMED RESIDUAL MODEL.

## 8. What this test cannot decide

The published pipeline FFTs the full periodic box with no spatial window. An
off-grid single wave is not an eigenmode of a periodic box. "M1 adequate"
therefore licenses only *one spatial component suffices to describe the
profile*, never *one physical eigenmode*. Likewise "two components" is a
statement about the spatial profile; pair generation needs separate dynamical
verification.

<!-- BEGIN GENERATED FROM G9_CRITERIA.json -- do not edit by hand -->

## PRE-REGISTRATION -- G9 idler-injection decision rule

Generated from G9_CRITERIA.json, version 2026-09-18-common-path.
Regenerate with `python runs/_criteria.py render`; verify with `python runs/_criteria.py check`.

### G9.1 Growth-rate compatibility (P3)

    compatible <=> |Gamma_1 - Gamma_2| <= K_SIGMA * hypot(sigma_stat, sigma_sys) + tol_phys

Conditional compatibility under a common eigenmode and the declared synthetic envelope/noise family. sigma_stat is an approximate statistical uncertainty; sigma_sys is a synthetic-calibrated estimator allowance in standard-error units, not an independently identified variance. Their quadrature is an operational rule, not a measured confidence level or physical false-positive rate.

| term | value / formula | basis |
|---|---|---|
| `K_SIGMA` | `3` | Declared multiplier 3 for the operational error budget. Approximate overlap adjustment and empirical systematic calibration do not establish three-sigma coverage. |
| `sigma_stat` | `sqrt(seg_len/stride) * hypot(se_1, se_2)` | OLS errors of the GI-selected ln-amplitude slopes, multiplied by sqrt(seg_len/stride) as an approximate overlap adjustment. It does not determine residual autocorrelation or cross-member covariance and is not guaranteed coverage. |
| `sigma_sys` | `f_sys * max(|Gamma_1|, |Gamma_2|)`; f_sys = `3.826173575940336e-09` | Minimum additional scale needed to cover the finite calibration set. This construction neither isolates the true systematic variance nor establishes coverage outside that set. |
| `tol_phys` | `0` | Both components of a single eigenmode have the same exponential eigenvalue, hence zero allowed physical rate difference within this scope. ALPHA*abs(omega1-omega2) is a spread of free damping rates and is not a bound on transient logarithmic slopes of superposed modes. Transient or invalid fits cannot be classified by this equality test. |

Physical scope: linear common-eigenmode exponential growth on the same GI-valid early interval

Invalid fit scope, negative/non-finite uncertainty, malformed geometry or stale/unmeasured calibration makes P3 NOT_DETERMINABLE and G9 NOT_EVALUABLE, never evidence against pair generation.
Both rates must be positive. No relative-error OR clause is allowed.

Calibration: G9.pair_rates and G9.rate_comparison with shared GI fits; 130/166 calibration records eligible. Calibration seeds 7,11; independent validation seeds 101,103,107. Reduced nx=128; transfer to the campaign grid and physical noise is unvalidated.
Evidence: runs/out/G9_common_path_validation_v4.json; n = 130; measured UTC = 2026-09-18T05:47:04.982205+00:00.

Fit applicability: Both GI fits must be OK, use exactly the same interval, have no detected turnover, and agree with a centered independent OLS recomputation. These are necessary diagnostics, not proof of eigenmode dominance. Any failure is NOT_DETERMINABLE.
Linearity limit: 0.1.
Growth layout: 64 samples. Fixed 64 samples independent of acquisition duration; at DT_REC=20 ps the support is 1.28 ns and bin spacing 781.25 MHz. This preserves early time resolution but does not certify material-specific frequency resolution. Coherence retains its previous max(64,nt//8) layout separately.

### G9.2 Other predicates

| symbol | value | meaning |
|---|---|---|
| `P1_over_nopump_dB` | `5` | the partner bin's late amplitude must also exceed the pump-off control's by at least this many dB |
| `P1_partner_rise_dB` | `10` | power in the partner bin must rise by at least this many dB over its own value in the first tenth of the record |
| `P2_resolution_floor_MHz` | `1` | floor on the frequency-sum tolerance, used when the growth-broadened record resolution is narrower than this |
| `P4_null_percentile` | `99` | percentile of the pair-coherence surrogate distribution that C(k_1) must exceed |
| `control_max_rise_dB` | `10` | a control's partner bin must rise by LESS than this, i.e. stay dark, for the nulls to hold |

### G9.3 Prerequisites

Validate all required records before spectra or predicates: scalar boolean done, finite 2D samples, complete counters, declared and stamped conditions matching each other and the analysis geometry, and recorded role requirements. Missing, incomplete, ambiguous, malformed or mismatched inputs yield NOT_EVALUABLE and a summary. No unstamped legacy record is certified.

Condition matching uses _conditions.compare on stamps and _control_match.match on declarations.
Selection: A complete finite nonnegative table over every non-degenerate grid bin is required. Ties use the smallest bin, matching choose_k1; reciprocal j and N-j necessarily have equal pair mismatch. A flat table cannot define distinct resonant and spectator cases.

**inj_res**: the test itself: injection at the resonant grid mode
Allowed declaration differences: .
Allowed stamped differences: .
- eps0 != 0: a pump-off record cannot be the pumped test
- inject_bin is the argmin over grid modes of the recorded pair mismatch |f(k)+f(q-k)-f_SAW|: the run is only the resonant case if its injected bin is the one the MEASURED dispersion picks out; with no recorded mismatch table the claim cannot be certified and the analysis is NOT_EVALUABLE rather than assumed

**inj_res_nopump**: pump-off control: the partner bin must not light up with eps_0 = 0
Allowed declaration differences: eps0.
Allowed stamped differences: pump.eps_0.
- eps0 == 0: a control run at non-zero strain is not a pump-off control
- inject_bin == inj_res.inject_bin: the pump-off control must interrogate the SAME partner bin, or it does not bound that bin's background

**inj_spec**: spectator control: same injection amplitude at the LARGEST pair mismatch; its partner bin must stay dark
Allowed declaration differences: inject_bin, inject_k, partner_bin, partner_k, initial_state.
Allowed stamped differences: .
- inject_bin is the argmax over grid modes of the recorded pair mismatch |f(k)+f(q-k)-f_SAW|: REQUIRED_CONTROLS defines the spectator as the FARTHEST-off-resonance mode. Requiring only 'a different bin' let bin 2, adjacent to the resonant bin 3, serve as the null, so a generic response to injection could not be distinguished from a resonant one and the verdict still read DETECTED (counterexample A8)
- spectator and resonant injected bins differ: a flat mismatch table does not furnish a distinct spectator

### G9.4 Coherence surrogate

P4 uses a block-phase surrogate with one product phase per ceil(seg_len/stride) segments. This preserves phases within blocks only; samples also overlap across block boundaries, and physical correlation may extend farther. The threshold is a conditional surrogate level, not a calibrated physical false-positive rate.
No false-positive rate is quoted from any variant. The threshold is reported as an exceedance LEVEL of a named phase surrogate, and the independent-phase level is kept only as a diagnostic showing how much the overlap matters.
P4 uses block_phase.

**independent_phase**: one independent uniform phase per segment applied to the PRODUCT m1*m2, measured magnitudes kept; distributionally identical to the historical SA.pair_correlator_null, where two independent uniform phases were drawn and their sum is again uniform per segment. a reference level ignoring overlap, used only as a diagnostic

**block_phase**: one independent product phase per block of ceil(seg_len/stride) segments, with measured magnitudes retained; partial treatment of overlap, not the full covariance. a conditional block-phase surrogate exceedance level; neither physical null coverage nor uniformly greater conservatism is guaranteed

<!-- END GENERATED FROM G9_CRITERIA.json -->

## Resumed Early-Window Clarification (2026-09-18)

This clarification applies to the existing early-linear measurand, not to a
new physical model. Eligibility ends at the first record-level linearity
failure or amplitude saturation cut. Later re-entry below the ceiling cannot
restore eligibility. A later readable slope is a separate diagnostic and
cannot replace an unread early rate or supply a negative threshold point.

The growth Fourier window is fixed independently of total acquisition length:
the present default is 64 records, stride 16. At 20 ps cadence this is 1.28 ns
support and 781.25 MHz frequency-bin spacing. The duration and resolution are
saved with the growth result. This is not evidence that nearby physical modes
are resolved. G9 keeps its separately declared coherence window and reports
the resulting null comparison as conditional on its surrogate assumptions.

Total acquisition time is not the fit interval. The current planner retains
its conservative predicted-linear-lifetime ceiling; the fitter independently
applies signal, first-linearity-failure, e-fold and adequacy checks. A readable
late tail does not rescue an unread early interval. No runtime extension or
material-parameter inflation is introduced by this correction.

An onset threshold certificate additionally requires complete current
source/build/condition stamps for every contributing input. Old arrays may be
inspected descriptively, but an unstamped or mismatched row cannot authorize
the next campaign stage.

---

## 9. THE GROWTH-RATE MEASURAND — fixed here, in advance, and shared with the code

Added 2026-09-18 (round 2). Sections 1–8 pre-register the ONE-vs-TWO component
test on the existing `sim40` record; they say nothing about what a *growth rate*
is, and that omission is what let a fitted interval be chosen after the fact. The
first round of corrections then classified a grow-then-damp record from its
overall slope and reported the DECAY branch as the growth rate, with
`status = ok` and `r2 = 1.0000`. Fixing the interval rule alone would not have
closed that: the defect was that no quantity had been named, so any interval with
a good `r2` could stand in for any other.

The statement below is the definition. It is not a paraphrase of the code: it is
the string `growth_interval.MEASURAND`, and `test_p2_measurand.py` compares this
block with that constant **byte-for-byte**, so the specification, the
implementation and every printed row cannot drift apart. The numerical criteria
(`snr_amp = 4`, `min_efold = 1.0`, `min_r2 = 0.9`, `turn_sse_frac = 0.5`,
`turn_efold = 1.0`, `min_points = 5`) are fixed here, before the campaign is run,
and are printed with every fit under `criteria`.

```
MEASURAND (growth rate). The quantity every Gamma in this campaign reports is
the growth rate of the EARLY LINEAR instability of one wavenumber bin at one
frequency slice: the slope of ln|a| against t on the FIRST interval of the
record that satisfies, in this order,
  (1) SIGNAL      a >= snr_amp * floor_eff, floor_eff = max(measured
                  out-of-band amplitude, floor_abs = 1e-7, the single-precision
                  floor of this build), with snr_amp = 4, i.e. 16x the floor in
                  power;
  (2) LINEARITY   a <= sat_frac * sat_level and, when a record-level mask is
                  supplied, max_x|m_y|(t) < m_linear: a saturating or reversing
                  record is not measuring a linear rate;
  (3) EARLINESS   the interval ENDS AT OR BEFORE the first turn-over of ln|a|,
                  a turn-over being a change of slope that both cuts the
                  two-segment residual to <= turn_sse_frac = 0.5 of the
                  single-line residual and makes the two branches diverge by
                  >= turn_efold = 1.0 e-foldings over the later branch;
  (4) ADEQUACY    the interval holds >= min_points = 5 samples, spans
                  >= min_efold = 1.0 e-foldings, and the fit on it reaches
                  r2 >= min_r2 = 0.9.
The sign of the measurand is not fixed. A point below threshold decays from the
start, and that decay IS the early linear rate: a MEASURED negative growth rate.
What is never the measurand is the slope of a LATE branch that follows a
turn-over, however well it fits; that branch is reported separately, under its
own name, and is never substituted for the early one.
If no interval satisfies (1)-(4) there is NO NUMBER. The result is HELD when the
record simply cannot be read, and NOT_SUMMARISABLE when the record turns over
and only the late branch could have been read -- the case a single growth rate
misrepresents. In both states gamma is nan, and a point in either state must not
enter a threshold bracket as a negative-growth point: "not measurable" is not
"measured negative".
```

### 9.1 Where the measurand may come from

`growth_interval.fit_growth_interval` (one series) and
`growth_interval.gamma_map` (a k-resolved record) are the ONLY implementations.
`saw_analysis.growth_fit` is the raw slope of whatever interval it is handed and
is refused when called from `revision_check/runs/`: a second fitter in a campaign
script is a second measurand, which is how the dead-bin defect survived in `G4b`
and `G9` after `G4` had been fixed.

### 9.2 The threshold bracket — what counts as a bracketing point

Fixed here, in advance:

1. A sweep brackets a threshold only if it contains at least one **measured**
   negative rate and at least one **measured** positive rate, and the largest
   negative-rate strain lies below the smallest positive-rate strain.
2. **"Not measurable" is not "measured negative".** A point whose early linear
   interval cannot be read (`HELD` or `NOT_SUMMARISABLE`) makes the sweep
   `NOT_EVALUABLE`. It is never entered as a negative-growth point and never
   silently dropped from the comparison.
3. A point whose only readable branch is the LATE one (after a turn-over) is not
   a bracketing point of either sign.
4. Bracket points are constructed from fit results (`point_from_fit`), not from
   bare numbers, so every point carries its status and its branch.
5. The gate compares the analysed runs against the **declared** design —
   `G4.RNG_SEEDS` and `ARMS[arm]["eps"]` — not against the runs that happen to be
   present. A missing seed or strain is `NOT_DETERMINABLE`, not a satisfied
   clause, and an unfinished run (`done = False`) supplies no gate point.

### 9.3 The run plan participates

A growth rate is only claimed from a run whose plan was runnable. `plan_runtime`
returns a plan whose status is `ok` or `floor_clipped_to_linear_window` (runnable)
or `INSUFFICIENT_LINEAR_WINDOW` or `NO_LINEAR_WINDOW` (blocking). On a blocking
status the plan has **no readable run length**: asking for `t_run` raises, so a
caller cannot obtain a number to run with by ignoring the status.
