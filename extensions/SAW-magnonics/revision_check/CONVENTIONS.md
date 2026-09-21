# CONVENTIONS — SAW-magnonics revision check

One page. Everything in `revision_check/` obeys this; every later step imports
`saw_analysis.py` and uses nothing else. Verified by `test_saw_analysis.py`
(all controls pass; full log in `TEST_RESULTS.txt`).

## 1. Sign of k and f

Derived from the code that **generated** the data, not from a plotting script:

| step | file | fact |
|---|---|---|
| kernel | `src/physics/chiralsawfield.cu` L69 | `phase = sawK*coord - sawOmega*t + sawPhi` |
| wrapper | `extension/.../src/saw_chiral.py` L147 | `saw_wavevector = +k` when `K_mr >= 0` |
| run | `extension/.../src/sim40_eps_kresolved.py` | `KMR = +1.0e6`, `direction='x'` |

So the pump is `eps_xx = eps_0 sin(q x - omega_p t)` with **q > 0**: it travels
toward +x. The fixed transform is therefore

```
M(k, f) = sum_x sum_t  m(x,t) w_t(t) w_x(x) exp(-i k x) exp(+i 2 pi f t)
```

implemented as `M = N_t * ifft_t( fft_x( m*w ) )`. A wave `exp(i(k0 x - w0 t))`
with `k0 > 0, f0 > 0` lands at **(+k0, +f0)**, and the sim40 pump sits at
`(+q, +f_SAW)` with **no sign flip anywhere downstream**.

**`make_fig5_v3.py` is correct.** It builds `F` with `np.fft.fft2`, whose kernel
is `exp(-ikx - i2 pi f t)`, slices at positive `f`, and plots `k_disp = -k_fft`.
For real `m`, `M_here(k,+f) = conj(F_fft2(-k,+f))`, so `|M_here(k,+f)| =
|F_fft2(-k,+f)|`: the negation restores the physical `k`. Verified to
3.8e-16 relative (`test_fft2_equivalence`). `analyze_sim40.py::fft_kf` applies
the same flip and is also correct **on this point**.

## 2. Amplitude vs power

`spectrum()` returns the **complex amplitude** `M`. Power is `|M|^2` and is
obtained only through `power()`. Every consumer takes an explicit
`quantity="power"` or `"amplitude"`; an unrecognised value raises rather than
defaulting. The two are never summed together or compared.

## 3. Units

`k` in rad/m; `to_inv_um(k) = k/1e6` gives the manuscript's "um^-1" (strictly
rad/um). `f` in Hz, `omega` in rad/s. The sim40 grid: `N_x = 1024`,
`dx = 5 nm`, `L = 5.12 um`, `dk = 2*pi/L = 1.2271846 um^-1`.

## 4. Band bookkeeping — disjoint by construction

`band_content()` partitions one f-slice into six categories assigned in a fixed
priority order (`k0`, `plus`, `minus`, `saw_plus`, `saw_minus`, `rest`), each
later mask with the already-claimed bins removed. The six fractions sum to
exactly 1, no bin is ever counted twice, and the number of bins each category
lost to a collision is returned in `stolen` so the collision is visible.

This replaces `analyze_sim40.py::I_pair`, whose `+k_pair` and `-k_pair` windows
**overlap**: with `dk_window = 16 dk = 19.635 um^-1` and
`k_pair = 5.386 um^-1` they are `(-14.25, +25.02)` and `(-25.02, +14.25)`
um^-1, so every bin with `|k| < 14.25 um^-1` — **including k = 0** — enters
twice. On a spatially uniform k = 0 oscillation the legacy routine returns
exactly **2.000000000000 x** the slice total; `band_content` returns
`k0 = 1.000000000000` and finite-k content `0`.

## 5. Pump-phase sign in the pair correlator

```
C(k) = | < M_s(k) M_s(K_pump-k) exp(+i omega_p t_s) >_s |
       / sqrt( <|M_s(k)|^2> <|M_s(K_pump-k)|^2> )
```

In **this** convention a mode contributes `M_s = A exp(-i 2 pi f t_s)`, the pair
product carries `exp(-i omega_p t_s)`, and the cancelling factor is
`exp(+i omega_p t_s)`. Control (a) gives `C = 1.000000000000`; the opposite
sign gives `0.2357022604`, matching the analytic residual
`|<exp(-2 i omega_p t_s)>|` to 1e-9.

`sim39_normalized_correlator.py` builds its spectra with `np.fft.fft2`, whose
segment factor is the **conjugate**, `exp(+i 2 pi f t_s)`. Its product already
carries `exp(+i omega_p t_s)`; multiplying by `exp(+i omega_p t_s)` again
**doubles** the rotation instead of removing it. The published formula is right
for the convention here and wrong for the one it was used with. Do not copy it.

Segmentation caveat: if `f_pump * stride * dt` is an integer, `exp(i omega_p
t_s) = 1` for every segment and the sign is untestable. The control suite uses
`stride = 67` against `seg_len = 64` on purpose. The pump frequency is known
independently (6.000 GHz), so use the exact value, never the segment bin centre.

## 6. Null hypothesis

`pair_correlator_null()` keeps the **measured per-segment, per-k magnitudes**
and randomises only the phases, returning the null **distribution** with
quantiles. For the control data, `M = 6` segments give `q50 = 0.348`,
`q95 = 0.688`, `q99 = 0.826`, while the single line `1/sqrt(M) = 0.408` sits at
the **61st percentile** — it is not a 95% threshold and must not be drawn as
one. 197/200 random-phase realisations land inside the null's [0.5%, 99%]
interval.

## 7. Frequency resolution the sim40 record can support

Measured from `data/sim40_checkpoints/sim40_full_2fK_eps7e-05.npz`:
`my_xt` is `(751, 1024)`, `DT_REC = 20.0 ps`, so the legacy second half
`trange = (375, 751)` is **376 samples over T = 7.520 ns**, giving
`df_bin = 132.97872 MHz`.

| window | ENBW | −3 dB width | first zero |
|---|---|---|---|
| rectangular | 132.98 MHz | 117.62 MHz | 132.98 MHz |
| Hann | 199.47 MHz | **191.25 MHz** | 265.96 MHz |

A growth envelope `exp(Gamma t)` adds a HWHM of `Gamma/(2 pi)`. Measured for
this run (sliding 64-sample windows, `ln|bin amplitude|` fitted over 2–7 ns):
`Gamma(bin 4) = 3.057e8 1/s` (r² 0.994), `Gamma(bin 5) = 2.992e8 1/s`
(r² 0.995) — an envelope HWHM of 48.7 MHz, 0.37 bin, giving an effective line
width near **215 MHz**. This sizes the linewidth only; it says nothing about how
many modes the bins contain.

**Usable frequency resolution of this record is a few hundred MHz, not
133 MHz.** Welch segmentation makes it far worse: 6 segments of 62 samples give
806.5 MHz per bin, and 3 GHz is not on a bin.

**Added 2026-09-18 (independent audit, section 4).** The line width is not an
error bar for a peak ESTIMATOR either, and neither is the 133 MHz bin. A
parabolic-interpolated peak can locate a line to far better than its width, and
a biased estimator on a short record with a growth envelope can do considerably
worse; which of the two applies to
`model_comparison.project_and_estimate_frequency` has not been measured. Any
`omega_1 + omega_2 = omega_p` statement therefore needs that estimator's bias
and spread, measured on synthetic controls carrying the same record length,
growth envelope and window. No such budget exists in this directory. Until it
does, neither the pre-fix `+69.8 MHz` nor the corrected `-5.8 MHz` bin-4 + bin-5
sum-frequency residual may be used as evidence in either direction
(`model_comparison.md` section 10, `NUMBERS_FOR_MANUSCRIPT.md` section 4.4).

Two time-frequency conventions in one codebase is how the branch error of
section 4 of that audit arose. There is now exactly one: `spectrum` along the
time axis, and `time_spectrum` for a single projected series. Neither analysis
code nor campaign code should call `np.fft.fft` over time again.

## 8. What the module deliberately does NOT claim

Established by direct construction in `test_saw_analysis.py` and tabulated in
`DISCRIMINATION_TABLE.md`:

- **Zero padding cannot count modes.** Two components at exactly bins 4 and 5,
  amplitudes `+sqrt(0.318)` and `-sqrt(0.433)`, give under 16x padding **one**
  dominant maximum at **5.599030 um^-1**, side maxima at 0.82% and 0.25% —
  reproduced here exactly as stated in the brief.
- **Fixed inter-bin phase, proportional amplitudes and a common growth rate are
  non-discriminating.** One off-grid wave and two phase-locked components both
  show them. They can refute *independent* modes; they cannot decide leakage
  vs. locked pair.
- **The pair correlator is non-discriminating too.** It returns `C = 1.000000`
  for both. A high `C` is evidence against independent thermal populations, not
  evidence for two components.
- The only diagnostic in this module that separates the two is the **leakage
  tail**: a single-off-grid-sinusoid fit to the whole positive-k slice gives a
  residual of 1.66e-6 for one wave and 0.154 for two components. Its caveats
  (contamination, and that any spatial window destroys it) are in the table.

## 9. Two numbers that must not be repeated

1. Bins 4 + 5 sum to **11.044662 um^-1**, which is the grid's 9th harmonic
   (`9 dk`). The physical pump is `q = 2 pi * 6 GHz / 3500 m/s = 10.771175
   um^-1`. They differ by 0.273487 um^-1 = **2.54%**, and `q L / 2 pi = 8.7771`:
   the box is incommensurate, so the pump has **no exact bin**. Never call the
   bin sum "bin-exact agreement with the pump".
2. The quoted weights **31.8% / 43.3%** are not reproduced by the fixed
   pipeline under any of twelve window x time-range x denominator combinations
   (`CANDIDATE_FRACTIONS.txt`; closest is rect / second half / `k>0`:
   0.2994 / 0.4075). The **ratio** 1.3616 is reproduced to ~0.1% by the
   second-half slice, so the relative weight is robust; the absolute pair is
   not, because the denominator was never stated. Re-derive it with
   `band_content()`, which always reports its own total. File mtimes and
   hashes cannot establish which source produced these data; provenance for
   existing runs is **NOT DETERMINABLE FROM FILES** and must be fixed from the
   next run onward.

## 10. Register

`established` / `consistent with` / `indistinguishable with current data` /
`refuted` are different words. For the bin-4/bin-5 feature at
`eps_0 = 7e-5`, nothing in step 1 moves it off **strong candidate, not
established**, and no diagnostic validated here can move it by itself.

## Files

- `saw_analysis.py` — the module. Import it; do not re-implement.
- `test_saw_analysis.py` — controls (a)–(f) plus convention tests. Run it.
- `TEST_RESULTS.txt` — full log of the last run.
- `DISCRIMINATION_TABLE.md` — which diagnostics separate which hypotheses.
- `check_sim40_candidate_fractions.py`, `CANDIDATE_FRACTIONS.txt` — the 31.8/43.3
  reproducibility check.
