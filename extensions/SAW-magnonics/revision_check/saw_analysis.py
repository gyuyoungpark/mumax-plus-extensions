"""saw_analysis -- ONE analysis module for the SAW-magnonics revision check.

Every later verification step must import this module.  Nothing here reads or
writes anything outside `revision_check/`; all inputs are arrays passed in by
the caller.

================================================================================
THE CONVENTION  (derived from the code that GENERATED the data, not from plots)
================================================================================

Chain of derivation
-------------------
1. Production kernel `D:/mumax-plus-dev/src/physics/chiralsawfield.cu`, line ~69:

       const real phase = sawK * coord - sawOmega * sawTime + sawPhi;

   so every SAW drive channel is a function of  (k x - omega t).

2. `src/saw_chiral.py` line 147:

       magnet.saw_wavevector = self.k if self.K_mr >= 0 else -self.k

   and `src/saw.py` line 94:  self.k = 2*pi/wavelength  > 0.
   sim40 runs with K_mr = +1.0e6 (see `src/sim40_eps_kresolved.py`), therefore
   sawK = +q with q = 2*pi*f_SAW/v_SAW > 0.

3. Hence the pump is a wave travelling toward +x:
       eps_xx(x,t) = eps_0 sin(q x - omega_p t),   q > 0, omega_p > 0.

Fixed transform
---------------
This module ALWAYS uses the "physical" forward transform

       M(k, f) = sum_x sum_t  m(x,t) w_t(t) w_x(x) exp(-i k x) exp(+i 2 pi f t)

so that a wave  m ~ exp(+i(k0 x - 2 pi f0 t))  with k0 > 0, f0 > 0 appears at
(k = +k0, f = +f0).  With this convention the sim40 pump sits at
(k = +q, f = +f_SAW) with NO sign flip anywhere downstream.

Implementation:  M = N_t_padded * ifft_t( fft_x( m*w ) ).
`np.fft.fft` along x supplies exp(-i k x); `np.fft.ifft` along t supplies
exp(+i 2 pi f t), and the factor N_t restores the plain (unnormalised) sum, so
|M| is independent of zero padding of the *time* axis.

Relation to the legacy `np.fft.fft2` pipeline
---------------------------------------------
fft2 uses F(k,f) = sum m exp(-i k x) exp(-i 2 pi f t).  Therefore

       M_here(k, f) = F_fft2(k, -f) = conj( F_fft2(-k, +f) )      (m real)

so   |M_here(k, +f)|  =  |F_fft2(-k, +f)| .

=> `src/make_fig5_v3.py`, which builds F with `np.fft.fft2`, slices at positive
   f and then plots  k_disp = -k_fft, IS PLOTTING THE PHYSICAL k CORRECTLY.
   The negation there is a correct patch for an fft2-based pipeline on a
   positive-frequency slice, not an error.  `src/analyze_sim40.py` applies the
   same flip, also correctly.  This module removes the need for the patch by
   fixing the transform instead; test_saw_analysis.py::test_fft2_equivalence
   verifies the identity above numerically.

   (What IS wrong in analyze_sim40.py is `I_pair`: its +/-k_pair windows overlap
   -- see `band_content` below.  What IS wrong in sim39_normalized_correlator.py
   is the sign of the pump-phase removal -- see `pair_correlator` below.)

AMPLITUDE vs POWER
------------------
`spectrum` returns the COMPLEX amplitude M.  Power is |M|^2 and is obtained
only through `power()`.  Every routine that consumes a spectrum takes an
explicit `quantity` argument, either "power" (|M|^2) or "amplitude" (|M|).
There is no default that silently mixes them.

UNITS
-----
k is returned in rad/m.  `to_inv_um(k)` divides by 1e6; the manuscript writes
that unit as "um^-1" although it is strictly rad/um.  f is in Hz.
"""

from __future__ import annotations

import os
import sys

import numpy as np

__all__ = [
    "K_SIGN_DOC", "to_inv_um", "spectrum", "time_spectrum", "power",
    "amplitude",
    "f_slice", "band_content", "segment_spectra", "pair_correlator",
    "pair_correlator_null", "resolution_report", "growth_fit",
    "grid_k_axis", "pump_wavevector",
]

K_SIGN_DOC = (
    "M(k,f) = sum_x,t m w exp(-i k x + i 2 pi f t); a wave exp(i(k0 x - w0 t)) "
    "with k0>0, f0>0 appears at (+k0, +f0). sim40 pump sits at (+q, +f_SAW)."
)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def to_inv_um(k):
    """rad/m -> the manuscript's 'um^-1' (strictly rad/um)."""
    return np.asarray(k) / 1e6


def grid_k_axis(nx, dx, pad_x=1, shift=True):
    """k axis (rad/m) of the transform used by `spectrum`."""
    n = int(nx * pad_x)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return np.fft.fftshift(k) if shift else k


def pump_wavevector(f_saw, v_saw=3500.0):
    """q = 2 pi f_SAW / v_SAW  (rad/m), positive: pump travels toward +x."""
    return 2 * np.pi * f_saw / v_saw


def _window(n, kind):
    kind = (kind or "rect").lower()
    if kind in ("rect", "none", "boxcar"):
        return np.ones(n)
    if kind == "hann":
        # symmetric Hann, matching the np.hanning used by the legacy code
        return np.hanning(n)
    raise ValueError("unknown window %r" % (kind,))


# ----------------------------------------------------------------------------
# 1. spectrum
# ----------------------------------------------------------------------------

def spectrum(m_xt, dx, dt, window="hann", trange=None, window_x="rect",
             pad_t=1, pad_x=1, detrend=True, shift=True):
    """(k, f)-spectrum of m(x,t) in the fixed convention (see module docstring).

    Parameters
    ----------
    m_xt : (n_t, n_x) real array.  Axis 0 is time, axis 1 is space.
    dx, dt : sample spacings (m, s).
    window : time window, 'hann' (default) or 'rect'.
    trange : (i0, i1) slice of the time axis, or None for the whole record.
             Pass (n_t//2, n_t) to reproduce the legacy "second half".
    window_x : spatial window, 'rect' (default; the box is periodic under PBC)
             or 'hann'.  Only the leakage controls use 'hann'.
    pad_t, pad_x : integer zero-padding factors.  Padding INTERPOLATES the
             spectrum; it adds no resolution and cannot count modes.
    detrend : subtract the mean of the analysed block (removes the DC offset
             that would otherwise dominate the k=0, f=0 bin).
    shift : return fftshift-ed, monotonically sorted axes (default True).

    Returns
    -------
    k_axis : (n_x*pad_x,) rad/m
    f_axis : (n_t*pad_t,) Hz
    M      : (n_f, n_k) complex amplitude (NOT power)
    """
    m = np.asarray(m_xt, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("m_xt must be 2-D (n_t, n_x)")
    if trange is not None:
        i0, i1 = trange
        m = m[i0:i1, :]
    nt, nx = m.shape
    if detrend:
        m = m - m.mean()

    wt = _window(nt, window)[:, None]
    wx = _window(nx, window_x)[None, :]
    mw = m * wt * wx

    nxp = int(nx * pad_x)
    ntp = int(nt * pad_t)

    X = np.fft.fft(mw, n=nxp, axis=1)          # exp(-i k x)
    M = np.fft.ifft(X, n=ntp, axis=0) * ntp    # exp(+i 2 pi f t), unnormalised

    k_axis = 2 * np.pi * np.fft.fftfreq(nxp, d=dx)
    f_axis = np.fft.fftfreq(ntp, d=dt)
    if shift:
        k_axis = np.fft.fftshift(k_axis)
        f_axis = np.fft.fftshift(f_axis)
        M = np.fft.fftshift(M, axes=(0, 1))
    return k_axis, f_axis, M


def time_spectrum(s, dt, window="hann", pad_t=1, shift=False):
    """Time transform of ONE time series, in THIS module's time-axis sign.

        S(f) = sum_t s(t) w(t) exp(+i 2 pi f t)

    i.e. exactly the kernel `spectrum` uses along axis 0, so a series
    exp(-i 2 pi f0 t) -- what a +x-travelling wave leaves after projection on
    exp(-i k x) -- peaks at f = +f0.

    This helper exists so that no caller writes its own fft and picks its own
    sign.  `np.fft.fft` supplies exp(-i 2 pi f t) and would place that same
    series on the NEGATIVE frequencies; a positive-frequency search over such a
    transform returns the COUNTER-PROPAGATING component.  That is the defect
    found in `model_comparison.project_and_estimate_frequency` by the
    independent audit of 2026-09-18 (section 4), pinned by
    `test_saw_analysis.py::test_freq_estimator_branch`.

    Scaling matches `spectrum`: the returned S is the plain unnormalised sum,
    so |S| is independent of zero padding of the time axis.

    Returns (f_axis, S); axes are NOT fftshift-ed unless shift=True.
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("time_spectrum takes a 1-D time series")
    n = s.shape[0]
    w = _window(n, window)
    ntp = int(n * pad_t)
    S = np.fft.ifft(s * w, n=ntp) * ntp
    f = np.fft.fftfreq(ntp, d=dt)
    if shift:
        f, S = np.fft.fftshift(f), np.fft.fftshift(S)
    return f, S


def power(M):
    """|M|^2.  The ONLY way to get power in this module."""
    return np.abs(M) ** 2


def amplitude(M):
    """|M|."""
    return np.abs(M)


def f_slice(M, f_axis, f0):
    """(index, M[index, :]) for the f bin nearest f0.  No interpolation."""
    i = int(np.argmin(np.abs(f_axis - f0)))
    return i, M[i, :]


# ----------------------------------------------------------------------------
# 3. band_content -- strictly disjoint masks
# ----------------------------------------------------------------------------

def band_content(k_axis, M_slice, k_target, k_saw, halfwidth,
                 quantity="power", k0_halfwidth=None, saw_halfwidth=None):
    """Partition one f-slice into DISJOINT k bands and report fractions.

    Categories, assigned in this priority order so that no bin is ever counted
    twice:
        'k0'        |k| <= k0_halfwidth   (default: half a bin -> the k=0 bin alone)
        'plus'      |k - k_target| <= halfwidth
        'minus'     |k + k_target| <= halfwidth
        'saw_plus'  |k - k_saw|    <= saw_halfwidth   (default: halfwidth)
        'saw_minus' |k + k_saw|    <= saw_halfwidth
        'rest'      everything else
    A bin claimed by an earlier category is REMOVED from all later ones; the
    number of bins each category lost is returned in 'stolen' so the collision
    is visible rather than silent.  The six fractions sum to exactly 1.

    This is the routine that replaces `src/analyze_sim40.py::I_pair`, which
    summed a +k_pair window and a -k_pair window that OVERLAP whenever
    halfwidth > k_target (there halfwidth = 16*dk = 19.63 um^-1 against
    k_target = 5.386 um^-1), so every bin with |k| < 14.2 um^-1 -- including
    k = 0 -- was added twice.

    Parameters
    ----------
    k_axis : (n_k,) rad/m, as returned by `spectrum` (sorted if shift=True).
    M_slice : (n_k,) complex spectrum at one frequency.
    k_target, k_saw, halfwidth : rad/m.
    quantity : 'power' (|M|^2) or 'amplitude' (|M|).  No default mixing.

    Returns
    -------
    dict with 'fractions', 'values', 'masks', 'n_bins', 'stolen', 'total',
    'quantity', 'disjoint' (always True), 'sum_check'.
    """
    k = np.asarray(k_axis, dtype=np.float64)
    if quantity == "power":
        w = power(M_slice)
    elif quantity == "amplitude":
        w = amplitude(M_slice)
    else:
        raise ValueError("quantity must be 'power' or 'amplitude'")
    w = np.asarray(w, dtype=np.float64)
    if w.shape != k.shape:
        raise ValueError("k_axis and M_slice shapes differ")

    dk = float(np.min(np.abs(np.diff(np.sort(k)))))
    if k0_halfwidth is None:
        k0_halfwidth = 0.5 * dk
    if saw_halfwidth is None:
        saw_halfwidth = halfwidth

    raw = {
        "k0":        np.abs(k) <= k0_halfwidth,
        "plus":      np.abs(k - k_target) <= halfwidth,
        "minus":     np.abs(k + k_target) <= halfwidth,
        "saw_plus":  np.abs(k - k_saw) <= saw_halfwidth,
        "saw_minus": np.abs(k + k_saw) <= saw_halfwidth,
    }
    order = ["k0", "plus", "minus", "saw_plus", "saw_minus"]

    taken = np.zeros_like(k, dtype=bool)
    masks, stolen = {}, {}
    for name in order:
        m = raw[name] & ~taken
        stolen[name] = int(raw[name].sum() - m.sum())
        masks[name] = m
        taken |= m
    masks["rest"] = ~taken
    stolen["rest"] = 0

    total = float(w.sum())
    values = {n: float(w[masks[n]].sum()) for n in masks}
    fractions = {n: (values[n] / total if total > 0 else 0.0) for n in values}

    stack = np.array([masks[n] for n in masks])
    assert stack.sum(axis=0).max() <= 1, "masks overlap"
    assert stack.any(axis=0).all(), "masks do not cover the axis"

    return {
        "quantity": quantity,
        "total": total,
        "values": values,
        "fractions": fractions,
        "masks": masks,
        "n_bins": {n: int(masks[n].sum()) for n in masks},
        "stolen": stolen,
        "disjoint": True,
        "sum_check": float(sum(fractions.values())),
        "dk": dk,
        "halfwidth": float(halfwidth),
        "k0_halfwidth": float(k0_halfwidth),
    }


# ----------------------------------------------------------------------------
# 4. segmented spectra + pair correlator with the CORRECT pump-phase sign
# ----------------------------------------------------------------------------

def segment_spectra(m_xt, dx, dt, seg_len, n_seg, stride=None, t0=0,
                    window="hann", window_x="rect", pad_t=1, pad_x=1,
                    detrend=True):
    """Welch-style ensemble of (k,f) spectra in the fixed convention.

    stride : samples between segment starts (default seg_len, i.e. contiguous
             non-overlapping).  NOTE: when stride*dt*f_pump is an integer the
             pump-phase factor exp(i omega_p t_s) equals 1 for every segment
             and the pump-phase removal becomes untestable; the control suite
             uses a stride that avoids that degeneracy on purpose.

    Returns k_axis, f_axis, M_segs (n_seg, n_f, n_k), t_starts (s).
    """
    m = np.asarray(m_xt, dtype=np.float64)
    if stride is None:
        stride = seg_len
    need = t0 + (n_seg - 1) * stride + seg_len
    if need > m.shape[0]:
        raise ValueError("record too short: need %d, have %d"
                         % (need, m.shape[0]))
    Ms, t_starts = [], []
    k_axis = f_axis = None
    for s in range(n_seg):
        i0 = t0 + s * stride
        k_axis, f_axis, M = spectrum(m, dx, dt, window=window,
                                     trange=(i0, i0 + seg_len),
                                     window_x=window_x, pad_t=pad_t,
                                     pad_x=pad_x, detrend=detrend)
        Ms.append(M)
        t_starts.append(i0 * dt)
    return k_axis, f_axis, np.array(Ms), np.array(t_starts)


def _partner_index(k_axis, K_pump):
    """For every j return the index j2 with k[j2] ~= K_pump - k[j], a validity
    mask, and the snap error in rad/m."""
    k = np.asarray(k_axis, dtype=np.float64)
    dk = k[1] - k[0]
    j2 = np.rint((K_pump - k - k[0]) / dk).astype(int)
    ok = (j2 >= 0) & (j2 < k.size)
    j2c = np.clip(j2, 0, k.size - 1)
    snap = np.full(k.size, np.nan)
    snap[ok] = (K_pump - k[ok]) - k[j2c[ok]]
    return j2c, ok, snap


def pair_correlator(M_slices, t_starts, k_axis, K_pump, omega_pump,
                    phase_sign=+1):
    """Normalised pair coherence C(k) with the pump phase removed correctly.

        C(k) = | < M_s(k) M_s(K_pump-k) exp(i sgn omega_p t_s) >_s |
               / sqrt( <|M_s(k)|^2> <|M_s(K_pump-k)|^2> )

    SIGN DERIVATION (this module's convention).  A mode with complex amplitude
    A at (k_i, f_i) contributes to the segment starting at t_s

        M_s(k_i) = A exp(-i 2 pi f_i t_s) * (intra-segment shape factor)

    because the transform carries exp(+i 2 pi f tau) over the intra-segment
    time tau only.  For a pair with f_1 + f_2 = f_p the product therefore
    carries exp(-i omega_p t_s), and the correction that cancels it is
    exp(+i omega_p t_s):  phase_sign = +1.

    `src/sim39_normalized_correlator.py` builds its spectra with np.fft.fft2,
    whose segment factor is the complex CONJUGATE, exp(+i 2 pi f_i t_s); the
    product then already carries exp(+i omega_p t_s), and multiplying by
    exp(+i omega_p t_s) again DOUBLES the rotation instead of removing it.
    The published formula is right for this module's convention and wrong for
    the fft2 convention it was actually used with.

    phase_sign=-1 exists ONLY so the control suite can show the wrong sign
    fails.

    Returns dict: k_axis, C, pair (complex mean), norm1, norm2, j2, valid,
    snap_error_max, n_seg, phase_sign.
    """
    M = np.asarray(M_slices)
    if M.ndim != 2:
        raise ValueError("M_slices must be (n_seg, n_k)")
    n_seg, nk = M.shape
    t_s = np.asarray(t_starts, dtype=np.float64)
    if t_s.size != n_seg:
        raise ValueError("t_starts length mismatch")

    j2, ok, snap = _partner_index(k_axis, K_pump)
    rot = np.exp(1j * phase_sign * omega_pump * t_s)[:, None]

    prod = M * M[:, j2] * rot
    pair = prod.mean(axis=0)
    norm1 = (np.abs(M) ** 2).mean(axis=0)
    norm2 = (np.abs(M[:, j2]) ** 2).mean(axis=0)

    denom = np.sqrt(norm1 * norm2)
    C = np.where(denom > 0, np.abs(pair) / np.where(denom > 0, denom, 1.0), 0.0)
    C = np.where(ok, C, np.nan)
    return {
        "k_axis": np.asarray(k_axis), "C": C, "pair": pair,
        "norm1": norm1, "norm2": norm2, "j2": j2, "valid": ok,
        "snap_error_max": float(np.nanmax(np.abs(snap))) if ok.any() else np.nan,
        "n_seg": n_seg, "phase_sign": phase_sign,
    }


# ----------------------------------------------------------------------------
# 5. amplitude-preserving random-phase null
# ----------------------------------------------------------------------------

def pair_correlator_null(M_slices, t_starts, k_axis, K_pump, omega_pump,
                         n_real=2000, k_indices=None, seed=0, phase_sign=+1):
    """Null DISTRIBUTION of C(k) with segment phases randomised.

    The measured per-segment, per-k MAGNITUDES |M_s(k)| are kept exactly; only
    the phases are replaced by independent uniform draws.  This destroys any
    inter-segment phase rigidity and any k <-> K-k phase lock while leaving the
    normalisation <|M|^2> untouched, so the returned spread is the honest null
    for THIS dataset's amplitude structure -- not the single 1/sqrt(M) line
    that `sim39_normalized_correlator.py` draws, which assumes equal-magnitude
    segments and unit-modulus random phasors.

    Returns dict: 'k_indices', 'C_null' (n_real, len(k_indices)), 'quantiles'
    (q50/q90/q95/q99 per k), 'mean', 'std', 'analytic_1_over_sqrtM', 'valid'.
    """
    M = np.asarray(M_slices)
    n_seg, nk = M.shape
    if k_indices is None:
        k_indices = np.arange(nk)
    k_indices = np.asarray(k_indices, dtype=int)
    j2, ok, _ = _partner_index(k_axis, K_pump)

    A1 = np.abs(M[:, k_indices])              # (n_seg, nsel)
    A2 = np.abs(M[:, j2[k_indices]])
    rot = np.exp(1j * phase_sign * omega_pump * np.asarray(t_starts))[:, None]

    rng = np.random.default_rng(seed)
    out = np.empty((n_real, k_indices.size))
    norm = np.sqrt((A1 ** 2).mean(axis=0) * (A2 ** 2).mean(axis=0))
    norm = np.where(norm > 0, norm, 1.0)
    for r in range(n_real):
        p1 = np.exp(2j * np.pi * rng.random(A1.shape))
        p2 = np.exp(2j * np.pi * rng.random(A2.shape))
        prod = (A1 * p1) * (A2 * p2) * rot
        out[r] = np.abs(prod.mean(axis=0)) / norm

    qs = np.quantile(out, [0.5, 0.9, 0.95, 0.99], axis=0)
    return {
        "k_indices": k_indices, "C_null": out,
        "quantiles": {"q50": qs[0], "q90": qs[1], "q95": qs[2], "q99": qs[3]},
        "mean": out.mean(axis=0), "std": out.std(axis=0),
        "analytic_1_over_sqrtM": 1.0 / np.sqrt(n_seg),
        "valid": ok[k_indices],
    }


# ----------------------------------------------------------------------------
# resolution bookkeeping
# ----------------------------------------------------------------------------

def resolution_report(n_samples, dt, window="hann", growth_rate=None):
    """What frequency resolution a record of n_samples at dt can support.

    growth_rate : Gamma (1/s) of an exp(Gamma t) envelope, or None.
    Hann factors: noise-equivalent bandwidth 1.5 bins, -3 dB main-lobe width
    1.4382 bins, first zero at 2 bins.  Rectangular: 1.0 / 0.8845 / 1.0.
    """
    T = n_samples * dt
    df = 1.0 / T
    fac = {"rect": (1.0, 0.8845, 1.0), "hann": (1.5, 1.4382, 2.0)}[window]
    out = {
        "n_samples": int(n_samples), "dt": float(dt), "T": float(T),
        "df_bin": df, "window": window,
        "enbw_Hz": fac[0] * df, "fwhm_3dB_Hz": fac[1] * df,
        "first_zero_Hz": fac[2] * df,
    }
    if growth_rate is not None:
        out["growth_rate"] = float(growth_rate)
        out["growth_HWHM_Hz"] = float(growth_rate) / (2 * np.pi)
        out["growth_in_bins"] = out["growth_HWHM_Hz"] / df
        out["effective_width_Hz"] = float(
            np.hypot(out["fwhm_3dB_Hz"] / 2, out["growth_HWHM_Hz"]) * 2)
    return out


class CampaignFitterBypass(RuntimeError):
    """A campaign script asked for a growth rate from the raw fit."""


#: The campaign's growth rates have ONE definition and ONE implementation:
#: growth_interval.MEASURAND, read by growth_interval.fit_growth_interval and
#: growth_interval.gamma_map.  `growth_fit` below is the raw slope of whatever
#: interval it is handed -- no signal criterion, no linearity cut, no turn-over
#: cut, no status -- and calling it from a campaign script is how the dead-bin
#: defect survived in G4b and G9 after G4 had been fixed (audit 2026-09-18,
#: items 2 and 7).  It is therefore refused when the caller is a module in
#: revision_check/runs/, which is the campaign.  Everything outside that
#: directory -- the published-record analyses, the adversarial tests that
#: reproduce the old behaviour on purpose, one-off exploration -- keeps working
#: unchanged, and a campaign script that genuinely wants the raw slope can say
#: allow_raw=True and be visible in a grep.
_CAMPAIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _refuse_campaign_caller(fname, depth=2, max_frames=40):
    """Refuse if ANY frame below this one is a campaign module.

    The whole stack, not just the immediate caller: a campaign script that
    reached the raw fitter through one helper would otherwise defeat the guard,
    and "one code path over" is the failure mode this round exists to remove.
    normcase, because on Windows the same directory reaches this function
    spelled several ways and a guard that misses on a spelling is not a guard.
    """
    want = os.path.normcase(os.path.abspath(_CAMPAIGN_DIR))
    caller = ""
    try:
        f = sys._getframe(depth)
        for _ in range(max_frames):
            if f is None:
                break
            path = os.path.abspath(f.f_code.co_filename)
            if os.path.normcase(os.path.dirname(path)) == want:
                caller = path
                break
            f = f.f_back
    except Exception:                                             # noqa: BLE001
        return
    if not caller:
        return
    raise CampaignFitterBypass(
        "%s is a campaign script (revision_check/runs/) and may not read a "
        "growth rate from saw_analysis.%s: that function fits whatever interval "
        "it is given, with no signal criterion, no linearity cut, no turn-over "
        "cut and no status, so it is a DIFFERENT measurand from the one the "
        "campaign reports. Use growth_interval.fit_growth_interval (one series) "
        "or growth_interval.gamma_map (a k-resolved record); both state their "
        "interval and can return HELD or NOT_SUMMARISABLE instead of a number. "
        "Pass allow_raw=True only to compute something that is explicitly not a "
        "campaign growth rate." % (os.path.basename(caller), fname))


def growth_fit(t, a, t0=None, t1=None, allow_raw=False):
    """Unweighted least-squares fit of ln|a| vs t -> (Gamma [1/s], b, r2).

    THIS IS NOT THE CAMPAIGN'S GROWTH RATE.  See `CampaignFitterBypass` above:
    the campaign measurand is `growth_interval.MEASURAND` and the only two
    implementations of it are in `growth_interval`.  Calls from
    revision_check/runs/ are refused unless `allow_raw=True`.
    """
    if not allow_raw:
        _refuse_campaign_caller("growth_fit")
    t = np.asarray(t, dtype=np.float64)
    a = np.abs(np.asarray(a, dtype=np.float64))
    m = np.isfinite(a) & (a > 0)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    y = np.log(a[m])
    x = t[m]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(coef[1]), r2
