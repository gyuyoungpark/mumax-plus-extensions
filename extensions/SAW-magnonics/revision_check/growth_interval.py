"""THE ONE FITTER.  What a growth rate in this campaign means, and how it is read.

WHY THIS MODULE EXISTS, AND WHY IT WAS REBUILT (2026-09-18, round 2)
  Round 1 of this module fixed the two defects the audit wrote down -- a
  whole-record fit over a saturated record, and a dead bin winning an argmax --
  and introduced a worse one.  It decided GROWTH-OR-DECAY from the OVERALL slope
  of the surviving samples and then fitted one side of the peak, so a record that
  grew at +3.0e8 1/s for 4 ns and then damped at -1.2e8 1/s came back as
  `gamma = -1.2000e+08, status = "ok", r2 = 1.0000`.  The pre-round fit returned
  -5.539e7 at r2 = 0.4076, which the gate's own r2 > 0.9 clause rejected: the
  round converted a number the guard caught into a confident wrong one, and the
  longer a mode was observed after turning over the more confidently its damping
  was reported as its growth rate.

  The class of defect is not "the turn-over cut picked the wrong side".  It is
  that the code never said WHAT QUANTITY it was measuring, so any interval with a
  good r2 could stand in for any other.  This round fixes that first: the
  measurand is stated once, in ONE string, and every criterion, every status and
  every refusal below follows from it.  `MEASURAND` is reproduced verbatim in
  PREREGISTRATION.md section 9 and a regression test compares the two texts
  byte-for-byte, so the specification, the code and the output cannot drift.

  The same statement is why there is only one entry point.  A second fitter
  somewhere else in the tree is not a second implementation of the same
  measurand, it is a second measurand -- and that is exactly how the dead-bin
  defect survived in G4b and G9 after G4 was fixed.  `fit_growth_interval` (one
  series) and `gamma_map` (a k-resolved record) are the only places a campaign
  growth rate is allowed to come from, and `saw_analysis.growth_fit` refuses to
  be called from the campaign scripts at all.

Usage
    import growth_interval as GI
    r = GI.fit_growth_interval(t, a, floor=measured_floor, sat_level=0.1)
    if r["status"] != GI.OK:
        ...            # NO RATE.  Do not substitute a number, and do not enter
                       # this point in a threshold bracket: see bracket_from_fits.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FP_FLOOR_SINGLE", "MEASURAND", "fit_growth_interval", "gamma_map",
           "segment_layout",
           "bracket_check", "bracket_from_fits", "point_from_fit",
           "HELD", "OK", "NOT_SUMMARISABLE", "NOT_MEASURABLE_STATES"]

# ---------------------------------------------------------------------------
# THE MEASURAND.  One statement, shared by the specification (PREREGISTRATION.md
# section 9), this module and every printed row.  Edit it in ONE place: any other
# copy is compared against this one by test_p2_measurand.py and must match
# byte-for-byte.
# ---------------------------------------------------------------------------
MEASURAND = """\
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
"""

# RUN_PLAN.md sec.0e: the installed engine is _mumaxpluscpp_single, so the
# floating-point floor of any recorded amplitude is ~1e-7, not ~1e-15.
FP_FLOOR_SINGLE = 1e-7

OK = "ok"
HELD = "HELD"
NOT_SUMMARISABLE = "NOT_SUMMARISABLE"
#: every status that is NOT a measured rate.  A caller that branches on
#: `status == OK` is right; a caller that branches on `status == HELD` is wrong,
#: because it silently accepts NOT_SUMMARISABLE.
NOT_MEASURABLE_STATES = (HELD, NOT_SUMMARISABLE)

# Criteria of clause (3).  Named here, printed in every result under
# `criteria`, and stated in MEASURAND above.
TURN_SSE_FRAC = 0.5
TURN_EFOLD = 1.0
MAX_TURNS = 3


# ---------------------------------------------------------------------------
# least squares on a half-open index range, from cumulative sums
# ---------------------------------------------------------------------------
class _Cum:
    """Prefix sums of (x, y, x^2, xy, y^2) so any range fit is O(1)."""

    def __init__(self, x, y):
        self.n = x.size
        # x is a time in seconds: the records start at ~1e-8 with
        # spacings of ~1e-11, so raw prefix sums lose digits to
        # cancellation.  Shifting by x[0] costs nothing, leaves every
        # slope unchanged, and keeps this fit numerically identical to a
        # direct least squares -- which matters because G9 compares the
        # two against a 1e-9 relative tolerance.
        self.x0 = float(x[0]) if x.size else 0.0
        x = np.asarray(x, dtype=np.float64) - self.x0
        z = np.zeros(1)
        self.x = np.concatenate([z, np.cumsum(x)])
        self.y = np.concatenate([z, np.cumsum(y)])
        self.xx = np.concatenate([z, np.cumsum(x * x)])
        self.xy = np.concatenate([z, np.cumsum(x * y)])
        self.yy = np.concatenate([z, np.cumsum(y * y)])

    def fit(self, i, j):
        """OLS of y on x over [i, j): (slope, intercept, sse, sst, n)."""
        n = j - i
        if n < 2:
            return np.nan, np.nan, np.nan, np.nan, n
        sx = self.x[j] - self.x[i]
        sy = self.y[j] - self.y[i]
        sxx = self.xx[j] - self.xx[i]
        sxy = self.xy[j] - self.xy[i]
        syy = self.yy[j] - self.yy[i]
        den = n * sxx - sx * sx
        if not np.isfinite(den) or den <= 0:
            return np.nan, np.nan, np.nan, np.nan, n
        slope = (n * sxy - sx * sy) / den
        icept = (sy - slope * sx) / n
        # sum of squared residuals, algebraically
        sse = (syy - 2 * slope * sxy - 2 * icept * sy + slope * slope * sxx
               + 2 * slope * icept * sx + n * icept * icept)
        sst = syy - sy * sy / n
        # icept is in shifted coordinates; report it at x = 0
        return float(slope), float(icept - slope * self.x0), \
            float(max(sse, 0.0)), float(max(sst, 0.0)), int(n)

    def r2(self, i, j):
        _, _, sse, sst, _ = self.fit(i, j)
        return 1.0 - sse / sst if sst > 0 else np.nan

    def stderr(self, i, j):
        slope, _, sse, _, n = self.fit(i, j)
        if n < 3 or not np.isfinite(sse):
            return float("nan")
        sx = self.x[j] - self.x[i]
        sxx = self.xx[j] - self.xx[i]
        sxxc = sxx - sx * sx / n
        if sxxc <= 0:
            return float("nan")
        return float(np.sqrt(sse / (n - 2) / sxxc))


def _ols_log(t, a):
    """Kept for callers that want the bare slope of a stated interval."""
    y = np.log(a)
    c = _Cum(np.asarray(t, float), y)
    slope, icept, sse, sst, _ = c.fit(0, y.size)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return float(slope), float(icept), float(r2)


def _runs(mask):
    """All maximal contiguous True runs as (i0, i1), i1 exclusive, in order."""
    out, i, n = [], 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _longest_run(mask):
    """(i0, i1) of the longest contiguous True run; (0, 0) if none."""
    rr = _runs(mask)
    return max(rr, key=lambda r: r[1] - r[0]) if rr else (0, 0)


# ---------------------------------------------------------------------------
# clause (3): the turn-over
# ---------------------------------------------------------------------------
def _first_turn(t, cum, i0, i1, min_points, turn_sse_frac, turn_efold):
    """The earliest accepted change of slope inside [i0, i1), or None.

    Symmetric by construction: growth -> decay, growth -> saturation plateau and
    decay -> growth are all changes of slope and all detected by the same two
    numbers.  Nothing here knows which sign is interesting, which is the point:
    the round-1 defect came from code that decided the sign FIRST and then chose
    an interval to match it.
    """
    n = i1 - i0
    mp = max(3, int(min_points))
    if n < 2 * mp:
        return None
    _, _, sse1, _, _ = cum.fit(i0, i1)
    best_s, best_sse = -1, np.inf
    for s in range(i0 + mp, i1 - mp + 1):
        _, _, sa, _, _ = cum.fit(i0, s)
        _, _, sb, _, _ = cum.fit(s, i1)
        tot = sa + sb
        if np.isfinite(tot) and tot < best_sse:
            best_sse, best_s = tot, s
    if best_s < 0 or not np.isfinite(sse1):
        return None
    gA, _, _, _, _ = cum.fit(i0, best_s)
    gB, _, _, _, _ = cum.fit(best_s, i1)
    if not (np.isfinite(gA) and np.isfinite(gB)):
        return None
    t_turn = 0.5 * (t[best_s - 1] + t[best_s])
    diverge = abs(gA - gB) * abs(t[i1 - 1] - t_turn)
    accepted = bool(best_sse <= turn_sse_frac * sse1 and diverge >= turn_efold)
    if not accepted:
        return None
    return dict(i_split=int(best_s), gamma_early=float(gA),
                gamma_late=float(gB), t_turn=float(t_turn),
                sse_one=float(sse1), sse_two=float(best_sse),
                sse_ratio=float(best_sse / sse1) if sse1 > 0 else 0.0,
                diverge_efolds=float(diverge))


# ---------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------
def fit_growth_interval(t, a, floor=None, floor_abs=FP_FLOOR_SINGLE,
                        snr_amp=4.0, sat_level=None, sat_frac=0.5,
                        linear_mask=None, min_points=5, min_efold=1.0,
                        min_r2=0.9, turn_sse_frac=TURN_SSE_FRAC,
                        turn_efold=TURN_EFOLD, label=None):
    """Read the MEASURAND above, or return no number.

    Returns a dict.  Read `status` before `gamma`; `gamma` is nan whenever
    status != "ok", so a held or non-summarisable verdict cannot be mistaken for
    a rate.  On a record that turns over, `gamma` is the EARLY branch and the
    later branch is reported under `late` -- never in `gamma`.

    Keys a caller is expected to use:
      status      "ok" | "HELD" | "NOT_SUMMARISABLE"
      gamma       the measurand, 1/s, nan unless status == "ok"
      gamma_se    standard error of the slope on the fitted interval
      branch      which branch gamma came from: "whole" (no turn-over) or
                  "early" (a turn-over was found and gamma is the EARLY side)
      turnover    dict or None: the accepted change of slope
      late        dict or None: the branch AFTER the turn-over, with its own
                  gamma/r2/n.  Present so a grow-then-damp record can be
                  reported as what it is; it is never the measurand.
                  Also used for a readable interval after a linearity breach;
                  in that case turnover may be None. See late["note"].
      measurable  False for every status in NOT_MEASURABLE_STATES.  A threshold
                  bracket must read this, not the sign of gamma.
    """
    t = np.asarray(t, dtype=np.float64)
    a = np.abs(np.asarray(a, dtype=np.float64))
    if t.shape != a.shape or t.ndim != 1:
        raise ValueError("t and a must be 1-D of the same length")

    floor_meas = float(floor) if floor is not None else 0.0
    floor_eff = max(floor_meas, float(floor_abs))
    a_min = snr_amp * floor_eff

    # ---- clause (1) SIGNAL
    keep = np.isfinite(a) & (a > 0) & (a >= a_min)
    cuts = dict(n_total=int(a.size), n_above_floor=int(keep.sum()))

    # ---- clause (2) LINEARITY
    linear = np.ones(a.shape, dtype=bool)
    if linear_mask is not None:
        lm = np.asarray(linear_mask, dtype=bool)
        if lm.shape != a.shape:
            raise ValueError("linear_mask must match the series length")
        linear &= lm
        keep &= lm
        cuts["n_after_linear_mask"] = int(keep.sum())
    if sat_level is not None:
        below_sat = a <= sat_frac * float(sat_level)
        linear &= below_sat
        keep &= below_sat
        cuts["n_after_saturation_cut"] = int(keep.sum())

    # Linearity is a chronological boundary: returning below the ceiling does
    # not turn post-saturation damping into an early instability measurement.
    breaches = np.flatnonzero(~linear)
    linear_end = int(breaches[0]) if breaches.size else a.size
    readable = keep.copy()
    keep[linear_end:] = False
    cuts["i_linear_end"] = linear_end
    cuts["n_before_linear_end"] = int(keep.sum())

    criteria = dict(snr_amp=float(snr_amp), floor_abs=float(floor_abs),
                    sat_level=(None if sat_level is None else float(sat_level)),
                    sat_frac=float(sat_frac), min_points=int(min_points),
                    min_efold=float(min_efold), min_r2=float(min_r2),
                    turn_sse_frac=float(turn_sse_frac),
                    turn_efold=float(turn_efold))
    out = dict(status=HELD, measurable=False, gamma=float("nan"),
               gamma_se=float("nan"), intercept=float("nan"), r2=float("nan"),
               i0=0, i1=0, n=0, t0=float("nan"), t1=float("nan"),
               efolds=float("nan"), branch="none", turnover=None, late=None,
               n_turnovers=0, floor_eff=float(floor_eff),
               floor_measured=float(floor_meas), a_min=float(a_min),
               label=label, cuts=cuts, criteria=criteria, measurand=MEASURAND,
               reason="")

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.where(a > 0, a, np.nan))
    cum = _Cum(t, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))

    def with_late_diagnostic(result):
        if result["late"] is not None or linear_end == a.size:
            return result
        for l0, l1 in _runs(readable):
            if l0 < linear_end or l1 - l0 < mp:
                continue
            late_fit = _evaluate_window(t, y, cum, l0, l1, mp, min_efold,
                                        min_r2, turn_sse_frac, turn_efold)
            if late_fit["status"] != OK:
                continue
            result["late"] = _late_diagnostic(
                late_fit, "readable interval AFTER the first linearity breach. "
                "NOT the early linear measurand.")
            if result["status"] != OK:
                result.update(status=NOT_SUMMARISABLE, measurable=False)
                result["reason"] = (
                    "this record cannot be summarised by ONE growth rate: "
                    "only an interval AFTER the first linearity breach at "
                    "t = %.4g s is readable; the early interval is unread: %s"
                    % (t[linear_end], result["reason"]))
            break
        return result

    # ---- the EARLIEST readable interval, not the longest one: "early" is part
    # of the measurand, so the window is chosen by TIME (see WINDOW SELECTION
    # below), never by which interval gives the nicer slope.
    mp = max(3, int(min_points))
    runs = _runs(keep)
    cuts["n_runs"] = len(runs)
    cuts["run_lengths"] = [int(j - i) for i, j in runs]
    candidates = [r for r in runs if r[1] - r[0] >= mp]
    if not candidates:
        i0, i1 = _longest_run(keep)
        out.update(i0=int(i0), i1=int(i1), n=int(i1 - i0))
        out["reason"] = ("no contiguous interval of >= %d samples satisfies "
                         "a >= %.3e (%.1fx floor_eff %.3e) with the linearity "
                         "cut applied: longest run is %d of %d samples"
                         % (mp, a_min, snr_amp, floor_eff, i1 - i0, a.size))
        return with_late_diagnostic(out)

    # ---- WINDOW SELECTION.  The candidate windows are fixed by clauses (1) and
    # (2) BEFORE any fit.  The rule is then "the EARLIEST candidate that
    # satisfies clause (4)", a rule stated in times and sample counts that never
    # compares one slope with another.  It has to be "earliest that satisfies
    # (4)" rather than plainly "earliest", because a real record crosses the
    # signal criterion several times while the signal is still emerging:
    # measured on the sim40 bin-5 array at eps_0 = 7e-5, the surviving samples
    # form runs of 1, 2, 4, 5, 6, 15, 615, 6, 5, 4 samples, and a plain "first
    # run of >= 5 samples" rule picks the 5-sample blip at t = 1.36 ns and
    # throws the 615-sample interval away.
    tried, chosen = [], None
    for (w0, w1) in candidates:
        cand = _evaluate_window(t, y, cum, w0, w1, mp, min_efold, min_r2,
                                turn_sse_frac, turn_efold)
        tried.append(dict(i0=int(w0), i1=int(w1), n=int(w1 - w0),
                          t0=float(t[w0]), t1=float(t[w1 - 1]),
                          status=cand["status"], reason=cand["reason"]))
        if cand["status"] == OK:
            chosen = cand
            break
    cuts["candidate_windows"] = tried
    if chosen is None:
        # Nothing is readable.  Report the diagnosis of the LONGEST candidate,
        # the most informative one, and keep every candidate's reason in `cuts`
        # so the refusal itself can be audited.
        wl = max(candidates, key=lambda r: r[1] - r[0])
        chosen = _evaluate_window(t, y, cum, wl[0], wl[1], mp, min_efold,
                                  min_r2, turn_sse_frac, turn_efold)
    w_lo, w_hi = chosen.pop("window")
    out.update(chosen)
    cuts["i_window"] = [int(w_lo), int(w_hi)]
    cuts["n_window"] = int(w_hi - w_lo)

    # ---- PRECEDENCE.  A negative rate may only be reported if the record did
    # NOT rise into the fitted window.  A decay measured after the amplitude has
    # already climbed is the LATE branch of a mode that grew -- exactly the
    # number round 1 returned as "the growth rate" -- and it stays wrong when the
    # growth happened in an earlier window rather than earlier in the same one.
    # The rise is measured against every earlier sample of the record, including
    # the ones clause (1) rejected, because a rise that happened below the signal
    # floor is still a rise and its rate is simply not measurable.
    out["rise_into_window_efolds"] = 0.0
    if out["status"] == OK and out["gamma"] < 0:
        before = y[:w_lo][np.isfinite(y[:w_lo])]
        rise = float(y[w_lo] - before.min()) if before.size else 0.0
        out["rise_into_window_efolds"] = rise
        if rise >= float(turn_efold):
            out["late"] = _late_diagnostic(
                out, "readable decay AFTER an earlier rise. NOT the measurand.")
            out.update(status=NOT_SUMMARISABLE, measurable=False,
                       gamma=float("nan"), gamma_se=float("nan"),
                       branch="late")
            out["reason"] = (
                "the amplitude ROSE by %.3f e-foldings (>= %.3f) into the "
                "fitted interval, which starts at t = %.4g s, so the negative "
                "slope on it is the LATE branch of a mode that grew and not the "
                "early linear rate. No earlier interval satisfied all the "
                "measurement criteria (including a >= %.3e), so this record "
                "cannot be summarised by one growth rate."
                % (rise, turn_efold, t[w_lo], a_min))
    return with_late_diagnostic(out)


def _late_diagnostic(fit, note):
    keys = ("gamma", "gamma_se", "r2", "n", "i0", "i1", "t0", "t1", "efolds")
    return dict({key: fit[key] for key in keys}, note=note)


def _evaluate_window(t, y, cum, w0, w1, mp, min_efold, min_r2, turn_sse_frac,
                     turn_efold):
    """Clauses (3) and (4) on ONE candidate window.  No selection happens here."""
    res = dict(window=(int(w0), int(w1)), status=HELD, measurable=False,
               gamma=float("nan"), gamma_se=float("nan"),
               intercept=float("nan"), r2=float("nan"), branch="whole",
               turnover=None, late=None, n_turnovers=0, i0=int(w0), i1=int(w1),
               n=int(w1 - w0), t0=float(t[w0]), t1=float(t[w1 - 1]),
               efolds=float("nan"), reason="")

    # ---- clause (3) EARLINESS: split off the earliest branch, repeatedly, so a
    # record with several turns still yields its FIRST branch.
    e0, e1 = int(w0), int(w1)
    turns = []
    for _ in range(MAX_TURNS):
        tr = _first_turn(t, cum, e0, e1, mp, turn_sse_frac, turn_efold)
        if tr is None:
            break
        turns.append(tr)
        e1 = tr["i_split"]
    res["n_turnovers"] = len(turns)
    if turns:
        first = turns[-1]                     # the earliest accepted split
        res["turnover"] = first
        res["branch"] = "early"
        l0, l1 = first["i_split"], (turns[-2]["i_split"] if len(turns) > 1
                                    else int(w1))
        g_l, _, sse_l, sst_l, n_l = cum.fit(l0, l1)
        res["late"] = dict(
            gamma=float(g_l),
            r2=float(1.0 - sse_l / sst_l) if sst_l > 0 else float("nan"),
            n=int(n_l), i0=int(l0), i1=int(l1),
            t0=float(t[l0]), t1=float(t[l1 - 1]),
            efolds=float(y[l1 - 1] - y[l0]), gamma_se=cum.stderr(l0, l1),
            note="the branch AFTER the turn-over. NOT the measurand.")

    # ---- clause (4) ADEQUACY, on the early branch and on nothing else
    n = e1 - e0
    res.update(i0=int(e0), i1=int(e1), n=int(n), t0=float(t[e0]),
               t1=float(t[e1 - 1]))
    efolds = float(y[e1 - 1] - y[e0])
    res["efolds"] = efolds

    def _fail(reason):
        """HELD, or NOT_SUMMARISABLE when only the LATE branch was readable."""
        res["reason"] = reason
        late = res["late"]
        if late is not None and int(late["n"]) >= mp \
                and abs(late["efolds"]) >= float(min_efold) \
                and np.isfinite(late["r2"]) and late["r2"] >= float(min_r2):
            res["status"] = NOT_SUMMARISABLE
            res["reason"] = (
                "this record cannot be summarised by ONE growth rate: it turns "
                "over at t = %.4g s and only the LATE branch satisfies the "
                "criteria (gamma_late = %+.4e 1/s, r2 = %.4f, %d samples), "
                "while the EARLY branch -- the measurand -- does not: %s. The "
                "late slope is NOT returned as the growth rate."
                % (res["turnover"]["t_turn"], late["gamma"], late["r2"],
                   late["n"], reason))
        else:
            res["status"] = HELD
        return res

    if not np.isfinite(efolds):
        return _fail("the interval contains a non-positive or non-finite "
                     "amplitude, so ln|a| is undefined on it")
    if n < mp:
        if turns:
            return _fail("the early branch holds only %d samples (< %d) once "
                         "the turn-over at t = %.4g s is cut off"
                         % (n, mp, res["turnover"]["t_turn"]))
        return _fail("the interval holds only %d samples (< %d)" % (n, mp))
    if abs(efolds) < float(min_efold):
        return _fail("interval spans only %.3f e-foldings (< %.3f): the slope "
                     "is not separable from the fluctuation of the series"
                     % (efolds, min_efold))
    g, b, sse, sst, _ = cum.fit(e0, e1)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    res["r2"] = float(r2)
    if min_r2 is not None and not (r2 >= float(min_r2)):
        return _fail("the series is not exponential on the stated interval: "
                     "r2 = %.4f < %.3f. The interval was fixed by the signal, "
                     "linearity and turn-over criteria BEFORE the fit, so this "
                     "is a statement about the data, not a search over "
                     "intervals." % (r2, min_r2))
    res.update(status=OK, measurable=True, gamma=float(g), intercept=float(b),
               gamma_se=cum.stderr(e0, e1), reason="")
    return res


# ---------------------------------------------------------------------------
# the k-resolved map: the SAME fitter, once, for every caller
# ---------------------------------------------------------------------------
def segment_layout(n_samples, seg=None, stride=None):
    """Return (seg, stride, n_seg), independent of acquisition duration.

    The default is the existing minimum window of 64 recorded samples, with
    quarter-window strides. A caller needing finer frequency resolution must
    supply a fixed seg that still resolves the early instability time scale.
    Increasing total runtime must never increase the fit's temporal averaging.
    """
    seg = 64 if seg is None else int(seg)
    stride = max(1, seg // 4) if stride is None else int(stride)
    if seg < 3 or stride < 1 or int(n_samples) < seg:
        raise ValueError("need n_samples >= seg >= 3 and stride >= 1")
    return seg, stride, 1 + (int(n_samples) - seg) // stride


def gamma_map(my, dx, dt, f0, seg=None, stride=None, m_linear=0.1,
              k_offband=None, snr_amp=4.0, min_efold=1.0, min_points=5,
              amp_calibration=None):
    """Band-resolved Gamma(k) at f0 through `fit_growth_interval`, per bin.

    THE single entry point for a k-resolved growth rate.  G4, G4b and G9 all
    call this; there is no second implementation, because a second
    implementation is a second measurand (that is how the dead-bin defect
    survived in G4b and G9 after G4 was fixed).

    Returns (k_ax, G, R, STAT, t0s, jf, f_ax, meta).  `STAT` is one status
    string per k bin and must be read before `G`.  `meta["best"]` holds the
    largest MEASURABLE positive-k rate and its bin, or a NOT_DETERMINABLE state:
    the maximum is never taken over bins whose rate was not measured.
    `meta["late"][j]` retains the late interval's diagnostics; gamma_late is
    retained as the compatible list of late slopes. Segment lengths come from
    segment_layout, not from a fraction of the record duration.
    """
    import saw_analysis as SA                                   # noqa: PLC0415

    my = np.asarray(my, dtype=float)
    nt, nx = my.shape[0], my.shape[1]
    seg, stride, n_seg = segment_layout(nt, seg=seg, stride=stride)
    k_ax, f_ax, Ms_, t0s = SA.segment_spectra(my, dx, dt, seg, n_seg,
                                              stride=stride)
    jf, _ = SA.f_slice(Ms_[0], f_ax, f0)
    if amp_calibration is None:
        k_ref = k_ax[int(np.argmin(np.abs(
            k_ax - abs(k_ax[k_ax > 0]).min())))] if (k_ax > 0).any() else 0.0
        amp_calibration = _amp_norm(nx, dx, dt, seg, stride, f0, k_ref)
    cal = float(amp_calibration)
    A = np.abs(Ms_[:, jf, :]) / cal                      # units of m
    tmid = t0s + seg * dt / 2

    i0s = np.rint(t0s / dt).astype(int)
    mmax = np.array([np.abs(my[i:i + seg]).max() for i in i0s])
    lin = mmax < m_linear

    k_offband = (np.abs(k_ax).max() / 2 if k_offband is None
                 else float(k_offband))
    off = np.abs(k_ax) > k_offband
    floor = float(np.median(A[:, off])) if off.any() else 0.0

    nb = A.shape[1]
    G = np.full(nb, np.nan)
    R = np.full(nb, np.nan)
    SE = np.full(nb, np.nan)
    STAT = np.full(nb, HELD, dtype=object)
    WHY = [""] * nb
    BRANCH = [""] * nb
    LATE = [None] * nb
    LATE_FITS = [None] * nb
    for jj in range(nb):
        r = fit_growth_interval(tmid, A[:, jj], floor=floor, snr_amp=snr_amp,
                                sat_level=m_linear, linear_mask=lin,
                                min_points=min_points, min_efold=min_efold,
                                label="k=%.4f um^-1" % (k_ax[jj] / 1e6))
        G[jj], R[jj], SE[jj] = r["gamma"], r["r2"], r["gamma_se"]
        STAT[jj], WHY[jj], BRANCH[jj] = r["status"], r["reason"], r["branch"]
        LATE[jj] = None if r["late"] is None else float(r["late"]["gamma"])
        LATE_FITS[jj] = r["late"]

    ok_pos = (STAT == OK) & (k_ax > 0)
    if ok_pos.any():
        jb = int(np.where(ok_pos)[0][int(np.nanargmax(G[ok_pos]))])
        best = dict(state=OK, gamma=float(G[jb]), k_um=float(k_ax[jb] / 1e6),
                    j=int(jb), r2=float(R[jb]))
    else:
        best = dict(state="NOT_DETERMINABLE", gamma=float("nan"),
                    k_um=float("nan"), j=-1, r2=float("nan"),
                    reason="no positive-k bin returned a measured rate")
    meta = dict(seg=seg, stride=stride, n_seg=int(n_seg), amp_calibration=cal,
                segment_duration=float(seg * dt),
                segment_df=float(1.0 / (seg * dt)),
                floor_amplitude=floor, floor_abs=FP_FLOOR_SINGLE,
                snr_amp=float(snr_amp), m_linear=float(m_linear),
                min_efold=float(min_efold), min_points=int(min_points),
                k_offband=float(k_offband),
                n_segments_linear=int(lin.sum()),
                max_my_first=float(mmax[0]), max_my_last=float(mmax[-1]),
                n_bins_ok=int((STAT == OK).sum()),
                n_bins_not_summarisable=int((STAT == NOT_SUMMARISABLE).sum()),
                best=best, gamma_se=SE, branch=BRANCH, gamma_late=LATE,
                late=LATE_FITS,
                held_reasons=WHY, measurand=MEASURAND)
    return k_ax, G, R, STAT, t0s, jf, f_ax, meta


def _amp_norm(nx, dx, dt, seg, stride, f0, k_ref):
    """|M| of a UNIT-amplitude on-bin travelling wave under exactly the same
    transform, window, segment length and f-slice, so the absolute amplitude
    criterion of clause (1) can be applied to unnormalised segment spectra."""
    import saw_analysis as SA                                   # noqa: PLC0415

    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(seg) * dt
    w = np.cos(k_ref * x[None, :] - 2 * np.pi * f0 * t[:, None])
    _, fc, Mc, _ = SA.segment_spectra(w, dx, dt, seg, 1, stride=stride)
    jfc, _ = SA.f_slice(Mc[0], fc, f0)
    return float(np.abs(Mc[0, jfc, :]).max())


# ---------------------------------------------------------------------------
# the threshold bracket
# ---------------------------------------------------------------------------
def point_from_fit(eps, fit, note=""):
    """One bracket point, carrying WHERE its number came from.

    A bare float cannot say whether it is a measured negative rate or the
    post-turn-over damping of a mode that grew, and that distinction is the
    whole content of the bracket.  So a bracket point is built from a fit
    result, never from a number.
    """
    if not isinstance(fit, dict) or "status" not in fit:
        raise TypeError("a bracket point is built from a fit_growth_interval "
                        "result, not from %r" % (type(fit).__name__,))
    return dict(eps=float(eps), gamma=float(fit["gamma"]),
                status=str(fit["status"]),
                branch=str(fit.get("branch", "")),
                r2=float(fit.get("r2", np.nan)),
                measurable=bool(fit.get("measurable",
                                        fit["status"] == OK)),
                reason=str(fit.get("reason", "")), note=str(note))


def bracket_from_fits(points, tol=0.0):
    """Does a strain sweep bracket a sign change, given WHERE each rate is from?

    `points` is a sequence of `point_from_fit` dicts (or anything with the same
    keys).  Every point must be MEASURABLE: a HELD or NOT_SUMMARISABLE point is
    not a negative-growth point, it is an unread one, and a bracket built on it
    is not a bracket.  This is the only bracket a campaign gate may use.
    """
    pts = list(points)
    for p in pts:
        for k in ("eps", "gamma", "status"):
            if k not in p:
                raise TypeError("bracket point is missing %r: %r" % (k, p))
    bad = [p for p in pts if str(p["status"]) != OK]
    eps = [float(p["eps"]) for p in pts]
    gam = [float(p["gamma"]) for p in pts]
    if bad:
        out = bracket_check(eps, gam, tol=tol)
        out["status"] = "NOT_EVALUABLE"
        out["n_neg"], out["n_pos"] = 0, 0
        out["eps_threshold_interval"] = None
        out["reason"] = (
            "%d of %d points in the sweep have NO measured early growth rate "
            "(%s). Not measurable is not measured negative: %s. Resolve or drop "
            "these runs; the sweep cannot be said to bracket anything until "
            "then."
            % (len(bad), len(pts),
               ", ".join("eps=%.3e %s" % (p["eps"], p["status"]) for p in bad),
               "; ".join("eps=%.3e: %s" % (p["eps"], p["reason"])
                         for p in bad if p.get("reason"))[:600]))
        out["unmeasured"] = [dict(eps=float(p["eps"]), status=str(p["status"]),
                                  reason=str(p.get("reason", ""))[:300])
                             for p in bad]
        return out
    out = bracket_check(eps, gam, tol=tol,
                        status=[str(p["status"]) for p in pts],
                        branch=[str(p.get("branch", "")) for p in pts])
    out["points"] = [dict(eps=float(p["eps"]), gamma=float(p["gamma"]),
                          status=str(p["status"]),
                          branch=str(p.get("branch", ""))) for p in pts]
    return out


def bracket_check(eps, gamma, tol=0.0, status=None, branch=None):
    """Does a strain sweep actually bracket a sign change of the growth rate?

    Audit 2026-09-18 sec.8: the threshold needs an interval containing a real
    negative AND a real positive growth rate; the YIG sweep as planned started
    ABOVE its own predicted threshold, so that condition was not guaranteed.

    A nan anywhere (an unread point) makes the sweep NOT_EVALUABLE: an unread
    point blocks the threshold statement instead of being dropped from it.  When
    `status` is supplied, any status other than "ok" does the same, whatever the
    value of gamma -- which is the case a bare nan check cannot catch, because
    round 1 returned a post-turn-over damping rate as a FINITE negative number.
    Campaign gates call `bracket_from_fits`, which always supplies it.
    """
    eps = np.asarray(eps, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if eps.shape != g.shape:
        raise ValueError("eps and gamma must have the same shape")
    out = dict(eps=[float(x) for x in eps], gamma=[float(x) for x in g],
               n_neg=0, n_pos=0, status="NOT_BRACKETED", reason="",
               eps_neg_max=float("nan"), eps_pos_min=float("nan"),
               eps_threshold_interval=None)
    if status is not None:
        st = [str(s) for s in status]
        if len(st) != g.size:
            raise ValueError("status must match the sweep length")
        bad = [(float(e), s) for e, s in zip(eps, st) if s != OK]
        if bad:
            out["status"] = "NOT_EVALUABLE"
            out["reason"] = ("%d of %d points carry no measured rate (%s): a "
                             "point that could not be read is not a "
                             "negative-growth point"
                             % (len(bad), g.size,
                                ", ".join("eps=%.3e %s" % b for b in bad)))
            return out
    if branch is not None:
        br = [str(b) for b in branch]
        if len(br) != g.size:
            raise ValueError("branch must match the sweep length")
        late = [float(e) for e, b in zip(eps, br) if b == "late"]
        if late:
            out["status"] = "NOT_EVALUABLE"
            out["reason"] = ("points at eps = %s report a LATE branch as their "
                             "rate; the measurand is the early branch and a "
                             "post-turn-over slope may not enter a bracket"
                             % late)
            return out
    if not np.all(np.isfinite(g)):
        out["status"] = "NOT_EVALUABLE"
        out["reason"] = ("%d of %d growth rates are HELD/nan: the sweep cannot "
                         "be said to bracket anything until they are resolved"
                         % (int((~np.isfinite(g)).sum()), g.size))
        return out
    neg, pos = g < -tol, g > tol
    out["n_neg"], out["n_pos"] = int(neg.sum()), int(pos.sum())
    if neg.any():
        out["eps_neg_max"] = float(eps[neg].max())
    if pos.any():
        out["eps_pos_min"] = float(eps[pos].min())
    if out["n_neg"] and out["n_pos"]:
        lo, hi = out["eps_neg_max"], out["eps_pos_min"]
        if lo < hi:
            out["status"] = OK
            out["eps_threshold_interval"] = [lo, hi]
        else:
            out["reason"] = ("signs are not ordered in eps: the largest "
                             "negative point (%.3e) is above the smallest "
                             "positive one (%.3e), so no single threshold is "
                             "bracketed" % (lo, hi))
        return out
    out["reason"] = ("sweep has %d negative and %d positive rates; a threshold "
                     "needs at least one of each" % (out["n_neg"], out["n_pos"]))
    return out
