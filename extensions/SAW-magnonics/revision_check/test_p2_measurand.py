"""ROUND-2 counterexamples: one measurand, one fitter, one plan, one bracket.

Every test here attacks a MECHANISM, from an angle none of the earlier suites
used, and each one is written so that it would go RED against the shape of the
round-1 code rather than against the single example the auditor wrote down.

  M1  THE MEASURAND IS DECLARED ONCE.  PREREGISTRATION.md section 9 and
      growth_interval.MEASURAND must be byte-identical, and the definition must
      reach the output: every fit result carries it.  New angle: nobody has tested
      that the specification and the code say the same thing.
  M2  THE FITTER KNOWS EARLY FROM LATE.  Not just the auditor's
      +3.0e8-then--1.2e8 record: a family of turn-overs (five rates, three
      turn-over times, growth->decay, growth->plateau, decay->growth), each of
      which must come back as the EARLY branch or as an explicit refusal, and
      NEVER as a confident number of the wrong sign.  Also: the late branch must
      be reported, so a grow-then-damp record is describable rather than merely
      refused.
  M3  A LATE BRANCH IN A SEPARATE WINDOW IS STILL A LATE BRANCH.  The nastiest
      variant: the growth happens BELOW the signal criterion and only the decay
      is above it, so there is no turn-over inside any single window to detect.
      Round 1 would have fitted the decay and called it the growth rate; so would
      a fix that only looked for a peak inside its own window.
  M4  ONE FITTER, ENFORCED.  No campaign script may take a growth rate from
      saw_analysis.growth_fit -- checked by CALLING it from a campaign module and
      requiring the refusal, not by grepping -- while non-campaign callers keep
      working.  New angle: the earlier suite pinned the defect per file, so a
      fourth script would have reopened it.
  M5  THE MAP'S MAXIMUM IS OVER MEASURED BINS ONLY, in G4 AND in G4b, through the
      same call.  Attacked with a dead bin whose rate is enormous AND a
      turned-over bin whose late slope is steeply negative, in one record.
  M6  A BLOCKED PLAN HAS NO RUN LENGTH.  Not "the callers now read the status":
      an ARBITRARY caller -- including this test, which is a caller nobody wrote
      the fix for -- cannot get a run length or a record count out of a blocked
      plan.  And a runnable plan still hands both over.
  M7  THE BRACKET REFUSES WHAT IT CANNOT READ.  Built from fit results: an
      unreadable point, a late-branch point, and a point present for one seed
      only are each refused; a genuine sign change is still accepted.
  M8  THE COST TABLE IS DERIVED.  campaign_cost.py must contain no hard-coded
      strain sweep at all and its total must reproduce RUN_PLAN section 5.

Run:  python test_p2_measurand.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
for p in (HERE, RUNS):
    if p not in sys.path:
        sys.path.insert(0, p)

import _harness as H                                              # noqa: E402
import growth_interval as GI                                      # noqa: E402
import saw_analysis as SA                                         # noqa: E402


# ------------------------------------------------------------- fixtures -----
def _wave(nt, nx, dx, dt, k, f, env):
    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(nt) * dt
    return np.asarray(env)[:, None] * np.cos(k * x[None, :]
                                             - 2 * np.pi * f * t[:, None])


def _two_branch(t, a0, g_up, g_dn, t_pk):
    return np.where(t <= t_pk, a0 * np.exp(g_up * t),
                    a0 * np.exp(g_up * t_pk) * np.exp(g_dn * (t - t_pk)))


# ------------------------------------------------------------------- M1 -----
def test_m1_measurand_is_one_text_shared_by_spec_and_code():
    spec = open(os.path.join(HERE, "PREREGISTRATION.md"),
                encoding="utf-8").read()
    blocks = re.findall(r"```\n(MEASURAND \(growth rate\)\..*?)\n```", spec,
                        re.S)
    assert len(blocks) == 1, (
        "PREREGISTRATION.md must carry the measurand statement exactly once as a "
        "fenced block; found %d" % len(blocks))
    assert blocks[0].strip() == GI.MEASURAND.strip(), (
        "the pre-registered measurand and growth_interval.MEASURAND are NOT the "
        "same text, so the specification and the code can drift. First "
        "difference at character %d."
        % next((i for i, (x, y) in enumerate(zip(blocks[0].strip(),
                                                 GI.MEASURAND.strip()))
                if x != y), min(len(blocks[0].strip()),
                                len(GI.MEASURAND.strip()))))
    t = np.arange(400) * 20e-12
    r = GI.fit_growth_interval(t, 1e-4 * np.exp(3e8 * t), floor=1e-9,
                              sat_level=0.1)
    assert r.get("measurand", "").strip() == GI.MEASURAND.strip(), (
        "a fit result must carry the measurand it reports, so a number can "
        "never be read without its definition")
    for key in ("snr_amp", "min_efold", "min_r2", "turn_sse_frac",
                "turn_efold", "min_points"):
        assert key in r["criteria"], (
            "criterion %r is named in the measurand but is not reported with "
            "the fit" % key)


# ------------------------------------------------------------------- M2 -----
def test_m2_no_turnover_record_is_ever_reported_with_the_wrong_sign():
    """A FAMILY of turn-overs, not the one example that was written down."""
    dt, bad = 20e-12, []
    for g_up in (3.0e8, 1.5e8, 5.0e7):
        for g_dn, kind in ((-1.2e8, "decay"), (-4.0e8, "fast decay"),
                           (0.0, "plateau")):
            for t_pk in (2e-9, 4e-9, 8e-9):
                t = np.arange(1501) * dt
                a = _two_branch(t, 1e-4, g_up, g_dn, t_pk)
                r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
                if r["status"] == GI.OK:
                    if r["gamma"] <= 0 or abs(r["gamma"] / g_up - 1) > 0.05:
                        bad.append("g_up=%.2e %s t_pk=%.0f ns -> %s gamma=%+.4e"
                                   % (g_up, kind, t_pk * 1e9, r["status"],
                                      r["gamma"]))
                elif r["status"] not in GI.NOT_MEASURABLE_STATES:
                    bad.append("unknown status %r" % r["status"])
                elif np.isfinite(r["gamma"]):
                    bad.append("status %s but gamma = %r" % (r["status"],
                                                             r["gamma"]))
    assert not bad, ("a record that grew and then turned over was summarised by "
                     "a rate that is not its early growth rate:\n    "
                     + "\n    ".join(bad))


def test_m2_the_late_branch_is_reported_not_merely_discarded():
    t = np.arange(1501) * 20e-12
    a = _two_branch(t, 1e-4, 3.0e8, -1.2e8, 4e-9)
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert r["status"] == GI.OK and abs(r["gamma"] - 3.0e8) < 1e-3 * 3.0e8, (
        "the early branch of the auditor's own record must come back as "
        "+3.0000e+08: got %s / %r" % (r["status"], r["gamma"]))
    assert r["late"] is not None, (
        "the collaborator asked for the early AND the late rate to be "
        "distinguished; the late branch is absent from the result")
    assert abs(r["late"]["gamma"] - (-1.2e8)) < 1e-3 * 1.2e8, (
        "the late branch must be reported as -1.2000e+08, got %r"
        % r["late"]["gamma"])
    assert r["branch"] == "early" and r["turnover"] is not None


def test_m2_decay_then_growth_is_not_reported_as_a_clean_decay():
    """The mirror image, which no earlier test covers: a record that DECAYS and
    then grows must not be summarised by the decay alone without saying that a
    turn-over was found."""
    t = np.arange(1501) * 20e-12
    a = _two_branch(t, 1e-2, -1.0e8, +2.0e8, 6e-9)
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert r["status"] != GI.OK or r["turnover"] is not None, (
        "a decay-then-growth record came back as one rate %+.4e with no "
        "turn-over recorded" % r["gamma"])
    if r["status"] == GI.OK:
        assert abs(r["gamma"] - (-1.0e8)) < 0.05 * 1.0e8, (
            "the early branch of a decay-then-growth record is -1.0e8; got "
            "%+.4e" % r["gamma"])


def test_m2_a_clean_single_exponential_is_not_split():
    """Do not over-block: the turn-over machinery must leave an ordinary record
    alone, with and without noise."""
    rng = np.random.default_rng(11)
    for g in (3.0e8, -9.578e6, 1.0e7):
        # Each record is given the SAME 3 e-foldings of span, so the test is
        # about the turn-over machinery and not about whether 0.14 e-foldings
        # under 2% noise look exponential (they do not, and the fitter is right
        # to refuse that -- see test_m2_short_span_with_noise_is_refused).
        t = np.linspace(0.0, 3.0 / abs(g), 751)
        for rel in (0.0, 0.02, 0.05):
            a = 1e-3 * np.exp(g * t) * (1 + rel * rng.standard_normal(t.size))
            r = GI.fit_growth_interval(t, np.abs(a), floor=1e-9, sat_level=0.1)
            assert r["status"] == GI.OK, (
                "a clean exponential at %+.2e 1/s spanning 3 e-foldings with "
                "%.0f%% noise was refused: %s" % (g, 100 * rel, r["reason"]))
            assert r["n_turnovers"] == 0, (
                "a single exponential at %+.2e with %.0f%% noise was split into "
                "%d branches" % (g, 100 * rel, r["n_turnovers"]))
            assert abs(r["gamma"] - g) < 0.05 * abs(g), (
                "%+.4e vs true %+.4e" % (r["gamma"], g))


def test_m2_short_span_with_noise_is_refused_rather_than_fitted():
    """The other half of the previous test, stated as its own requirement: a
    record that spans 0.14 e-foldings under 2% multiplicative noise carries no
    separable rate, and the fitter must say so instead of returning the slope."""
    rng = np.random.default_rng(11)
    t = np.arange(751) * 20e-12                       # 15 ns
    g = -9.578e6                                      # 0.144 e-foldings in all
    a = 1e-3 * np.exp(g * t) * (1 + 0.02 * rng.standard_normal(t.size))
    r = GI.fit_growth_interval(t, np.abs(a), floor=1e-9, sat_level=0.1,
                              min_efold=0.1)
    assert r["status"] != GI.OK, (
        "0.144 e-foldings under 2%% noise were fitted to %+.4e 1/s at "
        "r2 = %.4f" % (r["gamma"], r["r2"]))
    assert not np.isfinite(r["gamma"])


# ------------------------------------------------------------------- M3 -----
def test_m3_a_record_that_starts_after_its_peak_is_not_a_growth_rate():
    """THE angle nobody has used, case (a): the RECORD, not the signal, is late.
    The mode peaks 0.5 ns in and then damps for 29.5 ns, so the early branch
    holds 0.15 e-foldings and the late branch holds 3.5 -- every sample is far
    above the signal criterion, the late fit has r2 = 1.0000, and there is
    nothing wrong with the data.  Round 1 returned the damping rate here; a fix
    that preferred "the growing side of the peak" would return a slope measured
    over 25 samples and a sixth of an e-folding.  The only correct answer is that
    this record does not carry the measurand.

    Note why an earlier version of this test was wrong, because it is the
    interesting part: if the SIGNAL FLOOR is what cuts both branches, the two
    branches span the SAME number of e-foldings by construction (both run from
    the floor to the peak), so a short early branch forces a short late one and
    the answer is simply HELD.  Asymmetry has to come from the record boundary,
    from saturation, or from a gap -- which is what cases (a) and (b) use."""
    t = np.arange(1501) * 20e-12
    a = _two_branch(t, 1e-4, 3.0e8, -1.2e8, 0.5e-9)
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert r["status"] != GI.OK, (
        "the early branch spans 0.15 e-foldings over 25 samples and the late one "
        "spans 3.5; the fitter returned gamma = %+.4e with status %s, r2 = %.4f "
        "on t = %.2f-%.2f ns"
        % (r["gamma"], r["status"], r["r2"], r["t0"] * 1e9, r["t1"] * 1e9))
    assert r["status"] == GI.NOT_SUMMARISABLE, (
        "a record whose only readable branch is the LATE one must be "
        "NOT_SUMMARISABLE, got %s: %s" % (r["status"], r["reason"]))
    assert r["late"] is not None and r["late"]["gamma"] < 0
    assert abs(r["late"]["gamma"] + 1.2e8) < 0.02 * 1.2e8, (
        "the late branch is -1.2e8 by construction; reported %+.4e"
        % r["late"]["gamma"])
    assert "cannot be summarised" in r["reason"]


def test_m3_a_decay_in_a_LATER_window_is_not_an_early_rate():
    """Case (b), which no turn-over cut inside a window can catch: the growth and
    the long decay sit in DIFFERENT surviving windows, separated by a dip below
    the signal criterion, so the window that is finally readable contains no
    turn-over at all.  It is clean, it is long, its r2 is 1.0000 -- and it is
    still the late branch of a mode that grew, which is why the fitter also
    checks what the amplitude did BEFORE the window it fitted."""
    t = np.arange(2001) * 20e-12
    a = _two_branch(t, 1e-4, 3.0e8, -1.2e8, 0.5e-9)
    a = np.where((t > 3e-9) & (t < 4e-9), 1e-9, a)     # an interference null
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert r["status"] != GI.OK, (
        "the amplitude rose by %.2f e-foldings into the window t = %.2f-%.2f ns "
        "before decaying, and the fitter still returned gamma = %+.4e as the "
        "EARLY linear growth rate at r2 = %.4f"
        % (r.get("rise_into_window_efolds", float("nan")), r["t0"] * 1e9,
           r["t1"] * 1e9, r["gamma"], r["r2"]))
    assert r["status"] == GI.NOT_SUMMARISABLE and r["branch"] == "late", (
        "got %s / branch %s: %s" % (r["status"], r["branch"], r["reason"]))
    assert r["rise_into_window_efolds"] > 1.0
    assert "ROSE" in r["reason"]


def test_m3_a_genuine_sub_threshold_decay_is_still_measured():
    """Do not over-block: the bracket NEEDS measured negative rates.  A mode
    seeded at its largest amplitude and decaying from t = 0 has not risen into
    anything and must still return a number."""
    t = np.arange(751) * 20e-12
    for g in (-9.578e6, -5.0e7, -1.5e8):
        a = 1e-2 * np.exp(g * t)
        r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1,
                                  min_efold=0.1)
        assert r["status"] == GI.OK and r["gamma"] < 0, (
            "a clean sub-threshold decay at %+.3e 1/s must be MEASURED, not "
            "refused: %s / %s" % (g, r["status"], r["reason"]))
        assert r["rise_into_window_efolds"] == 0.0


# ------------------------------------------------------------------- M4 -----
def test_m4_campaign_scripts_cannot_reach_the_raw_fitter():
    """Enforced, not grepped: call SA.growth_fit FROM a campaign module's own
    namespace and require the refusal.  A new script in runs/ inherits this
    without anyone adding a test for it."""
    import G4_commensurate_onset as G4                            # noqa: PLC0415

    t = np.arange(50) * 20e-12
    a = 1e-4 * np.exp(3e8 * t)
    src = ("def probe(SA, t, a):\n"
           "    return SA.growth_fit(t, a)\n")
    ns = {}
    exec(compile(src, os.path.join(RUNS, "G4_commensurate_onset.py"), "exec"),
         ns)                          # a function whose file IS a campaign file
    try:
        got = ns["probe"](SA, t, a)
    except SA.CampaignFitterBypass:
        pass
    else:
        raise AssertionError(
            "a campaign module obtained Gamma = %.4e from saw_analysis."
            "growth_fit, which applies no signal criterion, no linearity cut "
            "and no turn-over cut and returns no status. That is a second "
            "measurand, and it is how the dead-bin defect survived in G4b and "
            "G9 after G4 was fixed." % got[0])
    # and the campaign's own map still works, from the one fitter
    res = G4._gamma_map(_wave(400, G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN,
                              G4.K_HALF, G4.F_SAW / 2,
                              1e-4 * np.exp(3e8 * np.arange(400)
                                            * G4.DT_REC_RUN)),
                        G4.BOX["dx"], G4.DT_REC_RUN, G4.F_SAW / 2)
    assert res[3][int(np.argmin(np.abs(res[0] - G4.K_HALF)))] == GI.OK


def test_m4_non_campaign_callers_are_not_blocked():
    """Do not over-block: the published-record analyses and the adversarial
    fixtures live outside runs/ and must keep working."""
    t = np.arange(50) * 20e-12
    g = SA.growth_fit(t, 1e-4 * np.exp(3e8 * t))[0]
    assert abs(g - 3e8) < 1e-3 * 3e8, g
    txt = open(os.path.join(HERE, "model_comparison.py"),
               encoding="utf-8").read()
    assert "growth_fit" in txt          # the caller that must stay unaffected
    out = subprocess.run([sys.executable, os.path.join(HERE,
                                                       "growth_refit_sim40.py")],
                         capture_output=True, text=True, cwd=HERE)
    assert out.returncode == 0, out.stderr[-800:]
    assert "2.9020e+08" in out.stdout or "2.902" in out.stdout, (
        "growth_refit_sim40.py must still reproduce the corrected 7e-5 rates")


def test_m4_one_fitter_is_the_only_one_in_the_campaign():
    """Structural backstop to the runtime one: no module in runs/ may define its
    own interval-selection-and-fit."""
    offenders = []
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".py"):
            continue
        txt = open(os.path.join(RUNS, fn), encoding="utf-8").read()
        for m in re.finditer(r"(?<![\w.])(SA|saw_analysis)\.growth_fit\(", txt):
            line = txt[:m.start()].count("\n") + 1
            if "allow_raw=True" not in txt[m.start():m.start() + 200]:
                offenders.append("%s:%d" % (fn, line))
    assert not offenders, (
        "campaign modules still call the raw fitter: %s" % ", ".join(offenders))


# ------------------------------------------------------------------- M5 -----
def test_m5_dead_bin_and_turned_over_bin_lose_the_maximum_in_both_scripts():
    """The dead-bin counterexample, hardened: the record also contains a bin
    that GREW and then collapsed, whose late slope is steeply negative, and the
    same assertion is made through G4's map AND through G4b's analyze() path."""
    import G4_commensurate_onset as G4                            # noqa: PLC0415
    import G4b_crosscheck_box as G4b                              # noqa: PLC0415

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN
    nt, f = 751, G4.F_SAW / 2
    t = np.arange(nt) * dt
    real = _wave(nt, nx, dx, dt, G4.K_HALF, f, 1e-4 * np.exp(3.0e8 * t))
    dead = _wave(nt, nx, dx, dt, 40 * G4.BOX["dk"], f,
                 1e-13 * np.exp(8.0e8 * t))
    # grows SLOWER than q/2 and then collapses: its early branch spans too few
    # e-foldings to read, and its late slope is steeply negative, so any fitter
    # that reaches for the late branch shows up immediately.
    turned = _wave(nt, nx, dx, dt, 20 * G4.BOX["dk"], f,
                   _two_branch(t, 1e-4, 1.0e8, -6.0e8, 5e-9))
    my = real + dead + turned
    for who, res in (("G4._gamma_map", G4._gamma_map(my, dx, dt, f)),
                     ("GI.gamma_map(G4b path)",
                      GI.gamma_map(my, dx, G4b.DT_REC, f,
                                   k_offband=2 * G4.Q))):
        k_ax, G, R, STAT, _, _, _, meta = res
        best = meta["best"]
        assert best["state"] == GI.OK, (
            "%s found no measurable positive-k rate at all" % who)
        assert abs(best["k_um"] - G4.K_HALF / 1e6) < 0.51 * G4.BOX["dk"] / 1e6, (
            "%s reports its maximum growth rate at k = %.4f um^-1 "
            "(Gamma = %.4e) instead of q/2 = %.4f um^-1; the dead bin is at "
            "%.4f and the collapsed bin at %.4f"
            % (who, best["k_um"], best["gamma"], G4.K_HALF / 1e6,
               40 * G4.BOX["dk"] / 1e6, 20 * G4.BOX["dk"] / 1e6))
        j_dead = int(np.argmin(np.abs(k_ax - 40 * G4.BOX["dk"])))
        assert STAT[j_dead] != GI.OK, (
            "%s measured a rate for a bin at the numerical floor" % who)
        j_turn = int(np.argmin(np.abs(k_ax - 20 * G4.BOX["dk"])))
        assert not (STAT[j_turn] == GI.OK and G[j_turn] < 0), (
            "%s reports Gamma = %+.4e for the bin that grew at +1.0e8 and then "
            "collapsed at -6.0e8" % (who, G[j_turn]))
        j_half = int(np.argmin(np.abs(k_ax - G4.K_HALF)))
        assert STAT[j_half] == GI.OK and abs(G[j_half] - 3.0e8) < 0.1e8, (
            "%s must still measure the real mode at q/2: %s / %+.4e"
            % (who, STAT[j_half], G[j_half]))


def test_m5_g4b_uses_the_shared_map():
    txt = open(os.path.join(RUNS, "G4b_crosscheck_box.py"),
               encoding="utf-8").read()
    assert "GI.gamma_map(" in txt, (
        "G4b.analyze() must obtain Gamma(k) from the shared map, not from its "
        "own loop")
    assert "gamma_qhalf_status" in txt, (
        "G4b must record the per-bin status, or its rows cannot say which "
        "numbers are measurements")


# ------------------------------------------------------------------- M6 -----
def test_m6_a_blocked_plan_hands_no_run_length_to_an_arbitrary_caller():
    """Not "the three known callers now check": THIS caller never checked, and
    still cannot get a number."""
    mat, eps = H.SIM40, 7e-5
    plan = H.plan_runtime(mat, eps, 0.2,
                          t_window=H.auto_window(H.gamma_estimate(mat, eps)),
                          t_min=40e-9)
    assert str(plan["status"]) in H.PLAN_BLOCKING, (
        "a seed at 0.2 with m_sat_level = 0.1 has no linear window; the plan's "
        "status is %r" % (plan["status"],))
    for key in ("t_run", "a_end"):
        try:
            v = plan[key]
        except H.PlanRefused:
            pass
        else:
            raise AssertionError(
                "the blocked plan handed back %s = %r, from which a caller "
                "computes nt = %d" % (key, v, int(float(v) / 20e-12) + 1))
    try:
        H.nt_records(plan, 20e-12, "an arbitrary caller")
    except H.PlanRefused:
        pass
    else:
        raise AssertionError("nt_records accepted a blocked plan")
    # diagnostics must stay readable, or the caller cannot explain itself
    assert plan["status"] and plan["note"] and np.isfinite(plan["t_linear"])


def test_m6_runnable_plans_still_give_a_run_length_and_a_record_count():
    """Do not over-block: every configuration the campaign actually declares
    must still plan."""
    import G4_commensurate_onset as G4                            # noqa: PLC0415

    seen = 0
    for arm, cfg in G4.ARMS.items():
        mat = cfg["mat"]
        B0 = H.kittel_field(3.0e9, mat["MS"])
        V = G4.BOX["NX"] * H.NY * G4.BOX["dx"] * H.CY * H.CZ
        ap = H.thermal_mode_amplitude(G4.K_HALF, 300.0, B0, mat["MS"],
                                      mat["AEX"], V)
        for eps in cfg["eps"]:
            g = H.gamma_estimate(mat, eps)
            w = H.auto_window(g)
            a, _ = H.choose_seed_amplitude(mat, eps, ap, t_window=w)
            plan = H.plan_runtime(mat, eps, a, t_window=w, t_min=40e-9)
            assert str(plan["status"]) in H.PLAN_OK, (
                "%s eps=%.1e is a DECLARED configuration and its plan is "
                "blocked: %s" % (arm, eps, plan["status"]))
            nt = H.nt_records(plan, G4.DT_REC_RUN, "%s eps=%.1e" % (arm, eps))
            assert nt > 1 and plan["t_run"] > 0
            seen += 1
    assert seen == sum(len(c["eps"]) for c in G4.ARMS.values())


def test_m6_no_caller_converts_a_run_length_by_hand():
    """The conversion that produced nt = -113 must exist in ONE place."""
    offenders = []
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        txt = open(os.path.join(RUNS, fn), encoding="utf-8").read()
        for m in re.finditer(r"int\(\s*plan\[.t_run.\]", txt):
            offenders.append("%s:%d" % (fn, txt[:m.start()].count("\n") + 1))
    assert not offenders, (
        "these call sites convert a planned run length to a record count "
        "themselves instead of using H.nt_records, so a nonpositive count is "
        "reachable there: %s" % ", ".join(offenders))


# ------------------------------------------------------------------- M7 -----
def _fit(status, gamma, branch="whole", r2=0.99, reason=""):
    return dict(status=status, gamma=gamma, branch=branch, r2=r2,
                measurable=status == GI.OK, reason=reason)


def test_m7_bracket_refuses_unreadable_and_late_points_but_accepts_real_ones():
    ok = GI.bracket_from_fits([GI.point_from_fit(3e-5, _fit(GI.OK, -9.58e6)),
                               GI.point_from_fit(7e-5, _fit(GI.OK, +3.56e6))])
    assert ok["status"] == GI.OK and ok["eps_threshold_interval"] == [3e-5, 7e-5]
    for bad_point, why in (
            (_fit(GI.HELD, float("nan")), "HELD"),
            (_fit(GI.NOT_SUMMARISABLE, float("nan")), "NOT_SUMMARISABLE"),
            (_fit(GI.NOT_SUMMARISABLE, -1.2e8, branch="late"),
             "a FINITE late-branch rate"),
            (_fit(GI.OK, -1.2e8, branch="late"), "a late branch called ok")):
        br = GI.bracket_from_fits(
            [GI.point_from_fit(3e-5, bad_point),
             GI.point_from_fit(7e-5, _fit(GI.OK, +3.56e6))])
        assert br["status"] == "NOT_EVALUABLE", (
            "a sweep whose negative point is %s was said to bracket a "
            "threshold: %r" % (why, br))
        assert br["eps_threshold_interval"] is None


def test_m7_bracket_will_not_take_a_bare_number():
    try:
        GI.point_from_fit(3e-5, -9.58e6)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "a bracket point was built from a bare float, which cannot say "
            "whether it is a measured rate or a post-turn-over damping")


def test_m7_gate_needs_the_declared_seeds_and_strains():
    import G4_commensurate_onset as G4                            # noqa: PLC0415

    def row(eps, seed, g, r2=0.95, st=GI.OK, done=True):
        return dict(arm="YIGlit", eps0=eps, rng_seed=seed, gamma_qhalf=g,
                    r2_qhalf=r2, gamma_qhalf_status=st, done=done,
                    artifact_state="PASS",  # explicitly certified synthetic input
                    file="synthetic_eps%.0e_seed%d" % (eps, seed))

    full = [row(e, s, g) for s in G4.RNG_SEEDS
            for e, g in zip(G4.ARMS["YIGlit"]["eps"],
                            (-9.58e6, +3.56e6, +1.34e7, +4.63e7))]
    assert G4.onset_gate(full, "YIGlit").state == "PASS", (
        "the complete, valid sweep must still PASS -- this fix must not "
        "over-block: %s" % G4.onset_gate(full, "YIGlit").reason)
    uncertified = [dict(r) for r in full]
    uncertified[0].pop("artifact_state")
    assert G4.onset_gate(uncertified, "YIGlit").state == "NOT_EVALUABLE"
    assert G4.onset_gate([r for r in full
                          if r["rng_seed"] == G4.RNG_SEEDS[0]],
                         "YIGlit").state != "PASS", "one seed of two passed"
    assert G4.onset_gate([r for r in full if r["eps0"] != 2e-4],
                         "YIGlit").state != "PASS", "a missing strain passed"
    part = [dict(r) for r in full]
    part[0]["done"] = False
    assert G4.onset_gate(part, "YIGlit").state != "PASS", (
        "an unfinished run supplied a gate point")
    held = [dict(r) for r in full]
    held[0].update(gamma_qhalf=float("nan"), gamma_qhalf_status=GI.HELD)
    g = G4.onset_gate(held, "YIGlit")
    assert g.state != "PASS" and "measured" in g.reason.lower(), (
        "an unreadable lowest-strain point still produced a bracket: %s"
        % g.reason)


# ------------------------------------------------------------------- M8 -----
def test_m8_cost_table_is_derived_and_matches_the_plan():
    import campaign_cost as CC                                    # noqa: PLC0415

    src = open(os.path.join(HERE, "campaign_cost.py"), encoding="utf-8").read()
    body = src[src.index('"""', src.index('"""') + 3):]
    lits = re.findall(r"\(\s*(?:[0-9.]+e-0[0-9]\s*,\s*){2,}", body)
    assert not lits, (
        "campaign_cost.py still hard-codes a strain sweep %s; the table must "
        "come from the live scripts" % lits)
    rows = CC.derive()
    tot = CC.totals(rows)
    want = CC.run_plan_total()
    assert want is not None, "RUN_PLAN section 5 total row could not be parsed"
    assert tot["configs"] == want["configs"], (
        "derived %d configurations against RUN_PLAN's %d"
        % (tot["configs"], want["configs"]))
    assert abs(tot["physics_us"] - want["physics_us"]) <= 0.05, (
        "derived %.2f us against RUN_PLAN's %.2f us"
        % (tot["physics_us"], want["physics_us"]))
    assert abs(tot["gpu_lo"] - want["gpu_lo"]) <= 1.0 and \
        abs(tot["gpu_hi"] - want["gpu_hi"]) <= 1.0, (
        "derived %.1f-%.1f GPU-hours against RUN_PLAN's %.1f-%.1f"
        % (tot["gpu_lo"], tot["gpu_hi"], want["gpu_lo"], want["gpu_hi"]))
    assert all(r["configs"] > 0 and r["physics_ns"] > 0 for r in rows)


def test_m7_no_campaign_gate_brackets_from_bare_numbers():
    """Structural backstop: a campaign gate may only build a bracket from fit
    results.  bracket_check with bare floats stays available for the sweep
    PREDICTIONS in the run plan and for the unit tests, but not for a gate."""
    offenders = []
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".py"):
            continue
        txt = open(os.path.join(RUNS, fn), encoding="utf-8").read()
        for m in re.finditer(r"GI\.bracket_check\(", txt):
            line = txt[:m.start()].count(chr(10)) + 1
            offenders.append("%s:%d" % (fn, line))
    assert not offenders, (
        "campaign modules call GI.bracket_check directly, so a point of unknown "
        "provenance can enter a threshold bracket: %s" % ", ".join(offenders))


def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    nfail = 0
    print("=" * 78)
    print("ROUND-2 mechanism counterexamples -- measurand / fitter / plan / "
          "bracket")
    print("=" * 78)
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            nfail += 1
            print("FAIL  %-*s  %s: %s" % (width, name, type(exc).__name__, exc))
            if os.environ.get("ADV_TRACE"):
                traceback.print_exc()
        else:
            print("pass  %-*s" % (width, name))
    print("-" * 78)
    print("%d passed, %d failed, %d total"
          % (len(tests) - nfail, nfail, len(tests)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
