"""Adversarial verification of the P1 / 8a-8e round (independent check, 2026-09-18).

The P1 round's own suite (test_p1_growth_and_seed.py, 15/15) was re-run and every
one of its 15 tests was confirmed to go RED when the corresponding fix is reverted
in a scratch copy, so those tests are load-bearing.  This file holds the
counterexamples that attack the SAME defects from a DIFFERENT angle and that the
fixed code does NOT survive.  Every test below FAILS against the tree as shipped
on 2026-09-18 11:57.  Nothing here is a physical result: each one asserts that a
wrong or unmeasurable quantity is not returned as a confident number.

  V1  fit_growth_interval / _gamma_map: the turn-over cut resolves a record that
      GROWS and then DAMPS in favour of the decay branch, returning the damping
      rate as "the growth rate" with status ok and r2 = 0.9998.  The pre-fix
      whole-record fit returned -5.539e7 with r2 = 0.4076, which the gate's own
      r2 > 0.9 clause rejected -- so the fix converts a rate the guard caught
      into a clean-looking wrong one.  REGRESSION introduced by the fix.
  V2  end-to-end consequence of V1: stage_analyze + onset_gate return
      GATE G4.3 PASS, "sign change of Gamma(q/2) bracketed in [3e-5, 7e-5]",
      built entirely on that mis-signed point.  That is the 8b deliverable.
  V3  G6: the legacy path the fix itself added (a_seed absent from the npz) maps
      every such file to the key "nan", so the auditor's pooling returns: two
      runs 50x apart in amplitude came back as one n = 2 bucket.
  V4  G6: an all-zero (aborted) pump-off control gives den.mean() = 0 and the
      row is returned with status "ok" and ratio_mean = inf.
  V5  k0_signal_check: an OFF-GRID finite-k mode with NO uniform component leaks
      5.9e-3 of the power into the k = 0 column, passes both stated criteria,
      and gate G4.2 says PASS -- the 8a hole from the other side.
  V6  plan_runtime returns a NEGATIVE t_run when the seed is past m_sat_level,
      and no caller reads plan["status"], so INSUFFICIENT_LINEAR_WINDOW blocks
      nothing.
  V7  campaign_cost.py still hardcodes the retired YIG sweep (7e-5 ... 3e-4) and
      prints 63 configs / 9.7 us / 67-88 GPU-h against RUN_PLAN's 66 / 10.3 us /
      71-94.  The plan's cost table is no longer derivable from the tree.
  V8  stage_analyze prints "[partial]" for a done = False record but stores no
      such field on the row, so an unfinished run is silently eligible to
      supply a gate point.

Run:  python test_adversarial_p1_verify.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
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


def _wave(nt, nx, dx, dt, k, f, env):
    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(nt) * dt
    return np.asarray(env)[:, None] * np.cos(k * x[None, :]
                                             - 2 * np.pi * f * t[:, None])


def _grow_then_damp(nt, dt, a0=1e-4, g_up=3.0e8, g_dn=-1.2e8, t_pk=4e-9):
    t = np.arange(nt) * dt
    return t, np.where(t <= t_pk, a0 * np.exp(g_up * t),
                       a0 * np.exp(g_up * t_pk) * np.exp(g_dn * (t - t_pk)))


# ------------------------------------------------------------------- V1 -----
def test_adv_fit_does_not_report_damping_as_the_growth_rate():
    t, a = _grow_then_damp(1501, 20e-12)
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert not (r["status"] == GI.OK and r["gamma"] < 0), (
        "a mode that grew at +3.000e+08 1/s for 4 ns and then damped at "
        "-1.200e+08 1/s was fitted on the DECAY branch and returned "
        "gamma = %+.4e 1/s with status=%s, r2 = %.4f, interval %.2f-%.2f ns. "
        "The turn-over cut picks the side of the peak with the longer run of "
        "surviving samples, so the longer a growing mode is observed after it "
        "turns over the more confidently its damping is reported as its growth "
        "rate. The pre-fix whole-record fit gave -5.539e+07 with r2 = 0.4076, "
        "which onset_gate's r2 > 0.9 clause rejected."
        % (r["gamma"], r["status"], r["r2"], r["t0"] * 1e9, r["t1"] * 1e9))


def test_adv_gamma_map_does_not_report_damping_at_qhalf():
    import G4_commensurate_onset as G4

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN
    t, env = _grow_then_damp(751, dt)
    my = _wave(751, nx, dx, dt, G4.K_HALF, G4.F_SAW / 2, env)
    k_ax, G, R, STAT = G4._gamma_map(my, dx, dt, G4.F_SAW / 2)[:4]
    j = int(np.argmin(np.abs(k_ax - G4.K_HALF)))
    assert not (STAT[j] == GI.OK and G[j] < 0), (
        "_gamma_map reports Gamma(q/2) = %+.4e 1/s, status=%s, r2 = %.4f for a "
        "mode whose true growth rate is +3.000e+08 1/s"
        % (G[j], STAT[j], R[j]))


# ------------------------------------------------------------------- V2 -----
def test_adv_onset_gate_does_not_pass_on_a_grow_then_damp_point():
    import G4_commensurate_onset as G4

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN
    f = G4.F_SAW / 2
    nt = 751
    t = np.arange(nt) * dt
    grow = _wave(nt, nx, dx, dt, G4.K_HALF, f, 1e-4 * np.exp(3.0e8 * t))
    _, env = _grow_then_damp(nt, dt)
    damp = _wave(nt, nx, dx, dt, G4.K_HALF, f, env)
    tmp = tempfile.mkdtemp(prefix="adv_v2_")
    old = G4.CKPT
    try:
        G4.CKPT = tmp
        for i, (my, eps) in enumerate(((grow, 7e-5), (damp, 3e-5))):
            np.savez(os.path.join(tmp, "G4_onset_%d.npz" % i),
                     my_xt=my.astype(np.float32), done=True, eps0=eps,
                     arm="YIGlit", rng_seed=20260917,
                     provenance=json.dumps(dict(schema="adversarial fixture")))
        buf = io.StringIO()
        with redirect_stdout(buf):
            G4.stage_analyze()
        out = buf.getvalue()
        st = str(np.load(os.path.join(tmp, "G4_summary.npz"),
                         allow_pickle=True)["gate_state"])
        assert st != "PASS", (
            "GATE G4.3 returned PASS and the threshold was declared bracketed "
            "on the strength of a point whose negative growth rate is the "
            "post-peak damping of a mode that grew:\n    %s"
            % "\n    ".join(ln.strip() for ln in out.splitlines()
                            if "GATE" in ln))
    finally:
        G4.CKPT = old
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- V3 -----
def _write_g6(path, my, T, eps, cls, a_seed=None, **overrides):
    kw = dict(my_xt=my.astype(np.float32), done=True, temperature=float(T),
              next_index=my.shape[0], nt_total=my.shape[0],
              eps0=float(eps), seed_class=str(cls), realisation=0,
              provenance=json.dumps(dict(schema="adversarial fixture")))
    if a_seed is not None:
        kw["a_seed"] = float(a_seed)
    kw.update(overrides)
    np.savez(path, **kw)


def test_adv_g6_legacy_files_without_a_seed_are_not_pooled():
    import G6_clean_seed as G6

    nt, nx, dx, dt = 256, G6.BOX["NX"], G6.BOX["dx"], G6.DT_REC
    k, f = G6.K_HALF, G6.F_SAW / 2
    tmp = tempfile.mkdtemp(prefix="adv_v3_")
    old = G6.CKPT
    try:
        G6.CKPT = tmp
        one = np.ones(nt)
        for fn, amp, eps in (("G6_T0_phys_eps1e-04_r0.npz", 2e-3, G6.EPS_WORK),
                             ("G6_T0_phys_eps1e-04_r1.npz", 4e-5, G6.EPS_WORK),
                             ("G6_T0_phys_eps0e+00_r0.npz", 1e-3, 0.0)):
            _write_g6(os.path.join(tmp, fn),
                      _wave(nt, nx, dx, dt, k, f, amp * one), 0.0, eps,
                      "T0_phys")                      # NO a_seed: legacy npz
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = G6.analyze()
        bad = [r for r in rows if r["status"] == "ok" and r["n"] > 1]
        assert not bad, (
            "two pumped runs 50x apart in amplitude (2e-3 and 4e-5), written "
            "without the a_seed field that the fix keys on, were pooled as one "
            "condition: n = %d, ratio = %.4f +- %.4f. The amplitude half of the "
            "key is formatted with %%.12e, which is the same string 'nan' for "
            "every legacy file, so the collision happens on exactly the path "
            "the fix added (_class_from_name and the 'a_seed in d' fallback)."
            % (bad[0]["n"], bad[0]["ratio_mean"], bad[0]["ratio_sd"]))
    finally:
        G6.CKPT = old
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- V4 -----
def test_adv_g6_zero_control_is_not_a_ratio():
    import G6_clean_seed as G6

    nt, nx, dx, dt = 256, G6.BOX["NX"], G6.BOX["dx"], G6.DT_REC
    k, f = G6.K_HALF, G6.F_SAW / 2
    tmp = tempfile.mkdtemp(prefix="adv_v4_")
    old = G6.CKPT
    try:
        G6.CKPT = tmp
        _write_g6(os.path.join(tmp, "G6_T0_phys_eps1e-04_r0.npz"),
                  _wave(nt, nx, dx, dt, k, f, 2e-3 * np.ones(nt)),
                  0.0, G6.EPS_WORK, "T0_phys", 1e-3)
        _write_g6(os.path.join(tmp, "G6_T0_phys_eps0e+00_r0.npz"),
                  np.zeros((nt, nx)), 0.0, 0.0, "T0_phys", 1e-3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = G6.analyze()
        bad = [r for r in rows
               if r["status"] == "ok" and not np.isfinite(r["ratio_mean"])]
        assert not bad, (
            "an all-zero pump-off control (an aborted run that still wrote its "
            "array) was accepted: status='ok', ratio_mean=%r. A control with no "
            "band power is not a control; the correct return is "
            "NOT_DETERMINABLE." % (bad[0]["ratio_mean"],))
    finally:
        G6.CKPT = old
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- V5 -----
def test_adv_k0_check_rejects_leakage_from_an_offgrid_mode():
    import G4_commensurate_onset as G4

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_DISP
    nt = 2001
    mat = H.YIG_LIT
    B0 = H.kittel_field(3.0e9, mat["MS"])
    dk = 2 * np.pi / (nx * dx)
    k = 6.5 * dk                                  # deliberately OFF grid
    ex = 2 * mat["AEX"] * k ** 2 / mat["MS"]
    fk = H.GAMMA * np.sqrt((B0 + ex) * (B0 + H.MU0 * mat["MS"] + ex)) \
        / (2 * np.pi)
    my = _wave(nt, nx, dx, dt, k, fk, 1e-4 * np.ones(nt))   # NO uniform tone
    k_ax, f_ax, M = SA.spectrum(my, dx, dt, window="hann")
    chk = G4.k0_signal_check(k_ax, f_ax, M, 2 * G4.Q)
    f0 = G4._fpeak(f_ax, M[:, int(np.argmin(np.abs(k_ax)))])[0]
    fK = H.kittel_freq(B0, mat["MS"])
    g = G4.disp_gate(resid_Hz=0.0, f_k0=f0, f_kittel=fK,
                     df_bin=1.0 / (nt * dt), k0=chk)
    assert chk["status"] != "ok" or g.state != "PASS", (
        "a single OFF-GRID finite-k mode, with no uniform component at all, "
        "leaks rel_power = %.3e (criterion %.0e) and offband SNR = %.3e "
        "(criterion %.0f) into the k = 0 column; the check says '%s' and gate "
        "G4.2 says %s on f(0) = %.4f GHz vs Kittel %.4f GHz. The two stated "
        "criteria measure power, not whether the column is spectrally distinct "
        "from leakage of the finite-k modes."
        % (chk["rel_power"], chk["criteria"]["rel_min"], chk["snr_offband"],
           chk["criteria"]["snr_offband"], chk["status"], g.state,
           f0 / 1e9, fK / 1e9))
    assert chk["status"] == "NOT_EVALUABLE"
    assert "origin" in chk["reason"]


# ------------------------------------------------------------------- V6 -----
def test_adv_planner_never_returns_a_nonpositive_runtime():
    """UPDATED 2026-09-18 (round 2), and the change is recorded rather than made
    quietly: this counterexample was written as `assert p["t_run"] > 0` against a
    planner that returned a plain dict.  The round-2 fix does not give a seed past
    the saturation level a positive run length -- there is none -- it REFUSES:
    `plan["t_run"]` raises H.PlanRefused on a blocking status, so no caller,
    including one written before the fix, can obtain a run length from a plan that
    cannot be run.  A refusal is strictly stronger than the property asserted
    here (nothing reaches a caller at all), so the test accepts either, and fails
    on the original defect -- a nonpositive number handed back silently.  The
    positive-control half, that runnable plans still hand over a run length, is
    test_m6_runnable_plans_still_give_a_run_length_and_a_record_count."""
    mat, eps = H.SIM40, 7e-5
    p = H.plan_runtime(mat, eps, 0.2, t_window=H.auto_window(
        H.gamma_estimate(mat, eps)), t_min=40e-9)
    assert str(p["status"]) in H.PLAN_BLOCKING, (
        "a seed of 0.2 against m_sat_level = 0.1 has no linear window at all; "
        "the plan's status is %r" % (p["status"],))
    try:
        t_run = p["t_run"]
    except H.PlanRefused:
        return                          # refused: the strictly stronger outcome
    assert t_run > 0, (
        "a seed above m_sat_level gives t_linear = %+.4e s and the planner "
        "returns t_run = %+.4e s, from which the callers compute "
        "nt = int(t_run/dt_rec)+1 = %d. status is %r but no caller reads it."
        % (p["t_linear"], t_run, int(t_run / 20e-12) + 1, p["status"]))


def test_adv_planner_status_is_consumed_by_its_callers():
    unread = []
    for f in ("runs/G4_commensurate_onset.py", "runs/G6_clean_seed.py",
              "runs/G4b_crosscheck_box.py"):
        txt = open(os.path.join(HERE, f), encoding="utf-8").read()
        if "plan_runtime(" not in txt:
            continue
        if not ('plan["status"]' in txt or "plan['status']" in txt
                or "INSUFFICIENT_LINEAR_WINDOW" in txt):
            unread.append(f)
    assert not unread, (
        "plan_runtime sets status = INSUFFICIENT_LINEAR_WINDOW, documented as "
        "blocking a growth-rate verdict, but these callers take t_run and never "
        "look at it: %s. A status string nothing reads is not a block."
        % ", ".join(unread))


# ------------------------------------------------------------------- V7 -----
def test_adv_campaign_cost_uses_the_shipped_sweeps():
    """UPDATED 2026-09-18 (round 2), and the change is recorded rather than made
    quietly: this counterexample looked for a HARD-CODED literature-YIG sweep in
    campaign_cost.py and required it to equal G4.ARMS['YIGlit']['eps'].  The
    round-2 fix removes the hard-coded sweep instead of updating it -- the script
    now derives every row from G4.ARMS, G4b.VARIANTS, G4b/G4.RNG_SEEDS,
    G6.N_THERMAL_REALISATIONS and H.plan_runtime -- so the string this test
    searched for no longer exists.  The property the test was protecting is that
    RUN_PLAN's cost table can be reproduced from the tree, and that is what is
    asserted now, together with the absence of any hard-coded sweep.  Strictly
    stronger: the old form passed as soon as one literal matched, and would pass
    again the next time the sweep changed in only one of the two places."""
    import G4_commensurate_onset as G4
    import campaign_cost as CC

    txt = open(os.path.join(HERE, "campaign_cost.py"), encoding="utf-8").read()
    body = txt[txt.index('"""', txt.index('"""') + 3):]
    lits = re.findall(r"\(\s*(?:[0-9.]+e-0[0-9]\s*,\s*){2,}", body)
    assert not lits, (
        "campaign_cost.py still hard-codes a strain sweep %s, so its table can "
        "drift from G4.ARMS again" % lits)
    want = CC.run_plan_total()
    assert want is not None, "could not parse RUN_PLAN section 5's total row"
    tot = CC.totals(CC.derive())
    assert (tot["configs"] == want["configs"]
            and abs(tot["physics_us"] - want["physics_us"]) <= 0.05
            and abs(tot["gpu_lo"] - want["gpu_lo"]) <= 1.0
            and abs(tot["gpu_hi"] - want["gpu_hi"]) <= 1.0), (
        "the cost table derived from the live scripts is %d configs / %.2f us / "
        "%.1f-%.1f GPU-hours against RUN_PLAN section 5's %d / %.2f us / "
        "%.1f-%.1f, so the plan's cost table cannot be reproduced from the tree"
        % (tot["configs"], tot["physics_us"], tot["gpu_lo"], tot["gpu_hi"],
           want["configs"], want["physics_us"], want["gpu_lo"], want["gpu_hi"]))
    assert tuple(G4.ARMS["YIGlit"]["eps"])       # the live sweep is what was used


# ------------------------------------------------------------------- V8 -----
def test_adv_partial_runs_are_flagged_on_the_row():
    import G4_commensurate_onset as G4

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN
    nt = 751
    t = np.arange(nt) * dt
    my = _wave(nt, nx, dx, dt, G4.K_HALF, G4.F_SAW / 2,
               1e-4 * np.exp(3.0e8 * t))
    tmp = tempfile.mkdtemp(prefix="adv_v8_")
    old = G4.CKPT
    try:
        G4.CKPT = tmp
        np.savez(os.path.join(tmp, "G4_onset_0.npz"),
                 my_xt=my.astype(np.float32), done=False, eps0=7e-5,
                 arm="YIGlit", rng_seed=20260917,
                 provenance=json.dumps(dict(schema="adversarial fixture")))
        buf = io.StringIO()
        with redirect_stdout(buf):
            G4.stage_analyze()
        rows = json.loads(str(np.load(os.path.join(tmp, "G4_summary.npz"),
                                      allow_pickle=True)["rows"]))
        flags = [k for k in rows[0] if "done" in k or "partial" in k]
        assert flags, (
            "stage_analyze printed the [partial] ... analysed anyway, flagged "
            "line but the row it stored carries no done/partial field, so "
            "onset_gate cannot tell an unfinished run from a finished one: "
            "row keys = %s" % sorted(rows[0]))
    finally:
        G4.CKPT = old
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- V9 -----
def test_adv_g4b_dead_bin_defect_is_still_open():
    """The P1-2 dead-bin counterexample, replayed through G4b's analyze() code
    path (runs/G4b_crosscheck_box.py:213-221).  Disclosed by the P1 round as not
    fixed; nothing pins it, so it is recorded here."""
    import G4b_crosscheck_box as G4b
    import G4_commensurate_onset as G4

    txt = open(os.path.join(RUNS, "G4b_crosscheck_box.py"),
               encoding="utf-8").read()
    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4b.DT_REC
    nt = 751
    f = H.BOX_PRIMARY["f_saw"] / 2
    t = np.arange(nt) * dt
    k_dead = 40 * G4.BOX["dk"]
    my = _wave(nt, nx, dx, dt, G4.K_HALF, f, 1e-4 * np.exp(3.0e8 * t)) \
        + _wave(nt, nx, dx, dt, k_dead, f, 1e-13 * np.exp(8.0e8 * t))
    seg = max(64, nt // 8)
    stride = max(1, seg // 4)
    k_ax, f_ax, Ms_, t0s = SA.segment_spectra(my, dx, dt, seg,
                                              1 + (nt - seg) // stride,
                                              stride=stride)
    jf, _ = SA.f_slice(Ms_[0], f_ax, f)
    A = np.abs(Ms_[:, jf, :])
    G = np.array([SA.growth_fit(t0s + seg * dt / 2, A[:, j])[0]
                  for j in range(A.shape[1])])
    pos = k_ax > 0
    jb = int(np.where(pos)[0][np.nanargmax(G[pos])])
    assert "growth_interval" in txt, (
        "G4b.analyze() still uses the bare SA.growth_fit with argmax over "
        "positive k and no signal test, so the P1-2 dead-bin counterexample "
        "reproduces there verbatim: gamma_max is reported at k = %.4f um^-1 "
        "(peak amplitude 1e-13) instead of q/2 = %.4f um^-1, "
        "Gamma = %.4e vs %.4e 1/s. G4 was routed through growth_interval; G4b "
        "was not."
        % (k_ax[jb] / 1e6, G4.K_HALF / 1e6, G[jb],
           G[int(np.argmin(np.abs(k_ax - G4.K_HALF)))]))


# ------------------------------------------------------------------ V10 -----
def test_adv_onset_gate_both_seeds_clause_is_not_vacuous():
    """G4.3a is `len(seeds_grow) == len(seeds)`, where `seeds` is taken from the
    rows PRESENT.  With one seed's runs missing the clause is satisfied by one
    seed, so a gate whose stated requirement is "for both rng seeds" passes on
    half the evidence.  Absence must block, not satisfy."""
    import G4_commensurate_onset as G4

    def row(eps, seed, g, r2):
        return dict(arm="YIGlit", eps0=eps, rng_seed=seed, gamma_qhalf=g,
                    r2_qhalf=r2, gamma_qhalf_status=GI.OK)

    one = [row(3e-5, 20260917, -9.58e6, 0.95),
           row(7e-5, 20260917, +3.56e6, 0.95)]
    g = G4.onset_gate(one, "YIGlit")
    assert g.state != "PASS", (
        "gate G4.3 returned PASS on rows from ONE seed (%s) while G4.RNG_SEEDS "
        "declares %s; the clause reads len(seeds_grow) == len(seeds) over the "
        "seeds present, so a missing or unanalysed seed satisfies "
        "'for both rng seeds' instead of blocking. Reason given: %s"
        % (sorted({r["rng_seed"] for r in one}), G4.RNG_SEEDS, g.reason))


def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    nfail = 0
    print("=" * 78)
    print("ADVERSARIAL verification of the P1 / 8a-8e round -- 2026-09-18")
    print("=" * 78)
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
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
