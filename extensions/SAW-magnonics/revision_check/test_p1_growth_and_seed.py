"""Regression suite for the P1 defects and the RUN_PLAN inconsistencies of the
independent audit of 2026-09-18 (section 8).

Every test below encodes a COUNTEREXAMPLE the auditor actually fed to the
shipped functions.  Each one was run against the unfixed code first and FAILED;
the failing output is quoted in the handover note next to the fixed output.  No
test here asserts a physical result: they assert that a broken verdict is no
longer returned, and that an unmeasurable quantity is HELD instead of guessed.

  P1-1  G6_clean_seed.analyze pooled two seed classes (T0_phys, T0_small) whose
        amplitudes differ by 100x as if they were n = 2 repeats of one
        condition, and divided by the MEAN of their two different controls:
        true power gains 4 and 16 came back as "4.0012 +- 5.6540, n = 2".
  P1-2  the runtime planner let a 40 ns floor override the 20.805 ns linear
        window (predicted end amplitude 32.9, i.e. a run planned to be linear
        that cannot be), and the growth fit took the largest slope over the
        whole record with no saturation cut and no signal requirement.
  8a    the G4 dispersion seed injects no k = 0 component, so the "k = 0 vs
        Kittel" check was peak-picking on a residual ~1e-15 of the main
        component -- and the gate did not require the check at all.
  8b    the YIG strain sweep started ABOVE the plan's own predicted threshold,
        so it could not bracket a sign change of Gamma.
  8c-e  RUN_PLAN.md statements: the pi-phase test is not a decisive gate; the
        new box changes wavenumber AND seed AND B0 together; matched runs and
        re-tuned runs are separated.

Run:  python test_p1_growth_and_seed.py        (prints a report, exit 1 on any
      failure)                                  python -m pytest also works.
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

DATA = os.path.join(os.path.dirname(HERE), "data")
SIM40 = os.path.join(DATA, "sim40_checkpoints")


# ---------------------------------------------------------------- fixtures --
def synth_travelling(nt, nx, dx, dt, k, f, amp, growth=0.0, phase=0.0):
    """One travelling wave m_y(x,t) = amp e^{growth t} cos(k x - 2 pi f t + ph).

    A test fixture, not a simulation result.
    """
    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(nt) * dt
    env = amp * np.exp(growth * t)
    return env[:, None] * np.cos(k * x[None, :] - 2 * np.pi * f * t[:, None]
                                 + phase)


def write_g6_npz(path, my, temperature, eps0, a_seed, seed_class, realisation=0):
    np.savez(path, my_xt=my.astype(np.float32), done=True,
             next_index=my.shape[0], nt_total=my.shape[0],
             temperature=float(temperature), eps0=float(eps0),
             a_seed=float(a_seed), seed_class=str(seed_class),
             realisation=int(realisation), rng_seed_numpy=12345,
             provenance=json.dumps(dict(schema="test fixture")))


# ============================================================ P1-1: G6 ======
def test_p1_1_g6_never_pools_seed_classes():
    """Auditor's counterexample: two seed classes, amplitude ratio 100, TRUE
    band/control power gains 4 and 16.  The shipped analyze() returned
    '4.0012 +- 5.6540, n = 2'.  It must instead report the two classes
    separately, each against ITS OWN pump-off control."""
    import G6_clean_seed as G6

    nt, nx, dx, dt = 256, G6.BOX["NX"], G6.BOX["dx"], G6.DT_REC
    k, f = G6.K_HALF, G6.F_SAW / 2
    tmp = tempfile.mkdtemp(prefix="g6_p1_1_")
    old_ckpt = G6.CKPT
    try:
        G6.CKPT = tmp
        # class A ("physical" amplitude): control 1e-3, pumped 2e-3 -> gain 4
        # class B ("small" amplitude):    control 1e-5, pumped 4e-5 -> gain 16
        for cls, a_ctrl, a_pump, gain in (("T0_phys", 1e-3, 2e-3, 4.0),
                                          ("T0_small", 1e-5, 4e-5, 16.0)):
            for eps, amp in ((G6.EPS_WORK, a_pump), (0.0, a_ctrl)):
                my = synth_travelling(nt, nx, dx, dt, k, f, amp)
                write_g6_npz(os.path.join(
                    tmp, "G6_%s_eps%.0e_r0.npz" % (cls, eps)), my,
                    0.0, eps, a_ctrl, cls)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = G6.analyze()
        out = buf.getvalue()

        pooled = [ln for ln in out.splitlines() if re.search(r"n=\s*2", ln)]
        assert not pooled, ("two seed classes were pooled as n = 2: %r"
                            % (pooled[0].strip(),))
        assert rows is not None, ("analyze() returned nothing to check; it must "
                                 "return the per-class rows")
        got = {}
        for r in rows:
            key = (r.get("seed_class"), float(r.get("a_seed", np.nan)))
            got[key] = r
        assert len(rows) == 2, ("expected one row per seed class, got %d: %r"
                                % (len(rows), rows))
        for cls, a_ctrl, gain in (("T0_phys", 1e-3, 4.0),
                                  ("T0_small", 1e-5, 16.0)):
            r = got.get((cls, a_ctrl))
            assert r is not None, ("no row for seed class %s at a_seed=%.0e; "
                                   "rows=%r" % (cls, a_ctrl, rows))
            assert r["n"] == 1, ("class %s must be n = 1, got n = %s"
                                 % (cls, r["n"]))
            assert abs(r["ratio_mean"] - gain) < 1e-6 * gain, (
                "class %s: band/control = %.6f, true gain = %.1f"
                % (cls, r["ratio_mean"], gain))
    finally:
        G6.CKPT = old_ckpt
        shutil.rmtree(tmp, ignore_errors=True)


def test_p1_1_g6_control_must_match_amplitude():
    """A class whose pump-off control was run at a DIFFERENT seed amplitude is
    not a control for it: the ratio must be NOT_DETERMINABLE, not a number."""
    import G6_clean_seed as G6

    nt, nx, dx, dt = 256, G6.BOX["NX"], G6.BOX["dx"], G6.DT_REC
    k, f = G6.K_HALF, G6.F_SAW / 2
    tmp = tempfile.mkdtemp(prefix="g6_p1_1b_")
    old_ckpt = G6.CKPT
    try:
        G6.CKPT = tmp
        my = synth_travelling(nt, nx, dx, dt, k, f, 2e-3)
        write_g6_npz(os.path.join(tmp, "G6_T0_phys_eps1e-04_r0.npz"), my,
                     0.0, 1e-4, 1e-3, "T0_phys")
        my0 = synth_travelling(nt, nx, dx, dt, k, f, 1e-5)   # wrong amplitude
        write_g6_npz(os.path.join(tmp, "G6_T0_phys_eps0e+00_r0.npz"), my0,
                     0.0, 0.0, 1e-5, "T0_phys")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = G6.analyze()
        out = buf.getvalue()
        assert rows is not None, "analyze() returned nothing to check"
        assert rows == [] or all(r.get("status") != "ok" for r in rows), (
            "a control at a different seed amplitude was accepted as a "
            "control: rows=%r" % (rows,))
        assert "NOT_DETERMINABLE" in out, (
            "mismatched control must be reported as NOT_DETERMINABLE; got:\n%s"
            % out)
    finally:
        G6.CKPT = old_ckpt
        shutil.rmtree(tmp, ignore_errors=True)


# ====================================== P1-2: planner and growth fit ========
def test_p1_2_planner_contains_a_readable_early_linear_interval():
    """Updated on resume: the user's requirement concerns the EARLY fit, not
    whether acquisition stops before saturation. Check the planned signal with
    the actual fitter, and also retain a later saturated tail. The old assertion
    T_run <= t_linear conflated acquisition duration with the fit interval.
    Invalid/nonpositive runtimes remain covered by the adversarial plan test.
    This is an analytic regression fixture, not simulated material evidence.
    """
    mat = H.SIM40
    eps = 7e-5
    g = H.gamma_estimate(mat, eps)
    win = H.auto_window(g)
    a_phys = 9.8434e-3                      # RUN_PLAN sec.2, n=12 box, 300 K
    a_seed, _ = H.choose_seed_amplitude(mat, eps, a_phys, t_window=win)
    plan = H.plan_runtime(mat, eps, a_seed, t_window=win, t_min=40e-9)
    H.require_plan(plan, "early growth regression")
    dt = 20e-12
    nt = H.nt_records(plan, dt, "early growth regression")
    for n in (nt, max(nt, int(40e-9 / dt) + 1)):
        t = np.arange(n) * dt
        a = np.minimum(a_seed * np.exp(g * t), 0.1)
        fit = GI.fit_growth_interval(t, a, sat_level=0.1)
        assert fit["status"] == GI.OK, fit["reason"]
        assert abs(fit["gamma"] / g - 1) < 1e-6
        assert fit["t1"] < plan["t_linear"]
        assert np.all(a[fit["i0"]:fit["i1"]] <= 0.05)


def test_p1_2_planner_negative_gamma_is_not_t_max():
    """A predicted-decaying point (literature YIG below threshold) must be run
    long enough to MEASURE the decay, 1.5/|Gamma|, not padded to t_max."""
    mat = H.YIG_LIT
    eps = 3e-5
    g = H.gamma_estimate(mat, eps)
    assert g < 0, "3e-5 must be a predicted-negative point, got %.3e" % g
    plan = H.plan_runtime(mat, eps, 9.8434e-3, t_window=H.auto_window(g),
                          t_min=40e-9)
    want = 1.5 / abs(g)
    assert abs(plan["t_run"] - want) < 0.05 * want, (
        "predicted Gamma = %.3e 1/s needs T_run ~ 1.5/|Gamma| = %.1f ns to be "
        "measurable; planner returned %.1f ns"
        % (g, want * 1e9, plan["t_run"] * 1e9))


def test_p1_2_fit_excludes_saturation():
    """exp growth for 28.4 ns, then a hard ceiling.  The whole-record fit
    underestimates Gamma; the explicit interval must recover it."""
    dt = 20e-12
    t = np.arange(2001) * dt
    gamma_true = 3.0e8
    a = np.minimum(1e-5 * np.exp(gamma_true * t), 0.05)
    whole = SA.growth_fit(t, a)[0]
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1, sat_frac=0.5)
    assert r["status"] == "ok", r["reason"]
    assert abs(r["gamma"] - gamma_true) < 1e-3 * gamma_true, (
        "interval fit gave %.4e, true %.4e" % (r["gamma"], gamma_true))
    assert abs(whole - gamma_true) > 0.2 * gamma_true, (
        "the whole-record fit was supposed to be biased here; it gave %.4e"
        % whole)
    assert r["criteria"]["sat_level"] == 0.1 and r["criteria"]["snr_amp"] == 4.0


def test_p1_2_fit_holds_when_no_interval():
    """A series that never rises above the stated floor must return a HELD
    verdict, not a number."""
    rng = np.random.default_rng(7)
    t = np.arange(751) * 20e-12
    a = 3e-9 * (1 + 0.1 * rng.standard_normal(t.size))       # far below 4e-7
    r = GI.fit_growth_interval(t, np.abs(a), floor=1e-9)
    assert r["status"] == GI.HELD and not np.isfinite(r["gamma"]), (
        "expected HELD/nan, got %r" % (r,))
    flat = 1e-3 * (1 + 1e-6 * rng.standard_normal(t.size))   # loud but flat
    r2 = GI.fit_growth_interval(t, flat, floor=1e-9, sat_level=0.1)
    assert r2["status"] == GI.HELD and not np.isfinite(r2["gamma"]), (
        "a flat series spans no e-folding and must be HELD, got %r" % (r2,))


def test_p1_2_fit_measures_decay_not_only_growth():
    """The bracket of 8b needs a MEASURED negative rate, so the interval rules
    must not throw a decaying series away: a turn-over cut aimed at saturation
    must not fire on the first sample of a decay."""
    t = np.arange(751) * 20e-12
    gamma_true = -9.578e6                      # YIG_LIT at eps_0 = 3e-5
    a = 1e-2 * np.exp(gamma_true * t)
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1, min_efold=0.1)
    assert r["status"] == GI.OK, (
        "a clean decay was not fitted: %s" % r["reason"])
    assert abs(r["gamma"] - gamma_true) < 1e-3 * abs(gamma_true), (
        "decay fit gave %.4e, true %.4e" % (r["gamma"], gamma_true))


def test_p1_2_fit_reports_the_early_branch_of_a_grow_then_damp_record():
    """THE MIXED CASE.  Added 2026-09-18 (round 2): this suite tested pure growth
    and pure decay and nothing in between, and the interval rule it was written
    for classified a record from its OVERALL slope, so a mode that grew at
    +3.0e8 1/s for 4 ns and then damped at -1.2e8 came back as
    gamma = -1.2000e+08, status = ok, r2 = 1.0000 -- the damping reported as the
    growth rate, with more confidence the longer the mode was watched after it
    turned over.

    The measurand is the EARLY linear rate (growth_interval.MEASURAND,
    pre-registered in PREREGISTRATION.md section 9), so the answer here is
    +3.0e8, with the late branch reported separately and never in its place."""
    t = np.arange(1501) * 20e-12
    a = np.where(t <= 4e-9, 1e-4 * np.exp(3.0e8 * t),
                 1e-4 * np.exp(1.2) * np.exp(-1.2e8 * (t - 4e-9)))
    r = GI.fit_growth_interval(t, a, floor=1e-9, sat_level=0.1)
    assert r["status"] == GI.OK, r["reason"]
    assert r["gamma"] > 0 and abs(r["gamma"] - 3.0e8) < 1e-3 * 3.0e8, (
        "the early branch is +3.0000e+08 1/s; got %+.4e with status %s and "
        "r2 = %.4f on t = %.2f-%.2f ns"
        % (r["gamma"], r["status"], r["r2"], r["t0"] * 1e9, r["t1"] * 1e9))
    assert r["late"] is not None and abs(r["late"]["gamma"] + 1.2e8) \
        < 1e-3 * 1.2e8, (
        "the late branch must be reported, as -1.2000e+08: %r" % (r["late"],))
    assert r["turnover"] is not None and r["branch"] == "early"
    # and when the early branch cannot be read, there is NO rate at all
    a2 = np.where(t <= 0.5e-9, 1e-4 * np.exp(3.0e8 * t),
                  1e-4 * np.exp(0.15) * np.exp(-1.2e8 * (t - 0.5e-9)))
    r2 = GI.fit_growth_interval(t, a2, floor=1e-9, sat_level=0.1)
    assert r2["status"] == GI.NOT_SUMMARISABLE \
        and not np.isfinite(r2["gamma"]), (
        "a record whose only readable branch is the LATE one must return no "
        "number: %s / %r" % (r2["status"], r2["gamma"]))


def test_p1_2_fit_holds_when_interval_is_not_exponential():
    """Found by re-fitting the real arrays (growth_refit_sim40.py): at
    eps_0 = 5e-5 bin 5 the stated interval spans 1.39 e-foldings but the series
    on it is not exponential -- r^2 = 0.179 -- and a rate of -1.91e8 1/s was
    returned anyway.  An interval the data do not follow defines no rate."""
    m, dt, _ = (lambda d: (d["my_xt"].astype(float), float(d["DT_REC"]),
                           float(d["eps0"])))(
        np.load(os.path.join(SIM40, "sim40_full_2fK_eps5e-05.npz"),
                allow_pickle=True))
    t = np.arange(m.shape[0]) * dt
    X = np.fft.fft(m - m.mean(axis=1, keepdims=True), axis=1) / m.shape[1]
    floor = float(np.median(np.abs(X[:, 12:22])))
    lin = np.abs(m).max(axis=1) < 0.1
    r = GI.fit_growth_interval(t, np.abs(X[:, 5]), floor=floor, sat_level=0.1,
                               linear_mask=lin, min_efold=1.0, min_points=5)
    assert r["status"] == GI.HELD, (
        "reported Gamma = %.4e 1/s on an interval with r^2 = %.3f"
        % (r["gamma"], r["r2"]))
    assert r["criteria"].get("min_r2") is not None, (
        "the r^2 adequacy criterion must be stated in the returned criteria")
    # and the published 7e-5 fit must still pass it
    m7, dt7, _ = (lambda d: (d["my_xt"].astype(float), float(d["DT_REC"]),
                             float(d["eps0"])))(
        np.load(os.path.join(SIM40, "sim40_full_2fK_eps7e-05.npz"),
                allow_pickle=True))
    t7 = np.arange(m7.shape[0]) * dt7
    X7 = np.fft.fft(m7 - m7.mean(axis=1, keepdims=True), axis=1) / m7.shape[1]
    r7 = GI.fit_growth_interval(
        t7, np.abs(X7[:, 5]), floor=float(np.median(np.abs(X7[:, 12:22]))),
        sat_level=0.1, linear_mask=np.abs(m7).max(axis=1) < 0.1,
        min_efold=1.0, min_points=5)
    assert r7["status"] == GI.OK and abs(r7["gamma"] - 2.9e8) < 0.1e8, (
        "the eps_0 = 7e-5 bin-5 rate must survive: %r" % (r7,))


def test_p1_2_gamma_map_ignores_dead_bins():
    """G4's Gamma(k) map with a floor-level bin whose slope is the steepest in
    the record.  The reported maximum must not be that bin."""
    import G4_commensurate_onset as G4

    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_RUN
    nt = 751
    f = G4.F_SAW / 2
    real = synth_travelling(nt, nx, dx, dt, G4.K_HALF, f, 1e-4, growth=3.0e8)
    k_dead = 40 * G4.BOX["dk"]
    dead = synth_travelling(nt, nx, dx, dt, k_dead, f, 1e-13, growth=8.0e8)
    my = real + dead + 1e-12 * np.random.default_rng(3).standard_normal((nt, nx))

    # (i) the counterexample, evaluated the way the shipped code did it:
    #     whole-record slope per bin, maximum over positive k, no signal test.
    seg = max(64, nt // 8)
    stride = max(1, seg // 4)
    kk, ff, MM, t0s = SA.segment_spectra(my, dx, dt, seg,
                                         1 + (nt - seg) // stride, stride=stride)
    jf, _ = SA.f_slice(MM[0], ff, f)
    AA = np.abs(MM[:, jf, :])
    G_old = np.array([SA.growth_fit(t0s + seg * dt / 2, AA[:, j])[0]
                      for j in range(AA.shape[1])])
    pos = kk > 0
    jb_old = int(np.where(pos)[0][np.nanargmax(G_old[pos])])
    assert abs(kk[jb_old] - k_dead) < 0.51 * G4.BOX["dk"], (
        "fixture broken: the dead bin was supposed to win the old argmax, the "
        "winner is at %.3f um^-1" % (kk[jb_old] / 1e6))

    # (ii) the contract the fix must satisfy
    res = G4._gamma_map(my, dx, dt, f)
    k_ax = res[0]
    assert len(res) >= 4 and isinstance(res[3], np.ndarray) \
        and res[3].shape == k_ax.shape and res[3].dtype.kind in "USO", (
        "_gamma_map must return a per-bin status array (same length as the k "
        "axis) so dead bins can be excluded; res[3] is %r"
        % (type(res[3]).__name__,))
    G, R, STAT = res[1], res[2], res[3]
    j_dead = int(np.argmin(np.abs(k_ax - k_dead)))
    j_real = int(np.argmin(np.abs(k_ax - G4.K_HALF)))
    assert STAT[j_dead] != GI.OK, (
        "the bin at k = %.3f um^-1, peak amplitude %.2e, was fitted and "
        "reported Gamma = %.3e 1/s"
        % (k_dead / 1e6, np.abs(np.fft.fft(dead, axis=1)).max() / nx,
           G[j_dead]))
    assert STAT[j_real] == GI.OK, (
        "the real mode at q/2 was rejected: status=%r" % (STAT[j_real],))
    ok_pos = (STAT == GI.OK) & (k_ax > 0)
    jbest = int(np.where(ok_pos)[0][np.nanargmax(G[ok_pos])])
    assert abs(k_ax[jbest] - G4.K_HALF) < 0.51 * G4.BOX["dk"], (
        "maximum Gamma over evaluable positive-k bins sits at %.3f um^-1, not "
        "at q/2 = %.3f um^-1" % (k_ax[jbest] / 1e6, G4.K_HALF / 1e6))


# ================================== 8a: the k = 0 / Kittel diagnostic =======
def _disp_record(nt, nx, dx, dt, B0, mat, bins, a_mode, a_uniform=0.0,
                 f_uniform=None):
    """Synthetic SAW-off dispersion record: each grid mode oscillates at its own
    omega(k).  Fixture only."""
    dk = 2 * np.pi / (nx * dx)
    my = np.zeros((nt, nx))
    for j, m in enumerate(bins):
        k = m * dk
        ex = 2 * mat["AEX"] * k ** 2 / mat["MS"]
        fk = H.GAMMA * np.sqrt((B0 + ex) * (B0 + H.MU0 * mat["MS"] + ex)) \
            / (2 * np.pi)
        my += synth_travelling(nt, nx, dx, dt, k, fk, a_mode, phase=0.3 * j)
    if a_uniform:
        fu = f_uniform if f_uniform else H.kittel_freq(B0, mat["MS"])
        t = np.arange(nt) * dt
        my += a_uniform * np.cos(2 * np.pi * fu * t)[:, None]
    return my


def test_8a_k0_diagnostic_required_by_gate():
    """The dispersion seed is band-limited with no k = 0 mode, so the k = 0
    column is a residual ~1e-15 of the main component and peak-picking on it is
    not a measurement.  The gate must refuse to evaluate, and must pass only
    when an explicit uniform diagnostic is present."""
    import G4_commensurate_onset as G4

    assert hasattr(G4, "k0_signal_check") and hasattr(G4, "disp_gate"), (
        "G4 has no k0_signal_check()/disp_gate(): stage_disp decides gate G4.2 "
        "inline on the B0 residual alone and never checks the k = 0 column")
    nx, dx, dt = G4.BOX["NX"], G4.BOX["dx"], G4.DT_REC_DISP
    nt = 2001
    mat = H.YIG_LIT
    B0 = H.kittel_field(3.0e9, mat["MS"])
    bins = (G4.BOX["bin_qhalf"], G4.BOX["bin_q"], 20)

    # (i) no uniform component -> NOT_EVALUABLE, whatever the residual says
    my = _disp_record(nt, nx, dx, dt, B0, mat, bins, 1e-4)
    k_ax, f_ax, M = SA.spectrum(my, dx, dt, window="hann")
    chk = G4.k0_signal_check(k_ax, f_ax, M, 2 * G4.Q)
    assert chk["status"] != "ok", (
        "k = 0 column at %.3e of the strongest column was accepted as a "
        "measurement" % chk["rel_power"])
    g = G4.disp_gate(resid_Hz=0.0, f_k0=1.0e9, f_kittel=3.0e9,
                     df_bin=1.0 / (nt * dt), k0=chk)
    assert g.state == "NOT_EVALUABLE", (
        "gate returned %r with no measurable k = 0 column" % (g.state,))
    assert g.blocks, "a NOT_EVALUABLE dispersion gate must block the campaign"

    # (ii) with the explicit uniform diagnostic -> evaluable, and PASS
    my2 = _disp_record(nt, nx, dx, dt, B0, mat, bins, 1e-4,
                       a_uniform=G4.K0_DIAG_AMP)
    k2, f2, M2 = SA.spectrum(my2, dx, dt, window="hann")
    chk2 = G4.k0_signal_check(k2, f2, M2, 2 * G4.Q,
                              uniform_initial_amplitude=G4.K0_DIAG_AMP)
    assert chk2["status"] == "ok", (
        "the uniform diagnostic of amplitude %.1e was not detected: %r"
        % (G4.K0_DIAG_AMP, chk2))
    f0 = G4._fpeak(f2, M2[:, int(np.argmin(np.abs(k2)))])[0]
    fK = H.kittel_freq(B0, mat["MS"])
    g2 = G4.disp_gate(resid_Hz=0.0, f_k0=f0, f_kittel=fK,
                      df_bin=1.0 / (nt * dt), k0=chk2)
    assert g2.state == "PASS", (
        "measured f(0) = %.6f GHz vs Kittel %.6f GHz, residual 0: gate says %s"
        % (f0 / 1e9, fK / 1e9, g2))

    # (iii) measurable k = 0 but off Kittel by more than a bin -> FAIL
    g3 = G4.disp_gate(resid_Hz=0.0, f_k0=fK + 5e7, f_kittel=fK,
                      df_bin=1.0e7, k0=chk2)
    assert g3.state == "FAIL", (
        "a k = 0 column 50 MHz (5 bins) off Kittel must FAIL, got %s" % (g3,))


# ==================================== 8b: the threshold bracket ============
def test_8b_yig_sweep_brackets_the_threshold():
    """The literature-YIG sweep must contain a predicted-negative AND a
    predicted-positive growth rate; the shipped sweep started above the plan's
    own predicted threshold (eps_0 = 5.92e-5)."""
    import G4_commensurate_onset as G4

    eps = np.array(G4.ARMS["YIGlit"]["eps"], dtype=float)
    gpred = np.array([H.gamma_estimate(H.YIG_LIT, e) for e in eps])
    br = GI.bracket_check(eps, gpred)
    assert br["status"] == GI.OK, (
        "predicted rates over the planned sweep %s are %s 1/s: %s"
        % (["%.0e" % e for e in eps], ["%+.3e" % x for x in gpred],
           br["reason"]))
    assert br["eps_threshold_interval"][0] < 5.92e-5 < \
        br["eps_threshold_interval"][1], (
        "the plan's own predicted threshold 5.92e-5 is not inside the "
        "bracket %r" % (br["eps_threshold_interval"],))


def test_8b_gate_rejects_unbracketed_measurement():
    """An all-positive measured sweep cannot support a threshold, and a HELD
    rate must block the verdict rather than be dropped from it."""
    assert GI.bracket_check([7e-5, 1e-4, 2e-4],
                            [1e7, 2e7, 4e7])["status"] == "NOT_BRACKETED"
    assert GI.bracket_check([3e-5, 7e-5], [np.nan, 1e7])["status"] \
        == "NOT_EVALUABLE"
    assert GI.bracket_check([3e-5, 7e-5], [-1e7, 1e7])["status"] == GI.OK


# =============================== 8c/8d/8e: RUN_PLAN statements =============
def _plan_text():
    with open(os.path.join(HERE, "RUN_PLAN.md"), encoding="utf-8") as fh:
        return fh.read()


def test_8c_pi_phase_is_not_a_gate():
    txt = _plan_text()
    assert "b(phi + pi) = -b(phi)" in txt, (
        "RUN_PLAN.md must state the identity that kills the pi-phase test: "
        "b(phi + pi) = -b(phi) does not move the seam")
    assert re.search(r"pi-phase test.{0,400}not a (decisive )?gate", txt,
                     re.S | re.I) or \
        re.search(r"not a (decisive )?gate.{0,400}pi-phase", txt, re.S | re.I), \
        "RUN_PLAN.md must say in so many words that the pi-phase test is not a gate"
    assert "seam-displacement test" in txt, (
        "if a seam test is wanted the plan must specify one that actually moves "
        "the pump discontinuity, independently of the seed and the initial "
        "magnetisation")
    assert re.search(r"seam-displacement test.{0,600}not a prerequisite", txt,
                     re.S | re.I), (
        "the seam-displacement test must be stated NOT to be a prerequisite for "
        "the commensurate runs")


def test_8d_new_box_confounds_stated():
    txt = _plan_text()
    assert "changes the wavenumber, the seed and B0" in txt, (
        "RUN_PLAN.md must state plainly that the new box changes the "
        "wavenumber AND the seed AND B0 together")
    assert re.search(r"negative result.{0,300}does not by itself establish", txt,
                     re.S | re.I), (
        "and that a negative result there does not by itself establish a "
        "boundary cause for the original signal")


def test_8e_matched_and_retuned_separated():
    import G4b_crosscheck_box as G4b

    txt = _plan_text()
    assert "matched comparison with the original condition" in txt and \
        "re-tuned operating point" in txt, (
        "RUN_PLAN.md must separate the matched comparison from the re-tuned "
        "operating point: they answer different questions")
    roles = {v: cfg.get("role") for v, cfg in G4b.VARIANTS.items()}
    assert all(roles.values()), (
        "every G4b variant must carry an explicit role (matched / re-tuned): "
        "%r" % (roles,))
    assert set(roles.values()) == {"matched", "retuned"}, (
        "roles must be exactly 'matched' or 'retuned': %r" % (roles,))
    assert roles["ref"] == "matched", roles


# ============================================================== runner =====
def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    nfail = 0
    print("=" * 78)
    print("P1 + RUN_PLAN regression suite -- audit 2026-09-18 section 8")
    print("=" * 78)
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            nfail += 1
            print("FAIL  %-*s  %s: %s" % (width, name, type(exc).__name__, exc))
            if os.environ.get("P1_TRACE"):
                traceback.print_exc()
        else:
            print("pass  %-*s" % (width, name))
    print("-" * 78)
    print("%d passed, %d failed, %d total"
          % (len(tests) - nfail, nfail, len(tests)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
