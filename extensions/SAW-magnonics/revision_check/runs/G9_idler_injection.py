"""G9 idler injection: required controls, common early-growth analysis, P1-P4.

G9_CRITERIA.json declares the decision rule and renders its documentation.
P1 requires the declared rise over the early window and the pump-off control.
P2 tests the complementary frequencies; P4 uses a conditional phase surrogate.
P3 compares GI rates on the same valid, non-transient early interval using
statistical uncertainty, synthetic-calibrated estimator uncertainty, and zero
physical rate difference conditional on a single common eigenmode. Invalid or
transient fits are indeterminate. No damping-spread bound or 10% OR is used.

All required controls are checked before spectra or predicates. Missing,
incomplete, malformed or mismatched records produce NOT_EVALUABLE.
The optional --input-dir/--output-dir flags isolate synthetic entrypoint tests.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _conditions as CONDITIONS
import _control_match as COND                                     # noqa: E402
import _criteria as CRIT                                          # noqa: E402
import _gate                                                      # noqa: E402
import _harness as H                                              # noqa: E402
import growth_interval as GI                                      # noqa: E402
import saw_analysis as SA                                         # noqa: E402

BOX    = H.BOX_PRIMARY
Q      = BOX["q"]
DK     = BOX["dk"]
N      = BOX["bin_q"]                # 12
F_SAW  = BOX["f_saw"]
OMEGA  = 2 * np.pi * F_SAW
MAT    = H.YIG_LIT
DT_REC = 20e-12
T_RUN  = 150e-9                      # idler emerges in ns; 150 ns also gives
                                     # a 6.7 MHz frequency bin
EPS_WORK  = 1e-4
INJ_RATIO = 20.0                     # A_inj / a_bg  (+26 dB over the floor)
BG_DIVISOR = 100.0                   # G9 deliberately runs a QUIET background:
                                     # the injected mode has to dominate its own
                                     # bin, and 20 x the 300 K per-mode amplitude
                                     # (9.8e-3) would be a 20 % canting, i.e. not
                                     # linear.  The physically sized bath is G6's
                                     # job; this test needs a clean signal.
RNG_SEED  = 20260941
CKPT = os.path.join(H.OUT_DIR, "G9")
os.makedirs(CKPT, exist_ok=True)
V_TOT = BOX["NX"] * H.NY * BOX["dx"] * H.CY * H.CZ
G4_DIR = os.path.join(H.OUT_DIR, "G4")


def _prov(p):
    return H.prov_kw(__file__, p)


# ------------------------------------------------------- pair bookkeeping ---
def measured_dispersion():
    """omega(k) on this grid from G4's SAW-off ring-down, or None."""
    cands = sorted(f for f in os.listdir(G4_DIR)
                   if f.startswith("G4_disp_")) if os.path.isdir(G4_DIR) else []
    if not cands:
        return None, "NOT AVAILABLE (run G4 disp first)"
    fn = cands[-1]
    d = np.load(os.path.join(G4_DIR, fn), allow_pickle=True)
    my = d["my_xt"].astype(float)
    dt = 5e-12
    k_ax, f_ax, M = SA.spectrum(my, BOX["dx"], dt, window="hann")
    f = np.full(N + 1, np.nan)
    for j in range(N + 1):
        col = np.abs(M[:, int(np.argmin(np.abs(k_ax - j * DK)))])
        m = f_ax > 0.5e9
        f[j] = f_ax[m][np.argmax(col[m])]
    return f, fn


def choose_k1():
    """Pick the three injection wavevectors.  Recorded, not improvised later."""
    f, src = measured_dispersion()
    js = [j for j in range(1, N) if j != N // 2]
    if f is None:
        j_res, j_spec = 3, 1
        mism = {j: float("nan") for j in js}
        how = "FALLBACK: no measured dispersion; k1 = 3 dk = q/4 (far from " \
              "q/2 = 6 dk) and spectator 1 dk. RE-RUN once G4 disp exists."
    else:
        mism = {j: abs(f[j] + f[N - j] - F_SAW) for j in js}
        j_res = min(mism, key=mism.get)
        j_spec = max(mism, key=mism.get)
        how = ("from %s: |f(k1)+f(q-k1)-f_SAW| minimal at j=%d (%.1f MHz), "
               "maximal at j=%d (%.1f MHz)"
               % (src, j_res, mism[j_res] / 1e6, j_spec, mism[j_spec] / 1e6))
    return dict(j_res=int(j_res), j_half=int(N // 2), j_spec=int(j_spec),
                mismatch_Hz={int(k): float(v) for k, v in mism.items()},
                how=how, dispersion_source=str(src))



def _pair_two_f(m1, m2, t_starts, omega_p, n_real=4000, seed=0,
                seg_len=None, stride=None, dt=None):
    """Two-frequency coherence and conditional phase-surrogate levels.

    Blocks approximate some overlap dependence but miss cross-block overlap.
    Neither percentile is calibrated to a physical false-positive rate.
    """
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    ts = np.asarray(t_starts, dtype=float)
    spec = CRITERIA            # the ONE binding; see reload_criteria()
    pct = float(CRIT.get("predicates", "P4_null_percentile",
                         spec=spec)["value"])
    nspec = CRIT.get("null", spec=spec)
    want = str(nspec["p4_threshold_variant"])
    overlap = None
    if seg_len and stride:
        overlap = max(0.0, 1.0 - float(stride) / float(seg_len))
    block = int(np.ceil(float(seg_len) / float(stride)))         if (seg_len and stride) else 1
    n_eff = float(ts.size) / max(1, block)
    meta = dict(
        null_kind=nspec["variants"][want]["construction"],
        null_variant_used_for_P4=want,
        null_percentile=pct,
        null_n_segments=int(ts.size),
        null_block_len_segments=int(block),
        null_effective_independent_segments=n_eff,
        null_segment_overlap_fraction=overlap,
        null_segments_overlap=bool(overlap) if overlap is not None else None,
        null_is_calibrated_false_positive_rate=False,
        null_rate_policy=str(nspec["rate_policy"]),
        null_caveat=str(nspec["statement"]))
    den = np.sqrt((np.abs(m1) ** 2).mean() * (np.abs(m2) ** 2).mean())
    if den == 0:
        meta.update(null_level_independent_phase=float("nan"),
                    null_level_block_phase=float("nan"))
        return float("nan"), float("nan"), meta
    rot = rot_of(omega_p, ts)
    C = float(np.abs((m1 * m2 * rot).mean()) / den)
    rng = np.random.default_rng(seed)
    mag = np.abs(m1) * np.abs(m2)
    # Keep observed magnitudes; replace the product phase within each block.
    levels = {}
    for name in nspec["variants_order"]:
        b = 1 if name == "independent_phase" else block
        n_draw = int(np.ceil(ts.size / b))
        ph = rng.uniform(0, 2 * np.pi, (n_real, n_draw))
        ph = np.repeat(ph, b, axis=1)[:, :ts.size]
        nul = np.abs((mag * np.exp(1j * ph)).mean(axis=1)) / den
        levels[name] = float(np.percentile(nul, pct))
    meta.update(null_level_independent_phase=levels["independent_phase"],
                null_level_block_phase=levels["block_phase"],
                null_block_over_independent=(
                    levels["block_phase"] / levels["independent_phase"]
                    if levels["independent_phase"] > 0 else float("nan")))
    return C, levels[want], meta


def rot_of(omega_p, ts):
    """exp(+i omega_p t_s): the pump-phase factor that cancels a locked pair."""
    return np.exp(1j * omega_p * np.asarray(ts, dtype=float))


# ----------------------------------------------------------------- runs -----
def runs(plan_only=False):
    # B0* comes from G4's checked accessor: a missing or FAILED dispersion
    # stage halts here instead of silently becoming the sim40 field
    # (audit 2026-09-18 sec.8, P0-2).
    import G4_commensurate_onset as G4                            # noqa: PLC0415
    B0 = G4.b0star_for_dependents()
    sel = choose_k1()
    a_phys = H.thermal_mode_amplitude(Q / 2, 300.0, B0, MAT["MS"],
                                      MAT["AEX"], V_TOT)
    g = H.gamma_estimate(MAT, EPS_WORK)
    a_lin, why = H.choose_seed_amplitude(MAT, EPS_WORK, a_phys,
                                         t_window=H.auto_window(g))
    a_seed = a_lin / BG_DIVISOR
    a_inj = INJ_RATIO * a_seed
    why = "%s ; then divided by BG_DIVISOR=%g for a quiet background" % (why, BG_DIVISOR)
    if a_inj > 0.02:
        raise SystemExit("A_inj=%.3e is not a linear perturbation; raise BG_DIVISOR" % a_inj)
    nt = int(T_RUN / DT_REC) + 1
    print("  B0=%.4f mT  eps_work=%.0e  a_seed=%.3e  A_inj=%.3e (%.0fx)  "
          "T=%.0f ns" % (B0 * 1e3, EPS_WORK, a_seed, a_inj, INJ_RATIO,
                         T_RUN * 1e9))
    print("  k1 selection: %s" % sel["how"])
    for nm, j in (("res", sel["j_res"]), ("half", sel["j_half"]),
                  ("spec", sel["j_spec"])):
        print("    inj_%-4s j=%2d  k1=%+.6f um^-1  partner j=%2d "
              "k2=%+.6f um^-1" % (nm, j, j * DK / 1e6, N - j,
                                  (N - j) * DK / 1e6))
    if plan_only:
        np.savez(os.path.join(CKPT, "G9_plan.npz"), done=True,
                 selection=json.dumps(sel), a_seed=a_seed, a_inj=a_inj,
                 **_prov(dict(box=BOX, B0=B0)))
        return

    CONFIGS = [("inj_res", sel["j_res"], EPS_WORK),
               ("inj_half", sel["j_half"], EPS_WORK),
               ("inj_spec", sel["j_spec"], EPS_WORK),
               ("inj_res_nopump", sel["j_res"], 0.0)]
    for lbl, j, eps in CONFIGS:
        tag = "%s_j%d_eps%.0e" % (lbl, j, eps)
        path = os.path.join(CKPT, "G9_%s.npz" % tag)
        mag, smeta = H.make_seed(RNG_SEED, BOX["NX"], BOX["dx"], a_seed, 2 * Q)
        k1 = j * DK
        mag = H.inject_plane_wave(mag, k1, a_inj, 0.0, BOX["dx"])
        params = dict(box=BOX, material=MAT, B0=B0, eps0=eps, f_saw=F_SAW,
                      T_run=T_RUN, DT_REC=DT_REC, DT_STEP=H.DT_STEP,
                      rng_seed=RNG_SEED, seed=smeta, a_seed=a_seed,
                      a_inj=a_inj, inject_bin=j, inject_k=k1,
                      partner_bin=N - j, partner_k=(N - j) * DK,
                      selection=sel, seed_rule=why, temperature=0.0)
        extra = dict(eps0=eps, B0=B0, inject_bin=j, partner_bin=N - j,
                     a_seed=a_seed, a_inj=a_inj, rng_seed=RNG_SEED,
                     label=lbl, selection=json.dumps(sel),
                     seed_meta=json.dumps(smeta, default=str), **_prov(params))
        br = H.BlockRun(path, nt, BOX["NX"], DT_REC, manifest=params)
        if br.done:
            print("  [skip] %s" % tag)
            continue
        print("  ---- %s" % tag)
        world, magnet, sm = H.build(BOX, MAT, B0, eps, F_SAW, mag,
                                    conditions=br.conditions)
        extra["saw_meta"] = json.dumps(sm, default=str)
        br.integrate(world, magnet, extra)


# -------------------------------------------------------------- analyze -----
# ---------------------------------------------------------------------------
# EVERY G9 threshold lives in G9_CRITERIA.json and is read through _criteria.
# There is no numeric tolerance in this file.  The previous round declared
# K_SIGMA = 3.0 and TOL_REL = 0.10 here as literals, PREREGISTRATION.md carried
# no entry for either, and the 10 % OR-clause admitted a 9 % rate mismatch at
# z = 4.6e4 as "agreement".  Both the value and its location are fixed: the
# constants are declared once, the document section is generated from them, and
# the OR-clause is gone (see _criteria.compare_growth_rates).
# ---------------------------------------------------------------------------
CRITERIA = CRIT.load()
REQUIRED_CONTROLS = {lbl: c["role"] for lbl, c in
                     CRIT.get("required_controls", spec=CRITERIA).items()}
CONTROL_SPEC = CRIT.get("required_controls", spec=CRITERIA)
PREDICATES = CRIT.get("predicates", spec=CRITERIA)


def _thr(name):
    """One predicate threshold, from the single source.  Absent -> loud."""
    return float(CRIT.get("predicates", name, spec=CRITERIA)["value"])


def reload_criteria(path=None):
    """Re-read G9_CRITERIA.json and rebind every constant derived from it.

    There is exactly ONE binding of the criteria in this module, and this is the
    only way to change it.  It exists so that the single-source property can be
    TESTED behaviourally: point this at an edited copy of the spec and every
    threshold the analysis uses must move with it.  If any threshold had a second
    copy in this file, that test would fail.
    """
    global CRITERIA, REQUIRED_CONTROLS, CONTROL_SPEC, PREDICATES
    CRITERIA = CRIT.load(path)
    CONTROL_SPEC = CRIT.get("required_controls", spec=CRITERIA)
    REQUIRED_CONTROLS = {lbl: c["role"] for lbl, c in CONTROL_SPEC.items()}
    PREDICATES = CRIT.get("predicates", spec=CRITERIA)
    return CRITERIA


def _growth_fit_err(t, a):
    """Independent centered OLS check on exactly the GI-selected interval."""
    t = np.asarray(t, dtype=np.float64)
    a = np.abs(np.asarray(a, dtype=np.float64))
    m = np.isfinite(a) & (a > 0)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n
    x, y = t[m], np.log(a[m])
    x = x - x.mean()
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sxx = float(((x - x.mean()) ** 2).sum())
    if n <= 2 or sxx <= 0:
        return float(coef[0]), np.nan, r2, n
    se = float(np.sqrt((ss_res / (n - 2)) / sxx))
    return float(coef[0]), se, r2, n


def _manifest_of(d):
    if "run_manifest" not in d:
        return None
    try:
        return json.loads(str(d["run_manifest"]))
    except Exception:                                             # noqa: BLE001
        return None


def _table_argopt(manifest, path, which):
    """Complete measured table, deterministic smallest-bin tie breaking."""
    tab = COND.get_path(manifest, path)
    expected = {j for j in range(1, N) if j != N // 2}
    if not isinstance(tab, dict):
        return None, "missing or unmeasured %s table" % path
    vals = {}
    try:
        for k, v in tab.items():
            j, value = int(k), float(v)
            if str(j) != str(k) or j in vals or not np.isfinite(value) or value < 0:
                return None, "malformed or unmeasured entry in %s" % path
            vals[j] = value
    except (TypeError, ValueError, OverflowError):
        return None, "malformed or unmeasured entry in %s" % path
    if set(vals) != expected:
        return None, "%s does not cover all non-degenerate grid bins" % path
    if CRITERIA["selection_policy"]["tie_break"] != "smallest_bin":
        raise CRIT.SpecError("unsupported G9 selection tie rule")
    extreme = (min if which == "argmin" else max)(vals.values())
    return min(k for k, v in vals.items() if v == extreme), ""


def _check_requirement(lbl, req, rec, base_rec):
    """One entry of a control's must_satisfy list, from G9_CRITERIA.json.

    The KINDS are generic; the control-specific content is entirely in the
    JSON, so a new condition is added by declaring it there and cannot be
    forgotten in the code.  An unevaluable condition is a REASON, never a
    silent skip.
    """
    kind = str(req["kind"])
    field = str(req["field"])
    man = rec["manifest"]
    base_man = base_rec["manifest"]
    got = COND.get_path(man, field, default=rec.get(field))
    head = "%s: %s" % (lbl, req["statement"])
    if got is None:
        return ("%s -- the record carries no %r, so the condition is not "
                "checkable (%s)" % (head, field, req["why"]))
    if kind == "nonzero":
        return "%s -- pump is off" % head if float(got) == 0 else None
    if kind == "equals":
        if float(got) != float(req["value"]):
            return "%s -- recorded %s = %r (%s)" % (head, field, got,
                                                    req["why"])
        return None
    if kind == "equals_base":
        want = COND.get_path(base_man, field, default=base_rec.get(field))
        if want is None:
            return "%s -- inj_res carries no %r to compare against" % (head,
                                                                       field)
        if float(got) != float(want):
            return ("%s -- recorded %s = %r against inj_res's %r (%s)"
                    % (head, field, got, want, req["why"]))
        return None
    if kind == "differs_from_base":
        want = COND.get_path(base_man, field, default=base_rec.get(field))
        if want is None:
            return "%s -- base field is missing" % head
        if float(got) == float(want):
            return ("%s -- recorded %s = %r, the same as inj_res (%s)"
                    % (head, field, got, req["why"]))
        return None
    if kind in ("argmin_of_recorded_table", "argmax_of_recorded_table"):
        which = "argmin" if kind.startswith("argmin") else "argmax"
        opt, why = _table_argopt(base_man, str(req["table"]), which)
        if opt is None:
            return "%s -- %s (%s)" % (head, why, req["why"])
        if float(got) != int(opt):
            return ("%s -- recorded %s = %d, but the %s of %s is bin %d (%s)"
                    % (head, field, int(float(got)), which, req["table"], opt,
                       req["why"]))
        return None
    raise CRIT.SpecError("G9_CRITERIA.json declares requirement kind %r, which "
                         "_check_requirement does not implement" % kind)


def _control_gate(recs):
    """Require every declared control before reading any predicate."""
    reasons = []
    # -- presence ------------------------------------------------------------
    for lbl, what in sorted(REQUIRED_CONTROLS.items()):
        if lbl not in recs:
            reasons.append("required run %s is ABSENT (%s)" % (lbl, what))
    if "inj_res" not in recs or recs["inj_res"].get("unreadable"):
        if "inj_res" in recs:
            reasons.append("inj_res: the record could not be read as a G9 run "
                           "(%s)" % recs["inj_res"]["unreadable"])
        return _gate.Gate("G9.controls", _gate.NOT_EVALUABLE,
                          "; ".join(reasons), detail=dict(reasons=reasons))
    # -- completeness --------------------------------------------------------
    for lbl in sorted(recs):
        r = recs[lbl]
        if r.get("unreadable"):
            reasons.append("%s: the record could not be read as a G9 run (%s)"
                           % (lbl, r["unreadable"]))
            continue
        if r["n_files"] > 1:
            reasons.append("%s: %d files carry this label, so which run it is "
                           "is ambiguous" % (lbl, r["n_files"]))
        if not r["done"]:
            reasons.append("%s: run is INCOMPLETE (done=False, %d records)"
                           % (lbl, r["n_rec"]))
        if r["manifest"] is None:
            reasons.append("%s: no run manifest recorded, so its conditions "
                           "cannot be verified" % lbl)
    base = recs["inj_res"]
    # -- condition match, through the checkpoint-identity mechanism ----------
    for lbl in sorted(CONTROL_SPEC):
        if lbl == "inj_res":
            continue
        r = recs.get(lbl)
        if r is None or r.get("unreadable"):
            continue
        allow = tuple(CONTROL_SPEC[lbl]["allow_diff"])
        if r["manifest"] is None or base["manifest"] is None:
            continue
        fields = COND.match(base["manifest"], r["manifest"], allow=allow)
        fields += CONDITIONS.compare(
            r["stamp"], CONDITIONS.stored_set(base["stamp"]),
            CONDITIONS.ALL_FIELDS,
            allow=CONTROL_SPEC[lbl]["allow_condition_diff"])
        if fields:
            reasons.append("%s: conditions differ from inj_res in %d field(s) "
                           "outside the %d it is allowed to differ in "
                           "(inj_res -> %s): %s"
                           % (lbl, len(fields), len(allow), lbl,
                              "; ".join(fields)))
        if r["n_rec"] != base["n_rec"]:
            reasons.append("%s: %d records against inj_res's %d, so the early "
                           "and late windows are not the same windows"
                           % (lbl, r["n_rec"], base["n_rec"]))
    # -- the conditions each control MUST satisfy, from the JSON -------------
    for lbl in sorted(CONTROL_SPEC):
        r = recs.get(lbl)
        if r is None or r.get("unreadable") or r["manifest"] is None:
            continue
        for req in CONTROL_SPEC[lbl]["must_satisfy"]:
            try:
                why = _check_requirement(lbl, req, r, base)
            except (TypeError, ValueError, OverflowError) as exc:
                why = "%s: malformed requirement input (%s)" % (lbl, exc)
            if why:
                reasons.append(why)
    if reasons:
        return _gate.Gate("G9.controls", _gate.NOT_EVALUABLE,
                          "; ".join(reasons), detail=dict(reasons=reasons))
    return _gate.Gate("G9.controls", _gate.PASS,
                      "all %d required runs present, complete, "
                      "condition-matched and satisfying every declared "
                      "condition" % len(REQUIRED_CONTROLS),
                      detail=dict(reasons=[]))


def pair_rates(my, dx, dt_rec, k1, k2, omega_pump, seed=0):
    """The two pair members' frequencies, amplitudes, rates and coherence.

    Factored out of analyze() on 2026-09-18 so that the SAME code path the
    verdict is formed from is the one the estimator's systematic floor is
    measured on (runs/measure_g9_systematic_floor.py).  A floor measured on a
    re-implementation would not bound this estimator.

    Returns a dict; every Welch parameter that the error budget needs
    (seg_len, stride, n_seg) is returned with it, because
    _criteria.compare_growth_rates refuses to assume the overlap inflation.
    """
    my = np.asarray(my, dtype=float)
    if (my.ndim != 2 or my.shape[0] < 64 or my.shape[1] < 4
            or not np.isfinite(my).all() or not 0 < dt_rec or not 0 < dx):
        raise ValueError("pair_rates requires finite (nt>=64,nx>=4) samples and positive sampling")
    nt = my.shape[0]
    k_ax, f_ax, M_late = SA.spectrum(my, dx, dt_rec, window="hann",
                                     trange=(nt // 2, nt))
    _, f_early, M_early = SA.spectrum(my, dx, dt_rec, window="hann",
                                      trange=(0, max(16, nt // 10)))
    i1 = int(np.argmin(np.abs(k_ax - k1)))
    i2 = int(np.argmin(np.abs(k_ax - k2)))

    def peak(M, i, fax):
        col = np.abs(M[:, i])
        m = fax > 0.5e9
        jj = np.where(m)[0][np.argmax(col[m])]
        return float(fax[jj]), float(col[jj])

    f1, a1 = peak(M_late, i1, f_ax)
    f2, a2 = peak(M_late, i2, f_ax)
    # the early window is shorter, so it has its own f axis; comparing
    # amplitudes across windows of different length needs the same
    # normalisation, hence the explicit length ratio below
    _, a2e_raw = peak(M_early, i2, f_early)
    a2e = a2e_raw * (M_late.shape[0] / M_early.shape[0])

    # Preserve the existing coherence estimator independently of growth layout.
    cseg = max(64, nt // 8)
    cstride = max(1, cseg // 4)
    cns = 1 + (nt - cseg) // cstride
    _, cfseg, cMs, ct0s = SA.segment_spectra(my, dx, dt_rec, cseg, cns,
                                            stride=cstride)
    cjf1, _ = SA.f_slice(cMs[0], cfseg, f1)
    cjf2, _ = SA.f_slice(cMs[0], cfseg, f2)
    seg, stride, ns = GI.segment_layout(
        nt, seg=CRITERIA["growth_rate_compatibility"]["fit_applicability"]["growth_seg_len"])
    _, fseg, Ms_, t0s = SA.segment_spectra(my, dx, dt_rec, seg, ns,
                                           stride=stride)
    jf1, _ = SA.f_slice(Ms_[0], fseg, f1)
    jf2, _ = SA.f_slice(Ms_[0], fseg, f2)
    tmid = t0s + seg * dt_rec / 2

    # ---- growth rates through THE ONE FITTER (audit 2026-09-18, item 7).
    # Until now G9 read both rates from a bare SA.growth_fit over the WHOLE
    # record, with no signal criterion, no linearity cut and no turn-over cut --
    # the same defect that was fixed in G4 and left reachable here, one code path
    # over.  A bin at the numerical floor, or a signal that had already turned
    # over, produced a confident number that P3 then compared against another
    # one.  Both rates are now the measurand stated in GI.MEASURAND, on a stated
    # interval, with a status that says so when it cannot be read.
    M_LINEAR_G9 = float(CRITERIA["growth_rate_compatibility"]["fit_applicability"]["m_linear"])
    k_band = 2.0 * max(abs(float(k1)), abs(float(k2)))
    off = np.abs(k_ax) > k_band
    i0s = np.rint(t0s / dt_rec).astype(int)
    lin = np.array([np.abs(my[i:i + seg]).max() for i in i0s]) < M_LINEAR_G9
    fits = []
    for jf_, ii_, lbl_, freq_ in ((jf1, i1, "signal", f1), (jf2, i2, "idler", f2)):
        cal = GI._amp_norm(my.shape[1], dx, dt_rec, seg, stride, freq_, float(k_ax[ii_]))
        fl = (float(np.median(np.abs(Ms_[:, jf_, :][:, off]))) / cal
              if off.any() else 0.0)
        fits.append(GI.fit_growth_interval(tmid, np.abs(Ms_[:, jf_, ii_]) / cal,
                                           floor=fl, sat_level=M_LINEAR_G9,
                                           linear_mask=lin, label=lbl_))
    fit1, fit2 = fits
    fit_valid = bool(all(f["status"] == GI.OK and f["turnover"] is None for f in fits)
                     and (fit1["i0"], fit1["i1"]) == (fit2["i0"], fit2["i1"]))
    fit_reason = ("same valid early interval; eigenmode dominance remains conditional"
                  if fit_valid else
                  "invalid, transient, or different GI early fit intervals")
    G1, R1, se1, n1 = fit1["gamma"], fit1["r2"], fit1["gamma_se"], fit1["n"]
    G2, R2, se2, n2 = fit2["gamma"], fit2["r2"], fit2["gamma_se"], fit2["n"]
    # `_growth_fit_err` stays as the INDEPENDENT recomputation of the same slope,
    # now on the interval the one fitter chose rather than on the whole record,
    # so the consistency cross-check below still compares two computations of one
    # quantity instead of two quantities.
    g1e = (_growth_fit_err(tmid[fit1["i0"]:fit1["i1"]],
                           np.abs(Ms_[:, jf1, i1])[fit1["i0"]:fit1["i1"]])[0]
           if fit1["status"] == GI.OK else float("nan"))
    g2e = (_growth_fit_err(tmid[fit2["i0"]:fit2["i1"]],
                           np.abs(Ms_[:, jf2, i2])[fit2["i0"]:fit2["i1"]])[0]
           if fit2["status"] == GI.OK else float("nan"))

    # Pair coherence.  NOTE: SA.pair_correlator takes ONE f-slice and
    # correlates k with K_pump-k inside it.  That is right for the DEGENERATE
    # pair (both members at f_SAW/2) and wrong here: a non-degenerate idler
    # lives at f_2 = f_SAW - f_1, a different slice, where the partner bin holds
    # only background.  Verified on synthetic locked-pair data: the
    # single-slice call returns C = 0.058 against a null q99 of 0.427, i.e. it
    # reports "no coherence" for a pair that is locked by construction.  The
    # two-frequency form keeps SA's sign convention exactly and reduces to
    # SA.pair_correlator when f_1 = f_2.
    C_val, C_thr, nullmeta = _pair_two_f(cMs[:, cjf1, i1], cMs[:, cjf2, i2],
                                         ct0s, omega_pump, seg_len=cseg,
                                         stride=cstride, dt=dt_rec, seed=seed)
    rep = SA.resolution_report(nt // 2, dt_rec, "hann",
                               growth_rate=max(G1, G2, 0.0))
    out = dict(
        n_rec=int(nt), f1_Hz=f1, f2_Hz=f2,
        f1_GHz=f1 / 1e9, f2_GHz=f2 / 1e9,
        partner_amp_late=float(a2), partner_amp_early=float(a2e),
        signal_amp_late=float(a1),
        partner_dB_over_early=(float(20 * np.log10(a2 / a2e)) if a2e > 0
                               else float("nan")),
        gamma_signal=float(G1), r2_signal=float(R1),
        gamma_idler=float(G2), r2_idler=float(R2),
        gamma_signal_late=fit1["late"], gamma_idler_late=fit2["late"],
        gamma_signal_status=str(fit1["status"]),
        gamma_idler_status=str(fit2["status"]),
        gamma_signal_branch=str(fit1["branch"]),
        gamma_idler_branch=str(fit2["branch"]),
        gamma_signal_reason=str(fit1["reason"])[:400],
        gamma_idler_reason=str(fit2["reason"])[:400],
        gamma_fit_interval_ns=[[fit1["t0"] * 1e9, fit1["t1"] * 1e9],
                               [fit2["t0"] * 1e9, fit2["t1"] * 1e9]],
        growth_measurand=GI.MEASURAND,
        gamma_comparison_fit_valid=fit_valid,
        gamma_comparison_fit_reason=fit_reason,
        gamma_signal_stderr=float(se1), gamma_idler_stderr=float(se2),
        gamma_fit_points=(int(n1), int(n2)),
        welch_seg_len=int(seg), welch_stride=int(stride), welch_n_seg=int(ns),
        growth_segment_duration_s=seg*dt_rec, growth_segment_df_Hz=1/(seg*dt_rec),
        coherence_seg_len=cseg, coherence_stride=cstride, coherence_n_seg=cns,
        resolution_MHz=rep.get("effective_width_Hz", np.nan) / 1e6,
        gamma_refit_consistent=bool(
            np.isfinite(g1e) and np.isfinite(g2e) and
            abs(g1e - G1) <= 1e-9 * max(1.0, abs(G1)) and
            abs(g2e - G2) <= 1e-9 * max(1.0, abs(G2))),
        C_at_k1=float(C_val), C_null_threshold=float(C_thr),
        C_null_q99=float(C_thr))
    out.update(nullmeta)
    return out


_REQUIRED_RECORD_KEYS = ("label", "my_xt", "inject_bin", "partner_bin", "eps0")


def _load_records():
    """Read complete, stamped records without losing malformed-file reasons."""
    recs = {}
    if not os.path.isdir(CKPT):
        return recs
    for fn in sorted(os.listdir(CKPT)):
        if not fn.startswith("G9_inj") or not fn.endswith(".npz"):
            continue
        path = os.path.join(CKPT, fn)
        try:
            with np.load(path, allow_pickle=False) as d:
                def scalar(key):
                    value = np.asarray(d[key])
                    if value.ndim != 0:
                        raise ValueError("%s must be scalar" % key)
                    return value.item()

                lbl = scalar("label")
                if lbl not in (*REQUIRED_CONTROLS, "inj_half"):
                    raise ValueError("unknown G9 label %r" % lbl)
                my = np.asarray(d["my_xt"], dtype=float)
                if (my.ndim != 2 or my.shape[0] < 64 or my.shape[1] != BOX["NX"]
                        or not np.isfinite(my).all()):
                    raise ValueError("my_xt has malformed/non-finite samples or wrong geometry")
                done = scalar("done")
                if not isinstance(done, bool):
                    raise ValueError("done must be a scalar boolean")
                nt = my.shape[0]
                for key in ("next_index", "nt_total"):
                    val = scalar(key)
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or val != nt:
                        raise ValueError("%s does not certify the complete record" % key)
                man = json.loads(scalar("run_manifest"))
                if not isinstance(man, dict):
                    raise ValueError("run_manifest must be an object")
                stamp, why = CONDITIONS.read(d)
                if stamp is None:
                    raise ValueError(why)
                cond = CONDITIONS.stored_set(stamp)
                problems = CONDITIONS.compare(stamp, cond, CONDITIONS.ALL_FIELDS)
                if cond.build_link_blocks:
                    problems.append("recorded build linkage is not established")
                if problems:
                    raise ValueError("; ".join(problems))
                for key in ("inject_bin", "partner_bin", "eps0", "a_seed", "a_inj",
                            "rng_seed", "B0"):
                    raw, declared = scalar(key), man.get(key)
                    if (isinstance(raw, bool) or not np.isfinite(float(raw))
                            or declared is None or raw != declared):
                        raise ValueError("%s disagrees with its finite declaration" % key)
                j, jp = scalar("inject_bin"), scalar("partner_bin")
                if (int(j) != j or int(jp) != jp or not 0 < j < N
                        or jp != N-j or (lbl != "inj_half" and j == jp)):
                    raise ValueError("invalid injected/partner bins")
                if man.get("inject_k") != j*DK or man.get("partner_k") != jp*DK:
                    raise ValueError("declared wavevectors disagree with bins")
                expected = {"grid.NX": my.shape[1], "grid.CX": BOX["dx"],
                            "pump.q": Q, "pump.f_saw": F_SAW,
                            "pump.eps_0": scalar("eps0"),
                            "material.B0": scalar("B0"),
                            "numerics.dt_rec": DT_REC, "numerics.nt": nt,
                            "numerics.seed_amplitude": scalar("a_seed"),
                            "numerics.rng_seed": scalar("rng_seed")}
                for key, value in expected.items():
                    if cond.value(key) != value:
                        raise ValueError("%s differs from the samples/declaration/analysis" % key)
                entry = dict(file=fn, label=lbl, my=my, inject_bin=int(j),
                             partner_bin=int(jp), eps0=float(scalar("eps0")),
                             done=done, n_rec=nt, manifest=man, stamp=stamp,
                             unreadable=None)
        except Exception as exc:
            entry = dict(file=fn, label="UNREADABLE", unreadable=str(exc),
                         manifest=None, n_files=1, done=False, n_rec=0)
            recs["UNREADABLE:%s" % fn] = entry
            continue
        entry["n_files"] = recs.get(lbl, {}).get("n_files", 0) + 1
        recs[lbl] = entry
    return recs


def rate_comparison(pr, spec=None, alpha=None):
    """The comparison call shared by analysis and synthetic validation."""
    return CRIT.compare_growth_rates(
        pr["gamma_signal"], pr["gamma_idler"],
        pr["gamma_signal_stderr"], pr["gamma_idler_stderr"],
        pr["f1_Hz"], pr["f2_Hz"], MAT["ALPHA"] if alpha is None else alpha,
        seg_len=pr["welch_seg_len"], stride=pr["welch_stride"],
        fit_valid=bool(pr["gamma_comparison_fit_valid"] and pr["gamma_refit_consistent"]),
        spec=CRITERIA if spec is None else spec)


def analyze():
    try:
        return _analyze()
    except (CRIT.SpecError, ValueError, TypeError, KeyError, OverflowError) as exc:
        res = dict(verdict_state=_gate.NOT_EVALUABLE,
                   verdict="NOT_EVALUABLE: %s: %s" % (type(exc).__name__, exc),
                   control_gate=dict(name="G9.inputs", state=_gate.NOT_EVALUABLE,
                                     reasons=[str(exc)]))
        _save_summary(res)
        return res


def _analyze():
    recs = _load_records()
    cg = _control_gate(recs)
    if cg.blocks:
        res = dict(verdict_state=_gate.NOT_EVALUABLE,
                   verdict="NOT_EVALUABLE: " + cg.reason,
                   control_gate=dict(name=cg.name, state=cg.state,
                                     reason=cg.reason, reasons=cg.detail["reasons"]))
        print(res["verdict"])
        _save_summary(res)
        return res
    usable = {k: v for k, v in recs.items() if not v.get("unreadable")}

    res = {}
    for lbl in sorted(usable):
        rec = usable[lbl]
        j, jp = rec["inject_bin"], rec["partner_bin"]
        pr = pair_rates(rec["my"], BOX["dx"], DT_REC, j * DK, jp * DK, OMEGA)
        res[lbl] = dict(
            file=rec["file"], inject_bin=j, partner_bin=jp,
            eps0=rec["eps0"], done=rec["done"],
            f_sum_GHz=(pr["f1_Hz"] + pr["f2_Hz"]) / 1e9,
            f_saw_GHz=F_SAW / 1e9,
            f_sum_resid_MHz=(pr["f1_Hz"] + pr["f2_Hz"] - F_SAW) / 1e6,
            **pr)
        print("  %-18s k1=bin%-2d  f1=%.4f f2=%.4f  sum=%.4f GHz "
              "(resid %+.1f MHz, res %.1f MHz)  partner %+.1f dB  "
              "G_sig=%+.2e+-%.1e G_idl=%+.2e+-%.1e  C=%.4f"
              % (lbl, j, pr["f1_GHz"], pr["f2_GHz"],
                 (pr["f1_Hz"] + pr["f2_Hz"]) / 1e9,
                 (pr["f1_Hz"] + pr["f2_Hz"] - F_SAW) / 1e6,
                 pr["resolution_MHz"], pr["partner_dB_over_early"],
                 pr["gamma_signal"], pr["gamma_signal_stderr"],
                 pr["gamma_idler"], pr["gamma_idler_stderr"],
                 pr["C_at_k1"]))

    # -- required controls, BEFORE any predicate is read ---------------------
    cg = _control_gate(recs)
    res["control_gate"] = dict(name=cg.name, state=cg.state,
                               reason=cg.reason,
                               reasons=cg.detail.get("reasons", []))
    print("\n  controls (presence, completeness, condition match) ... %s"
          % cg.state)
    for why in cg.detail.get("reasons", []):
        print("      - %s" % why)

    if cg.blocks:
        res["verdict_state"] = _gate.NOT_EVALUABLE
        res["verdict"] = ("NOT_EVALUABLE: %s. No verdict on pair generation "
                          "can be formed from these runs." % cg.reason)
        print("  VERDICT: %s" % res["verdict"])
        _save_summary(res)
        return res

    r = res["inj_res"]
    nop = res["inj_res_nopump"]
    spec = res["inj_spec"]

    # both legs of P1 in dB: a linear amplitude ratio overflows for a large
    # declared threshold, and the threshold is declared in dB anyway
    over_nopump_dB = (20 * np.log10(r["partner_amp_late"] /
                                    nop["partner_amp_late"])
                      if nop["partner_amp_late"] > 0 else float("inf"))
    P1 = (r["partner_dB_over_early"] >= _thr("P1_partner_rise_dB") and
          over_nopump_dB >= _thr("P1_over_nopump_dB"))
    P2 = abs(r["f_sum_resid_MHz"]) <= max(r["resolution_MHz"],
                                          _thr("P2_resolution_floor_MHz"))
    # -- P3: the three-part error budget, from G9_CRITERIA.json --------------
    cmp_ = rate_comparison(r)
    g1, g2 = r["gamma_signal"], r["gamma_idler"]
    if cmp_["compatible"] == CRIT.NOT_DETERMINABLE or \
            not r["gamma_refit_consistent"]:
        P3 = _gate.NOT_DETERMINABLE
        if not r["gamma_refit_consistent"]:
            cmp_["reason"] = ("the independent re-fit of the two slopes "
                              "does not confirm valid GI rates, so neither rate "
                              "is trusted; " + str(cmp_.get("reason", "")))
    else:
        P3 = bool(g1 > 0 and g2 > 0 and cmp_["compatible"])
    P4 = r["C_at_k1"] > r["C_null_threshold"]
    dark = _thr("control_max_rise_dB")
    nulls_ok = (spec["partner_dB_over_early"] < dark and
                nop["partner_dB_over_early"] < dark)

    print("  P1 partner rises >=%.0f dB ........... %s (%+.1f dB over its "
          "own early window, %+.1f dB over pump-off)"
          % (_thr("P1_partner_rise_dB"), P1, r["partner_dB_over_early"],
             over_nopump_dB))
    print("  P2 f1+f2 = f_SAW within resolution .. %s (resid %+.1f MHz, "
          "res %.1f MHz)" % (P2, r["f_sum_resid_MHz"], r["resolution_MHz"]))
    print("  P3 both grow AND agree within the")
    print("     three-part error budget .......... %s" % P3)
    print("       |dGamma| = %.4e   allowance %.4e   dominant term: %s"
          % (cmp_["delta"], cmp_["allowance"], cmp_.get("dominant_term",
                                                        "n/a")))
    print("       statistical  K*sigma_stat = %.4e  (se %.3e / %.3e, "
          "overlap inflation x%.3g)"
          % (cmp_["K_SIGMA"] * cmp_["sigma_stat"], cmp_["stderr_1"],
             cmp_["stderr_2"], cmp_["overlap_inflation"]))
    print("       systematic   K*sigma_sys  = %.4e  (f_sys = %.3e, calibrated "
          "on synthetic controls)"
          % (cmp_["K_SIGMA"] * cmp_["sigma_sys"],
             cmp_["systematic_relative_floor"]))
    print("       physical     tol_phys = %.4e (common-eigenmode equality; conditional)"
          % cmp_["tol_phys"])
    print("  P4 C(k1) above the %s surrogate level  %s (%.4f vs %.4f; a LEVEL, "
          "not a false-positive rate)"
          % (r.get("null_variant_used_for_P4", "?"), P4, r["C_at_k1"],
             r["C_null_threshold"]))
    print("  nulls (spectator, pump-off) stay dark %s" % nulls_ok)

    if P3 == _gate.NOT_DETERMINABLE:
        state = _gate.NOT_EVALUABLE
        verdict = ("NOT_EVALUABLE: the two growth rates could not be compared "
                   "(%s), so P3 has no value" % cmp_.get("reason", ""))
    elif P1 and P2 and P3 and P4 and nulls_ok:
        state = "DETECTED"
        verdict = "stimulated pair-generation signature DETECTED conditionally under the declared G9 estimator scope"
    else:
        state = "NOT_DETECTED"
        verdict = ("NOT DETECTED at this strain/duration/injection level -- "
                   "this is a bound, not a refutation at all strains")
        if P3 is False and g1 > 0 and g2 > 0:
            verdict += ("; P3 failed because %s" % cmp_["reason"])
    # The three terms in full, plus the summary statistics the audit's own
    # counterexamples quote, so a reader can impose a different rule without
    # re-running: sigma is the COMBINED error (statistical and systematic), z is
    # dGamma in units of it, and rel is the relative difference.  None of them is
    # a threshold; the only threshold is `allowance`.
    sig_tot = float(np.hypot(cmp_["sigma_stat"], cmp_["sigma_sys"])) \
        if np.isfinite(cmp_["sigma_stat"]) and np.isfinite(cmp_["sigma_sys"]) \
        else float("nan")
    res["predicates"] = dict(P1=P1, P2=P2, P3=P3, P4=P4, nulls_ok=nulls_ok,
                             gamma_signal=g1, gamma_idler=g2,
                             gamma_compat_sigma=sig_tot,
                             gamma_compat_z=(cmp_["delta"] / sig_tot
                                             if sig_tot > 0 else
                                             float("inf")),
                             gamma_compat_rel=cmp_["relative_difference"],
                             gamma_compat_allowance=cmp_["allowance"],
                             gamma_compat_allowance_rel=cmp_[
                                 "relative_allowance"],
                             K_SIGMA=cmp_["K_SIGMA"],
                             rate_comparison=dict(cmp_),
                             # every threshold the analysis actually used, read
                             # back from the single source, so a reader (and the
                             # single-source regression test) can confirm that no
                             # second copy of any of them lives in this file
                             thresholds_used={k: _thr(k)
                                              for k in sorted(PREDICATES)},
                             P2_tolerance_MHz=max(
                                 r["resolution_MHz"],
                                 _thr("P2_resolution_floor_MHz")),
                             criteria_file=CRIT.SPEC_PATH,
                             criteria_version=CRITERIA["version"])
    res["verdict_state"] = state
    res["verdict"] = verdict
    print("  VERDICT: %s" % verdict)
    _save_summary(res)
    return res


def _save_summary(res):
    dest = globals().get("SUMMARY_DIR", CKPT)
    os.makedirs(dest, exist_ok=True)
    np.savez(os.path.join(dest, "G9_summary.npz"),
             rows=json.dumps(res, default=str),
             verdict_state=str(res.get("verdict_state", _gate.NOT_EVALUABLE)),
             **_prov(dict(stage="analyze")))
    print("  wrote G9_summary.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="plan",
                    choices=["plan", "runs", "analyze"])
    ap.add_argument("--input-dir", help="G9 record directory (analysis only)")
    ap.add_argument("--output-dir", help="summary directory (analysis only)")
    a = ap.parse_args()
    if (a.input_dir or a.output_dir) and a.stage != "analyze":
        ap.error("directory overrides are only supported for analyze")
    if a.input_dir:
        CKPT = os.path.abspath(a.input_dir)
    SUMMARY_DIR = os.path.abspath(a.output_dir) if a.output_dir else CKPT
    print("=" * 78)
    print("G9 idler injection -- box %s, q = bin %d, q/2 = bin %d"
          % (BOX["tag"], BOX["bin_q"], BOX["bin_qhalf"]))
    print("=" * 78)
    # A NOT_EVALUABLE analysis must not look like a completed step to whatever
    # calls this script, so it exits non-zero (audit 2026-09-18 sec.8, P0-3).
    try:
        if a.stage == "plan":
            runs(plan_only=True)
        elif a.stage == "runs":
            runs()
        else:
            out = analyze()
            if str(out.get("verdict_state")) == _gate.NOT_EVALUABLE:
                print("  BLOCKED: no verdict is available from these runs.")
                sys.exit(2)
    except _gate.GateHalt as halt:
        print("\n  HALTED: %s" % halt)
        sys.exit(2)
