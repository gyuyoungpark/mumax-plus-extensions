"""G4b -- STEP 3: box and seed cross-validation.

If the feature G4 reports is physics it must not care which commensurate box it
is measured in, nor which random seed started it.  Three things are varied
independently, one at a time:

  box      n = 18 (NX = 2100, dx = 5 nm, L = 10.500 um).  Different L, so a
           different dk (0.5983986 vs 0.8975979 um^-1) and a different bin
           index for q (18) and q/2 (9).  Any feature that is a property of the
           GRID moves; any feature that is a property of the PHYSICS stays at
           the same k in um^-1.
  cell     n = 12 at dx = lambda/125 = 4.666667 nm (NX = 1500).  Same L, same
           dk, same bin indices -- only the discretisation changes.  This is
           the cell-size convergence check, and it is deliberately separated
           from the box change so the two cannot be confounded.
  seed     three independent, explicitly recorded rng seeds per point.

TWO DIFFERENT QUESTIONS, KEPT APART (audit 2026-09-18 sec.8e)
  role="matched"  the reference box re-run at the SAME strains as G4 and the
                  same B0, with only the rng seed changed.  This is the only
                  comparison that isolates one variable, and it is the one that
                  answers "is the G4 result seed-dependent?".
  role="retuned"  the n=18 box and the dx=4.667 nm grid, run at strain levels
                  re-derived from the Gamma(eps_0) G4 measured.  This answers
                  "does the feature survive a different grid AT ITS OWN
                  operating point?" -- a different question, and the two are
                  never pooled in the summary.

WHAT A NEGATIVE RESULT HERE DOES NOT SHOW (audit 2026-09-18 sec.8d)
  Changing the box changes the WAVENUMBER GRID, the SEED (a different NX means a
  different number of modes and a different random draw) and B0 (adopted from a
  dispersion tuning performed in the primary box) all at once.  A null result in
  the new box therefore does NOT by itself establish that the original signal
  was a boundary artefact: three things moved together.  That inference would
  need a test that moves ONE of them -- see RUN_PLAN sec.4, the
  seam-displacement test.

Strain levels: below / near / above the onset that G4 measures.  The default
list below is used only if G4_summary.npz is absent; when it is present the
three levels are taken from the measured Gamma(eps_0) zero crossing, so this
script does not hard-code an onset it has not measured.

Usage
    python G4b_crosscheck_box.py            # all configurations
    python G4b_crosscheck_box.py --only box
    python G4b_crosscheck_box.py analyze
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H                                              # noqa: E402
import growth_interval as GI                                      # noqa: E402
import saw_analysis as SA                                         # noqa: E402

CKPT = os.path.join(H.OUT_DIR, "G4b")
os.makedirs(CKPT, exist_ok=True)
G4_DIR = os.path.join(H.OUT_DIR, "G4")

DT_REC    = 20e-12
RNG_SEEDS = (20260921, 20260922, 20260923)
MAT       = H.YIG_LIT        # the observability arm carries the cross-check
EPS_FALLBACK = (7e-5, 1e-4, 2e-4)     # below / near / above, from the Gamma fit

# role: "matched" = same condition as G4, one variable changed (the seed);
#       "retuned"  = different grid at its own re-derived operating point.
# The two roles answer different questions and are reported separately.
VARIANTS = {
    "box":  dict(box=H.BOX_CROSS, role="retuned",      # n=18, NX=2100, dx=5 nm
                 confounds=("wavenumber grid (dk 0.5984 vs 0.8976 um^-1), seed "
                            "(NX 2100 vs 1400: different mode count and draw) "
                            "and B0 (adopted from the primary-box dispersion "
                            "tuning) all change together")),
    "cell": dict(box=H.BOX_CELLCONV, role="retuned",   # n=12, NX=1500, dx=4.667
                 confounds=("discretisation and seed change; L and dk do not")),
    "ref":  dict(box=H.BOX_PRIMARY, role="matched",    # n=12, NX=1400, dx=5 nm
                 confounds=("nothing but the rng seed: same box, same B0, same "
                            "strain list as G4")),
}
# The matched arm is run at G4 own strain list, not at re-derived levels.
MATCHED_ARM = "YIGlit"


def _prov(p):
    return H.prov_kw(__file__, p)


def eps_levels():
    """below / near / above onset, from G4's measured Gamma(eps_0) if available."""
    p = os.path.join(G4_DIR, "G4_summary.npz")
    if not os.path.isfile(p):
        print("  (no G4_summary.npz: using the fallback strain list %s)"
              % (EPS_FALLBACK,))
        return EPS_FALLBACK, "fallback (G4 not yet analysed)"
    rows = json.loads(str(np.load(p, allow_pickle=True)["rows"]))
    rows = [r for r in rows if r["arm"] == MAT["tag"]]
    if len(rows) < 2:
        return EPS_FALLBACK, "fallback (too few G4 points for this arm)"
    e = np.array([r["eps0"] for r in rows])
    g = np.array([r["gamma_qhalf"] for r in rows])
    o = np.argsort(e)
    e, g = e[o], g[o]
    a, b = np.polyfit(e, g, 1)
    e_th = -b / a if a != 0 else float(np.median(e))
    lev = (0.7 * e_th, 1.15 * e_th, 2.0 * e_th)
    return tuple(float(x) for x in lev), \
        "from G4: Gamma=0 at eps_0=%.3e (linear fit, %d points)" % (e_th, e.size)


def run(only=None):
    # One checked accessor, no Kittel fallback (audit 2026-09-18 sec.8, P0-2).
    import G4_commensurate_onset as G4                            # noqa: PLC0415
    B0 = G4.b0star_for_dependents()
    b0src = "G4 dispersion tuning (gate PASS, run identity verified)"
    levels, how = eps_levels()
    try:
        import G4_commensurate_onset as G4
        matched_eps = tuple(G4.ARMS[MATCHED_ARM]["eps"])
        matched_how = "G4 own strain list for arm %s (matched comparison)" % MATCHED_ARM
    except Exception as exc:                                      # noqa: BLE001
        matched_eps, matched_how = levels, ("NOT DETERMINABLE (%s): falling "
                                            "back to the re-derived levels"
                                            % exc)
    print("  B0 = %.4f mT (%s)" % (B0 * 1e3, b0src))
    print("  retuned levels = %s  [%s]"
          % (["%.3e" % x for x in levels], how))
    print("  matched levels = %s  [%s]"
          % (["%.3e" % x for x in matched_eps], matched_how))

    for vname, vcfg in VARIANTS.items():
        if only and vname != only:
            continue
        bx = vcfg["box"]
        role = vcfg["role"]
        eps_list = matched_eps if role == "matched" else levels
        eps_how = matched_how if role == "matched" else how
        V_tot = bx["NX"] * H.NY * bx["dx"] * H.CY * H.CZ
        k_half = bx["q"] / 2
        a_phys = H.thermal_mode_amplitude(k_half, 300.0, B0, MAT["MS"],
                                          MAT["AEX"], V_tot)
        print("\n---- variant %-5s [%s] %s  dk=%.7f um^-1  bin(q)=%d "
              "bin(q/2)=%d" % (vname, role, bx["tag"], bx["dk"] / 1e6,
                               bx["bin_q"], bx["bin_qhalf"]))
        print("     confounded with: %s" % vcfg["confounds"])
        for eps in eps_list:
            g = H.gamma_estimate(MAT, eps)
            win = H.auto_window(g)
            a_seed, why = H.choose_seed_amplitude(MAT, eps, a_phys,
                                                  t_window=win)
            plan = H.plan_runtime(MAT, eps, a_seed, t_window=win, t_min=40e-9)
            # Same rule as G4: a blocking plan["status"]
            # (INSUFFICIENT_LINEAR_WINDOW / NO_LINEAR_WINDOW) stops this
            # configuration instead of being carried into a run length.
            if plan["status"] in H.PLAN_BLOCKING:
                raise H.PlanRefused(
                    "G4b variant %s eps=%.3e: run plan blocked (%s) -- %s"
                    % (vname, eps, plan["status"], plan["note"]))
            nt = H.nt_records(plan, DT_REC,
                              "G4b variant %s eps=%.3e" % (vname, eps))
            for s in RNG_SEEDS:
                tag = "%s_eps%.3e_seed%d" % (vname, eps, s)
                path = os.path.join(CKPT, "G4b_%s.npz" % tag)
                mag, smeta = H.make_seed(s, bx["NX"], bx["dx"], a_seed,
                                         2 * bx["q"])
                T_eq = H.seed_temperature(a_seed, k_half, B0, MAT["MS"],
                                          MAT["AEX"], V_tot)
                params = dict(variant=vname, role=role,
                              confounds=vcfg["confounds"], box=bx,
                              material=MAT, B0=B0,
                              B0_source=b0src, eps0=eps, eps_choice=eps_how,
                              f_saw=bx["f_saw"], T_run=plan["t_run"],
                              DT_REC=DT_REC, DT_STEP=H.DT_STEP, rng_seed=s,
                              seed=smeta, a_seed=a_seed, a_phys_300K=a_phys,
                              seed_equiv_temperature_K=T_eq, seed_rule=why,
                              gamma_expected=g, temperature=0.0)
                extra = dict(eps0=eps, B0=B0, variant=vname, role=role,
                             rng_seed=s,
                             a_seed=a_seed, gamma_expected=g,
                             NX=bx["NX"], dx=bx["dx"], dk=bx["dk"],
                             seed_meta=json.dumps(smeta, default=str),
                             **_prov(params))
                br = H.BlockRun(path, nt, bx["NX"], DT_REC,
                                manifest=params)
                if br.done:
                    print("  [skip] %s" % tag)
                    continue
                print("  ---- %s T=%.0f ns a=%.3e (T_eq=%.3g K)"
                      % (tag, plan["t_run"] * 1e9, a_seed, T_eq))
                world, magnet, sm = H.build(bx, MAT, B0, eps, bx["f_saw"],
                                            mag, conditions=br.conditions)
                extra["saw_meta"] = json.dumps(sm, default=str)
                br.integrate(world, magnet, extra)


def analyze():
    """Report Gamma and the band position IN PHYSICAL UNITS (um^-1), never in
    bin index, so the box comparison is meaningful.

    REFUTES a physical interpretation if: the peak of Gamma(k) sits at a fixed
    BIN INDEX across the two boxes rather than at a fixed k in um^-1, or if the
    seed-to-seed scatter of Gamma at fixed (box, eps) exceeds the difference
    between boxes.  Both outcomes are recorded, not hidden.
    """
    rows = []
    for fn in sorted(os.listdir(CKPT)):
        if not fn.startswith("G4b_") or not fn.endswith(".npz"):
            continue
        d = np.load(os.path.join(CKPT, fn), allow_pickle=True)
        my = d["my_xt"].astype(float)
        dx = float(d["dx"])
        dk = float(d["dk"])
        eps = float(d["eps0"])
        q = H.BOX_PRIMARY["q"]
        # THE one fitter, the same call G4 makes (audit 2026-09-18 item 7).
        # Before this, G4b ran its own bare SA.growth_fit over the whole record
        # with an argmax over positive k and NO signal test, so the dead-bin
        # counterexample reproduced here verbatim after G4 had been fixed: a
        # component of peak amplitude 1e-13 growing at 8e8 1/s won the maximum at
        # k = 303.3881 um^-1 and was reported as Gamma = 8.7948e+08 1/s instead
        # of the real mode's 3.0000e+08 at q/2.  A second fitter is a second
        # measurand; there is now one of each.
        k_ax, G, R, STAT, t0s, jf, f_ax, gmeta = GI.gamma_map(
            my, dx, DT_REC, H.BOX_PRIMARY["f_saw"] / 2, k_offband=2 * q)
        jh = int(np.argmin(np.abs(k_ax - q / 2)))
        best = gmeta["best"]
        rows.append(dict(file=fn, variant=str(d["variant"]),
                         role=(str(d["role"]) if "role" in d
                               else VARIANTS.get(str(d["variant"]), {})
                               .get("role", "UNKNOWN")),
                         eps0=eps,
                         seed=int(d["rng_seed"]), NX=int(d["NX"]),
                         dk_um=dk / 1e6,
                         gamma_qhalf=float(G[jh]),
                         gamma_qhalf_status=str(STAT[jh]),
                         gamma_qhalf_branch=str(gmeta["branch"][jh]),
                         gamma_qhalf_reason=str(gmeta["held_reasons"][jh])[:400],
                         gamma_qhalf_late=gmeta["late"][jh],
                         growth_segment_duration_s=gmeta["segment_duration"],
                         growth_segment_df_Hz=gmeta["segment_df"],
                         gamma_max=float(best["gamma"]),
                         gamma_max_status=str(best["state"]),
                         k_argmax_um=float(best["k_um"]),
                         bin_argmax=(int(round(k_ax[best["j"]] / dk))
                                     if best["j"] >= 0 else -1),
                         n_bins_evaluable=int(gmeta["n_bins_ok"])))
        print("  %-46s G(q/2)=%+.3e [%s]  argmax %s"
              % (fn, G[jh], STAT[jh],
                 ("k=%+.4f um^-1 (bin %d)" % (best["k_um"],
                                              round(k_ax[best["j"]] / dk)))
                 if best["state"] == GI.OK else str(best["state"])))
        if STAT[jh] != GI.OK:
            print("      q/2 %s: %s" % (STAT[jh], gmeta["held_reasons"][jh]))
    if not rows:
        print("  nothing to analyse")
        return
    np.savez(os.path.join(CKPT, "G4b_summary.npz"), rows=json.dumps(rows),
             measurand=GI.MEASURAND, **_prov(dict(stage="analyze")))
    by = {}
    for r in rows:
        if r.get("gamma_qhalf_status", GI.OK) != GI.OK:
            continue                 # a held rate is not a small rate
        by.setdefault((r["role"], r["variant"], r["eps0"]), []).append(
            r["gamma_qhalf"])
    n_held = sum(1 for r in rows if r.get("gamma_qhalf_status", GI.OK) != GI.OK)
    if n_held:
        print("\n  %d of %d rows have NO measured Gamma(q/2) and are excluded "
              "from the scatter below" % (n_held, len(rows)))
    print("\n  seed scatter of Gamma(q/2) at fixed (role, variant, eps) -- "
          "matched and re-tuned runs answer different questions and are NOT "
          "pooled:")
    for role in ("matched", "retuned"):
        sel = {k: v for k, v in by.items() if k[0] == role}
        if not sel:
            continue
        print("    [%s]" % role)
        for kk, v in sorted(sel.items()):
            v = np.array(v)
            print("      %-6s eps=%.3e  n=%d  mean=%+.3e  sd=%.3e"
                  % (kk[1], kk[2], v.size, v.mean(),
                     v.std(ddof=1) if v.size > 1 else np.nan))
    print("  wrote G4b_summary.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="run",
                    choices=["run", "analyze"])
    ap.add_argument("--only", default=None, choices=list(VARIANTS))
    a = ap.parse_args()
    print("=" * 78)
    print("G4b box / cell / seed cross-validation")
    print("=" * 78)
    # Same handler and the same exit codes as every other script in runs/
    # (open item 10: G4b had no _gate.GateHalt handler at all, so a blocked
    # prerequisite surfaced as a traceback).
    if a.stage == "run":
        sys.exit(H.run_entry("G4b.run", run, a.only))
    else:
        sys.exit(H.run_entry("G4b.analyze", analyze))
