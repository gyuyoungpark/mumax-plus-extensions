"""Cost of the commensurate campaign, DERIVED from the live run scripts.

Every row below is built from the configuration the campaign would actually run:
the strain sweeps come from `G4.ARMS`, the seed lists from `G4.RNG_SEEDS` and
`G4b.RNG_SEEDS`, the box variants from `G4b.VARIANTS`, the realisation count from
`G6.N_THERMAL_REALISATIONS`, and every run length from `H.plan_runtime` --
nothing in this file names a strain, a seed count or a duration of its own.

WHY IT WAS REWRITTEN (audit 2026-09-18, item 13)
  The previous version's docstring said "every runtime is derived, not asserted"
  while it still budgeted the RETIRED literature-YIG sweep (7e-5, 1e-4, 2e-4,
  3e-4) and the old G4b reference shape, and printed 63 configurations / 9.8 us /
  68-89 GPU-hours against RUN_PLAN section 5's table of 66 / 10.3 us / 71-94.  A
  cost table that cannot be regenerated from the tree is not a cost table, and a
  script whose docstring claims derivation while hard-coding a retired sweep is
  worse than a table marked "hand-maintained".  `--check` now compares the
  derived total against RUN_PLAN's own table and exits non-zero on a mismatch, so
  the two cannot drift again in silence.

THE ONE ASSUMPTION, unchanged and still flagged in the output
  Throughput is inferred from data/sim4*_run.log at NX=1024, NY=8, NZ=1,
  dt_step = 0.2 ps, DT_REC = 20 ps:
      sim41 (T=0) 100 ns / 27.7 min ; sim42 50/14.0 ; sim44 40/10.9 ;
      sim46 40/14.4 ; sim47 40/14.3 ; sim48 40/14.4 ; sim45 (no SAW) 20/6.5
    -> 0.273 to 0.360 min per ns of physics, median 0.325.
  Scaled LINEARLY in NX (demag FFT and per-cell work both scale that way).  The
  linear scaling is an assumption, not a measurement, and is printed with the
  table.

Usage
    python campaign_cost.py                # print the derived table
    python campaign_cost.py --check        # and compare with RUN_PLAN sec.5
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "runs"))
sys.path.insert(0, HERE)
import _harness as H                                              # noqa: E402
import G4_commensurate_onset as G4                                # noqa: E402
import G4b_crosscheck_box as G4b                                  # noqa: E402
import G6_clean_seed as G6                                        # noqa: E402
import G9_idler_injection as G9                                   # noqa: E402

LO, HI = 0.273, 0.360          # min per ns of physics at NX = 1024
RUN_PLAN = os.path.join(HERE, "RUN_PLAN.md")


def wall(T_ns, NX):
    f = NX / 1024.0
    return LO * f * T_ns / 60.0, HI * f * T_ns / 60.0


def _labels_in_configs(path, func):
    """Count the CONFIGS entries declared inside `func` in `path`.

    G6 and G9 declare their block shapes inside their run functions, so the only
    honest way to derive their run counts here is to read that declaration.  If
    the declaration cannot be parsed this RAISES: a cost table that silently
    falls back to a remembered number is the defect this rewrite removes.
    """
    src = open(path, encoding="utf-8").read()
    m = re.search(r"def %s\(.*?\n(.*?)\n(?=def |\Z)" % re.escape(func), src,
                  re.S)
    if not m:
        raise RuntimeError("cannot find %s() in %s to derive its run count"
                           % (func, os.path.basename(path)))
    body = m.group(1)
    c = re.search(r"CONFIGS\s*=\s*\[(.*?)\]\s*\n", body, re.S)
    if not c:
        raise RuntimeError("cannot find the CONFIGS declaration inside %s() in "
                           "%s: the cost table cannot be derived and must not be "
                           "guessed" % (func, os.path.basename(path)))
    labels = re.findall(r'\(\s*"([A-Za-z0-9_]+)"', c.group(1))
    if not labels:
        raise RuntimeError("CONFIGS in %s() of %s parsed to zero labels"
                           % (func, os.path.basename(path)))
    return labels


def plan_details(mat, eps, box, t_min=40e-9):
    """Planner diagnostics; total runtime is not a measured fit interval.

    frequency_window_ns is the planner's requested frequency window. The
    actual Fourier window comes from growth_interval.segment_layout and the
    early fit interval is selected from the recorded signal by the fitter.
    t_linear_ns is the planner's predicted saturation time, not a fit endpoint.
    """
    B0 = H.kittel_field(3.0e9, mat["MS"])
    V = box["NX"] * H.NY * box["dx"] * H.CY * H.CZ
    ap = H.thermal_mode_amplitude(box["q"] / 2, 300.0, B0, mat["MS"],
                                  mat["AEX"], V)
    g = H.gamma_estimate(mat, eps)
    w = H.auto_window(g)
    a, _ = H.choose_seed_amplitude(mat, eps, ap, t_window=w)
    plan = H.plan_runtime(mat, eps, a, t_window=w, t_min=t_min)
    if plan["status"] in H.PLAN_BLOCKING:
        raise H.PlanRefused(
            "cannot cost eps_0 = %.3e in box %s: the run plan is blocked (%s) "
            "-- %s" % (eps, box["tag"], plan["status"], plan["note"]))
    return dict(eps0=float(eps), gamma=float(g), a_seed=float(a),
                t_run_ns=float(plan["t_run"]) * 1e9,
                frequency_window_ns=float(w) * 1e9,
                t_linear_ns=float(plan["t_linear"]) * 1e9,
                planner_status=str(plan["status"]),
                window_ns=float(w) * 1e9)  # existing JSON key, same meaning


def plan_ns(mat, eps, box, t_min=40e-9):
    """Compatible (runtime_ns, predicted_gamma, seed, frequency_window_s)."""
    d = plan_details(mat, eps, box, t_min=t_min)
    return d["t_run_ns"], d["gamma"], d["a_seed"], d["frequency_window_ns"] * 1e-9


def sweep_row(label, mat, epss, box, seeds, note=""):
    ns_each, detail = [], []
    for e in epss:
        d = plan_details(mat, e, box)
        ns_each.append(d["t_run_ns"])
        detail.append(d)
    physics = float(sum(ns_each) * seeds)
    lo, hi = wall(physics, box["NX"])
    return dict(label=label, configs=int(len(epss) * seeds), physics_ns=physics,
                NX=int(box["NX"]), gpu_lo=lo, gpu_hi=hi, detail=detail,
                note=note, per_run_ns=[round(x, 2) for x in ns_each])


def derive():
    rows = []
    n_seeds_g4 = len(G4.RNG_SEEDS)
    for arm in sorted(G4.ARMS):
        cfg = G4.ARMS[arm]
        rows.append(sweep_row(
            "G4 arm %s (%d strains x %d seeds)" % (arm, len(cfg["eps"]),
                                                   n_seeds_g4),
            cfg["mat"], tuple(cfg["eps"]), G4.BOX, n_seeds_g4,
            note="G4.ARMS[%r]" % arm))

    # dispersion pre-pass: B0_MAXIT iterations of T_DISP, SAW off
    n_disp, t_disp = int(G4.B0_MAXIT) + 1, float(G4.T_DISP) * 1e9
    lo, hi = wall(n_disp * t_disp, G4.BOX["NX"])
    rows.append(dict(label="G4 dispersion pre-pass (<=%d x %.0f ns)"
                     % (n_disp, t_disp), configs=n_disp,
                     physics_ns=n_disp * t_disp, NX=int(G4.BOX["NX"]),
                     gpu_lo=lo, gpu_hi=hi, detail=[],
                     note="G4.B0_MAXIT + 1 iterations of G4.T_DISP",
                     per_run_ns=[t_disp]))

    # G4b: the retuned variants use the fallback levels, the matched one uses
    # G4's own strain list.  Both shapes come from G4b itself.
    n_seeds_g4b = len(G4b.RNG_SEEDS)
    for vname in sorted(G4b.VARIANTS):
        v = G4b.VARIANTS[vname]
        if v["role"] == "matched":
            epss = tuple(G4.ARMS[G4b.MATCHED_ARM]["eps"])
            note = "G4b.VARIANTS[%r] role=matched, G4.ARMS[%r]['eps']" % (
                vname, G4b.MATCHED_ARM)
        else:
            epss = tuple(G4b.EPS_FALLBACK)
            note = ("G4b.VARIANTS[%r] role=retuned, G4b.EPS_FALLBACK (the real "
                    "levels come from the measured Gamma(eps_0) and must be "
                    "recosted after G4 analyze)" % vname)
        rows.append(sweep_row(
            "G4b %s [%s] (%d levels x %d seeds)"
            % (vname, v["role"], len(epss), n_seeds_g4b),
            G4b.MAT, epss, v["box"], n_seeds_g4b, note=note))

    # G6: one run per (label, eps in {eps_work, 0}) with n_realisations at T > 0
    g6_labels = _labels_in_configs(os.path.join(HERE, "runs",
                                                "G6_clean_seed.py"), "runs")
    n_thermal = int(G6.N_THERMAL_REALISATIONS)
    n_g6 = 0
    for lbl in g6_labels:
        nreal = n_thermal if lbl.startswith("T") and not lbl.startswith("T0") \
            else 1
        n_g6 += nreal + 1                       # + the eps_0 = 0 control
    t_g6, _, _, _ = plan_ns(G6.MAT, G6.EPS_WORK, G6.BOX)
    lo, hi = wall(n_g6 * t_g6, G6.BOX["NX"])
    rows.append(dict(label="G6 seeding (%s, each with an eps_0 = 0 control)"
                     % ", ".join(g6_labels), configs=n_g6,
                     physics_ns=n_g6 * t_g6, NX=int(G6.BOX["NX"]),
                     gpu_lo=lo, gpu_hi=hi, detail=[],
                     note="G6 runs() CONFIGS x (1 + eps_0=0 control), "
                          "T_run from the planner at G6.EPS_WORK",
                     per_run_ns=[round(t_g6, 2)]))

    # G9: one run per injection configuration, all at G9.T_RUN
    g9_labels = _labels_in_configs(os.path.join(HERE, "runs",
                                                "G9_idler_injection.py"),
                                   "runs")
    t_g9 = float(G9.T_RUN) * 1e9
    lo, hi = wall(len(g9_labels) * t_g9, G9.BOX["NX"])
    rows.append(dict(label="G9 idler injection (%s)" % ", ".join(g9_labels),
                     configs=len(g9_labels),
                     physics_ns=len(g9_labels) * t_g9,
                     NX=int(G9.BOX["NX"]), gpu_lo=lo, gpu_hi=hi, detail=[],
                     note="G9 runs() CONFIGS at G9.T_RUN", per_run_ns=[t_g9]))
    return rows


def totals(rows):
    return dict(configs=int(sum(r["configs"] for r in rows)),
                physics_us=float(sum(r["physics_ns"] for r in rows)) / 1000.0,
                gpu_lo=float(sum(r["gpu_lo"] for r in rows)),
                gpu_hi=float(sum(r["gpu_hi"] for r in rows)))


def run_plan_total(path=RUN_PLAN):
    """The total row of RUN_PLAN section 5, parsed, so the two can be compared."""
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"\|\s*\*\*total\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*"
                  r"\*\*([0-9.]+)\s*us\*\*\s*\|\s*\*\*([0-9.]+)"
                  r"[–-]([0-9.]+)\*\*\s*\|", txt)
    if not m:
        return None
    return dict(configs=int(m.group(1)), physics_us=float(m.group(2)),
                gpu_lo=float(m.group(3)), gpu_hi=float(m.group(4)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare the derived total with RUN_PLAN sec.5")
    ap.add_argument("--json", default=None, help="also write the rows as JSON")
    a = ap.parse_args(argv)

    rows = derive()
    tot = totals(rows)
    print("=" * 78)
    print("CAMPAIGN COST -- derived from the live run scripts")
    print("=" * 78)
    print("%-52s %5s %9s %11s" % ("block", "cfgs", "physics", "GPU-h"))
    for r in rows:
        print("%-52s %5d %7.0f ns %5.1f-%5.1f"
              % (r["label"][:52], r["configs"], r["physics_ns"], r["gpu_lo"],
                 r["gpu_hi"]))
        if r["detail"]:
            print("      eps_0 %s -> T_run %s ns"
                  % (", ".join("%.0e" % d["eps0"] for d in r["detail"]),
                     ", ".join("%.1f" % d["t_run_ns"] for d in r["detail"])))
        print("      [%s]" % r["note"])
    print("-" * 78)
    print("TOTAL: %d configurations, %.1f us of physics, %.0f - %.0f GPU-hours"
          % (tot["configs"], tot["physics_us"], tot["gpu_lo"], tot["gpu_hi"]))
    print("  = %.1f - %.1f days on one GPU; the campaign is checkpointed at "
          "block granularity." % (tot["gpu_lo"] / 24, tot["gpu_hi"] / 24))
    print("ASSUMPTION: wall time scales linearly in NX from the NX=1024 logs. "
          "Not verified at NX=1400/1500/2100; verify on the first run.")
    print("WINDOWS: costs use total planned runtime. The planner's frequency "
          "window and predicted linear lifetime are not measured fit intervals; "
          "the common fitter selects the early interval from the record.")
    worst = max(max(r["per_run_ns"]) for r in rows if r["per_run_ns"])
    print("STORAGE: my_xt is float32 (n_t x NX). Worst single run "
          "(%.0f ns @ 20 ps, NX=%d) = %.0f MB."
          % (worst, G4.BOX["NX"], worst * 1e-9 / 20e-12 * G4.BOX["NX"] * 4 / 1e6))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, total=tot), fh, indent=1)
        print("wrote %s" % a.json)

    if a.check:
        want = run_plan_total()
        print("-" * 78)
        if want is None:
            print("CHECK FAILED: could not parse the total row of %s"
                  % os.path.basename(RUN_PLAN))
            return 2
        bad = []
        if want["configs"] != tot["configs"]:
            bad.append("configs %d vs %d" % (tot["configs"], want["configs"]))
        if abs(want["physics_us"] - tot["physics_us"]) > 0.05:
            bad.append("physics %.2f vs %.2f us" % (tot["physics_us"],
                                                    want["physics_us"]))
        if abs(want["gpu_lo"] - tot["gpu_lo"]) > 1.0 or \
                abs(want["gpu_hi"] - tot["gpu_hi"]) > 1.0:
            bad.append("GPU-hours %.1f-%.1f vs %.1f-%.1f"
                       % (tot["gpu_lo"], tot["gpu_hi"], want["gpu_lo"],
                          want["gpu_hi"]))
        if bad:
            print("CHECK FAILED: derived table does not match RUN_PLAN sec.5: %s"
                  % "; ".join(bad))
            return 1
        print("CHECK OK: derived %d configs / %.1f us / %.0f-%.0f GPU-h matches "
              "RUN_PLAN sec.5 (%d / %.1f us / %.0f-%.0f)"
              % (tot["configs"], tot["physics_us"], tot["gpu_lo"], tot["gpu_hi"],
                 want["configs"], want["physics_us"], want["gpu_lo"],
                 want["gpu_hi"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
