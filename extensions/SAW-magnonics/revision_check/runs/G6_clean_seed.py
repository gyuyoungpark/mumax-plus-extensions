"""G6 -- seeding: what sim43_clean_seed.py does, what has to change, and the
finite-temperature variant.

WHAT src/sim43_clean_seed.py DOES (read, not paraphrased from its docstring)
  build():  rng = np.random.default_rng(rng_seed)
            mag[0] = 1.0
            mag[1] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
            mag[2] = SEED_AMP * rng.standard_normal((NZ, NY, NX))
            magnet.magnetization = mag           # |m| > 1, engine renormalises
            magnet.temperature = 0.0
  run_one(): rng_seed = 1000 * hash(label) % 7919 + s
  savez():   my_xt, done, eps0, seed_amp, B0, direction, f_K, k_saw, f_saw,
             CX, DT_REC, NX, material          <-- no rng seed anywhere

WHAT MUST CHANGE, and why

1. hash() is salted.  CPython randomises str hashing per process unless
   PYTHONHASHSEED is set, so `1000*hash(label) % 7919 + s` is a DIFFERENT
   integer in every process.  Measured here, three processes, same string
   'YIG_fwd':   6524545326354041675 / -9040856993033622291 / 7888820862419940537
   and with PYTHONHASHSEED=0 twice: -707780873772855074 both times.
   FIX: pass explicit integer seeds.  Never call hash() on a label.

2. The seed value is never written to the output.  The npz records seed_amp but
   not rng_seed, so even a correctly generated seed could not be recovered.
   FIX: rng_seed, the mode list and the mode phases go into every npz.

3. The seed is per-cell Gaussian and therefore mostly invisible.  The observable
   is the y,z-averaged m_y(x,t); 7/8 of a per-cell Gaussian's energy sits at
   k_y != 0 and is averaged away.  FIX: seed y/z-uniform, band-limited in k.

4. The amplitude is not physical.  sigma = 1e-4 per cell over NX*NY modes gives
   a per-mode amplitude of 1.105e-6 (NX=1024).  The 300 K thermal amplitude of
   the SAME mode in the SAME box is 1.151e-2.  The seed is 9.25 e-foldings
   below the bath -- an equivalent magnon temperature of 2.8e-6 K.  Calling it
   a "small physical seed" is wrong; it is a numerical seed one decade above
   this build's single-precision floor (_mumaxpluscpp_single).  Needing ~9
   extra e-foldings is exactly what turns into pressure to inflate the
   coupling (CLAUDE.md sec.4).  FIX: set the per-mode amplitude from
   thermal_mode_amplitude() at a stated temperature, capped so the run stays
   linear for the frequency window it needs.

5. |m| is left > 1 for the engine to renormalise.  FIX: normalise in numpy so
   the initial condition is exactly what was intended.

FINITE TEMPERATURE -- A LIMIT THAT CANNOT BE FIXED FROM PYTHON
  src/physics/ferromagnet.cpp:118-120 seeds the Langevin generator with
      curandSetPseudoRandomGeneratorSeed(randomGenerator,
          static_cast<int>(std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count()));
  There is no binding for it (only wrap_voronoi.cpp takes a `seed`).  So a
  T > 0 run's random stream is NOT reproducible and its seed is NOT
  RECORDABLE.  T > 0 runs are therefore an ENSEMBLE: record the count and treat
  every realisation as independent, and do not claim reproducibility.  The
  one-line fix, if the workstation is willing to rebuild, is to expose that
  seed; it is NOT applied here.

Stages
  selftest   CPU only, no engine.  Seed flatness, cross-process reproducibility,
             amplitude calibration.  Run this first; it needs no GPU.
  runs       T = 0 (physical amplitude), T = 0 (small amplitude), T = 30 K,
             T = 300 K -- each with eps_0 = eps_work AND eps_0 = 0.
             The eps_0 = 0, T > 0 run is the control sim41 never had.
  analyze    band contrast against the eps_0 = 0 control at the same T.

Usage
    python G6_clean_seed.py selftest
    python G6_clean_seed.py runs
    python G6_clean_seed.py analyze
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H                                              # noqa: E402
import saw_analysis as SA                                         # noqa: E402

BOX    = H.BOX_PRIMARY
Q      = BOX["q"]
K_HALF = Q / 2
F_SAW  = BOX["f_saw"]
MAT    = H.YIG_LIT
DT_REC = 20e-12
CKPT   = os.path.join(H.OUT_DIR, "G6")
os.makedirs(CKPT, exist_ok=True)
V_TOT  = BOX["NX"] * H.NY * BOX["dx"] * H.CY * H.CZ

RNG_SEED_PRIMARY = 20260931
EPS_WORK = 1e-4                       # the literature-YIG working point
N_THERMAL_REALISATIONS = 3            # T>0 has no recordable seed: use a count


def _prov(p):
    return H.prov_kw(__file__, p)


# ------------------------------------------------------------- selftest -----
def selftest():
    B0 = H.kittel_field(3.0e9, MAT["MS"])
    a_phys = H.thermal_mode_amplitude(K_HALF, 300.0, B0, MAT["MS"],
                                      MAT["AEX"], V_TOT)
    print("  box %s  V_tot=%.4e m^3" % (BOX["tag"], V_TOT))
    print("  a_th(q/2, 300 K) = %.4e   a_th(q/2, 1 K) = %.4e"
          % (a_phys, H.thermal_mode_amplitude(K_HALF, 1.0, B0, MAT["MS"],
                                              MAT["AEX"], V_TOT)))

    # (1) hash() salting -- the sim43 defect, demonstrated not asserted
    hs = [subprocess.run([sys.executable, "-c",
                          "print(hash('YIG_fwd'))"], capture_output=True,
                         text=True).stdout.strip() for _ in range(3)]
    print("  hash('YIG_fwd') in 3 fresh processes: %s  -> %s"
          % (hs, "VARIES (sim43 defect reproduced)" if len(set(hs)) > 1
             else "stable (PYTHONHASHSEED is pinned in this shell)"))

    # (2) reproducibility of make_seed across processes
    code = ("import sys;sys.path.insert(0,r'%s');import _harness as H,hashlib,"
            "numpy as np;m,_=H.make_seed(%d,%d,%r,1e-3,%r);"
            "print(hashlib.sha256(np.ascontiguousarray(m)).hexdigest())"
            % (os.path.dirname(os.path.abspath(__file__)), RNG_SEED_PRIMARY,
               BOX["NX"], BOX["dx"], 2 * Q))
    digs = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True).stdout.strip() for _ in range(2)]
    ok_repro = len(set(digs)) == 1 and digs[0]
    print("  make_seed sha256 across 2 fresh processes: %s -> %s"
          % (digs[0][:16] if digs[0] else "FAILED",
             "REPRODUCIBLE" if ok_repro else "NOT REPRODUCIBLE"))

    # (3) flatness in k inside the band, and zero outside
    mag, meta = H.make_seed(RNG_SEED_PRIMARY, BOX["NX"], BOX["dx"],
                            a_phys, 2 * Q)
    p = np.abs(np.fft.rfft(mag[1, 0, 0])) ** 2
    nm = meta["n_modes"]
    inb = p[1:nm + 1]
    out = p[nm + 1:]
    flat = float(inb.max() / inb.min())
    print("  seed: %d modes, per-cell sigma=%.4f, max canting=%.4f"
          % (nm, meta["sigma_cell_y"], meta["max_canting"]))
    print("  in-band power max/min = %.6f (1.0 = exactly flat); "
          "out-of-band/max = %.2e" % (flat, out.max() / p.max()))
    print("  |m| deviation from 1: %.2e"
          % np.abs(np.sqrt((mag ** 2).sum(0)) - 1).max())

    # (4) what temperature is each candidate amplitude?
    print("\n  %-34s %12s %12s" % ("seed", "a_mode", "T_equiv [K]"))
    for lbl, a in (("sim43 sigma=1e-4, NX=1024 (per mode)", 1e-4 / np.sqrt(1024 * 8)),
                   ("sim43 sigma=1e-4, NX=1400 (per mode)", 1e-4 / np.sqrt(1400 * 8)),
                   ("G6 physical (300 K)", a_phys),
                   ("G6 small (300 K / 100)", a_phys / 100),
                   ("single-precision floor ~1e-7", 1e-7)):
        print("  %-34s %12.4e %12.4e"
              % (lbl, a, H.seed_temperature(a, K_HALF, B0, MAT["MS"],
                                            MAT["AEX"], V_TOT)))

    passed = bool(ok_repro and flat < 1.0001 and out.max() / p.max() < 1e-20)
    np.savez(os.path.join(CKPT, "G6_selftest.npz"), done=True,
             gate_pass=passed, a_phys=a_phys, flatness=flat,
             hash_values=np.array(hs), seed_digests=np.array(digs),
             seed_meta=json.dumps(meta, default=str),
             **_prov(dict(box=BOX, B0=B0)))
    print("\n  GATE G6.1 %s (seed reproducible across processes, flat in band, "
          "zero outside)" % ("PASS" if passed else "FAIL"))
    return passed


# ----------------------------------------------------------------- runs -----
def runs():
    # See G9/G4b: one checked accessor, no Kittel fallback (audit sec.8, P0-2).
    import G4_commensurate_onset as G4                            # noqa: PLC0415
    B0 = G4.b0star_for_dependents()
    a_phys = H.thermal_mode_amplitude(K_HALF, 300.0, B0, MAT["MS"],
                                      MAT["AEX"], V_TOT)
    g = H.gamma_estimate(MAT, EPS_WORK)
    win = H.auto_window(g)
    a_lin, why = H.choose_seed_amplitude(MAT, EPS_WORK, a_phys, t_window=win)
    plan = H.plan_runtime(MAT, EPS_WORK, a_lin, t_window=win, t_min=40e-9)
    # Same rule as G4/G4b: a blocking plan["status"]
    # (INSUFFICIENT_LINEAR_WINDOW / NO_LINEAR_WINDOW) stops the stage here.
    if plan["status"] in H.PLAN_BLOCKING:
        raise H.PlanRefused("G6 seeding at eps_work = %.3e: run plan blocked "
                            "(%s) -- %s"
                            % (EPS_WORK, plan["status"], plan["note"]))
    nt = H.nt_records(plan, DT_REC, "G6 seeding eps_work=%.3e" % EPS_WORK)

    # (label, temperature, deterministic seed amplitude, n_realisations)
    CONFIGS = [
        ("T0_phys",  0.0,   a_lin,         1),
        ("T0_small", 0.0,   a_lin / 100,   1),
        ("T30",      30.0,  a_lin / 100,   N_THERMAL_REALISATIONS),
        ("T300",     300.0, a_lin / 100,   N_THERMAL_REALISATIONS),
    ]
    print("  B0=%.4f mT  eps_work=%.0e  T_run=%.0f ns  a_lin=%.3e  [%s]"
          % (B0 * 1e3, EPS_WORK, plan["t_run"] * 1e9, a_lin, why))
    for lbl, T, a_seed, nreal in CONFIGS:
        for eps in (EPS_WORK, 0.0):          # 0.0 = the missing background control
            for r in range(nreal if eps > 0 else 1):
                tag = "%s_eps%.0e_r%d" % (lbl, eps, r)
                path = os.path.join(CKPT, "G6_%s.npz" % tag)
                rs = RNG_SEED_PRIMARY + 100 * r
                mag, smeta = H.make_seed(rs, BOX["NX"], BOX["dx"], a_seed,
                                         2 * Q)
                T_eq = H.seed_temperature(a_seed, K_HALF, B0, MAT["MS"],
                                          MAT["AEX"], V_TOT)
                params = dict(box=BOX, material=MAT, B0=B0, eps0=eps,
                              f_saw=F_SAW, temperature=T, realisation=r,
                              seed_class=lbl,
                              T_run=plan["t_run"], DT_REC=DT_REC,
                              DT_STEP=H.DT_STEP, rng_seed_numpy=rs, seed=smeta,
                              a_seed=a_seed, a_phys_300K=a_phys,
                              seed_equiv_temperature_K=T_eq,
                              langevin_seed=("NOT RECORDABLE: "
                                             "src/physics/ferromagnet.cpp:118 "
                                             "seeds curand from the wall clock; "
                                             "no binding exposes it")
                              if T > 0 else "n/a (T=0)")
                extra = dict(eps0=eps, B0=B0, temperature=T, realisation=r,
                             rng_seed_numpy=rs, a_seed=a_seed,
                             seed_class=lbl,
                             seed_meta=json.dumps(smeta, default=str),
                             **_prov(params))
                br = H.BlockRun(path, nt, BOX["NX"], DT_REC,
                                manifest=params)
                if br.done:
                    print("  [skip] %s" % tag)
                    continue
                print("  ---- %s  T=%.0f K  a=%.3e (T_eq=%.3g K)"
                      % (tag, T, a_seed, T_eq))
                world, magnet, sm = H.build(BOX, MAT, B0, eps, F_SAW, mag,
                                            temperature=T,
                                            conditions=br.conditions)
                extra["saw_meta"] = json.dumps(sm, default=str)
                br.integrate(world, magnet, extra)


# -------------------------------------------------------------- analyze -----
def _class_from_name(fn):
    """Seed class from the file name, for npz files written before seed_class
    was recorded.  'G6_T0_small_eps1e-04_r0.npz' -> 'T0_small'."""
    b = os.path.basename(fn)
    if b.startswith("G6_"):
        b = b[3:]
    return b.split("_eps")[0] or "UNKNOWN"


def analyze():
    """Band contrast against the eps_0 = 0 control at the SAME temperature, the
    SAME seed class and the SAME seed amplitude.

    At T = 300 K with these cells the per-cell thermal canting is large, so
    max|m_y| is useless as an observable -- sim41 read 0.20-0.22 from t = 10 ns
    onwards at every strain and no growth rate could be extracted from it.  The
    observable here is instead the power in the disjoint 'plus' band at
    f = Omega/2 DIVIDED by the same quantity in the eps_0 = 0 run of the same
    seed class, the same amplitude and the same temperature.

    REFUTES the pumped band if: the ratio is consistent with 1 within the
    realisation-to-realisation scatter of the T > 0 ensemble.

    FIXED 2026-09-18 (independent audit, sec.8, P1-1).  The grouping key was
    (temperature, eps_0) only, so T0_phys and T0_small -- two seed classes whose
    amplitudes differ by 100x -- were pooled as n = 2 repeats of one condition,
    and the denominator was the MEAN of their two different controls.  Fed true
    power gains of 4 and 16 the function returned "4.0012 +- 5.6540, n = 2",
    which is neither.  The key now carries the seed class AND the seed
    amplitude; only realisations of the SAME condition are averaged (that is
    what the T > 0 ensemble is), a condition whose control was run at a
    different amplitude is NOT_DETERMINABLE rather than ratioed, and the
    across-class summary is a TABLE -- there is no mean across classes.
    """
    store = {}
    invalid = []
    for fn in sorted(os.listdir(CKPT)):
        skip = ("selftest" in fn or "summary" in fn)
        if not fn.startswith("G6_") or not fn.endswith(".npz") or skip:
            continue
        d = np.load(os.path.join(CKPT, fn), allow_pickle=True)
        a_seed = float(d["a_seed"]) if "a_seed" in d else float("nan")
        if (not np.isfinite(a_seed) or a_seed < 0 or "seed_class" not in d
                or "done" not in d or not bool(d["done"])):
            invalid.append(dict(status="NOT_DETERMINABLE", file=fn,
                                seed_class=str(d.get("seed_class", "unknown")),
                                T=float(d.get("temperature", np.nan)), eps0=float(d.get("eps0", np.nan)),
                                a_seed=a_seed, n=1, ratio_mean=float("nan"), ratio_sd=float("nan"),
                                reason="incomplete run or missing/nonfinite seed condition"))
            continue
        my = d["my_xt"].astype(float)
        nt = my.shape[0]
        k_ax, f_ax, M = SA.spectrum(my, BOX["dx"], DT_REC, window="hann",
                                    trange=(nt // 2, nt))
        _, sl = SA.f_slice(M, f_ax, F_SAW / 2)
        bc = SA.band_content(k_ax, sl, K_HALF, Q, halfwidth=BOX["dk"] / 2,
                             quantity="power")
        cls = (str(d["seed_class"]) if "seed_class" in d
               else _class_from_name(fn))
        a_seed = float(d["a_seed"]) if "a_seed" in d else float("nan")
        key = (cls, float(d["temperature"]), float(d["eps0"]),
               "%.12e" % a_seed)
        store.setdefault(key, []).append(
            dict(file=fn, plus=float(bc["values"]["plus"]),
                 frac_plus=float(bc["fractions"]["plus"]),
                 total=float(bc["total"]), a_seed=a_seed))
        print("  %-44s %-9s T=%5.0f K eps=%.0e a=%.3e  plus=%.4e frac=%.4f"
              % (fn, cls, key[1], key[2], a_seed, bc["values"]["plus"],
                 bc["fractions"]["plus"]))

    rows = list(invalid)
    for (cls, T, eps, a_key), v in sorted(store.items()):
        if eps == 0.0:
            continue
        ctrl = store.get((cls, T, 0.0, a_key))
        if not ctrl:
            other = sorted(k for k in store
                           if k[0] == cls and k[1] == T and k[2] == 0.0)
            why = ("no eps_0 = 0 control for seed class %s at T = %.0f K and "
                   "a_seed = %s" % (cls, T, a_key))
            if other:
                why += ("; the only control(s) present were run at a_seed = %s "
                        "-- a different seed amplitude is NOT a control for "
                        "this condition"
                        % ", ".join(k[3] for k in other))
            print("  %-9s T=%5.0f K a=%s : ratio NOT_DETERMINABLE -- %s"
                  % (cls, T, a_key, why))
            rows.append(dict(status="NOT_DETERMINABLE", seed_class=cls, T=T,
                             eps0=eps, a_seed=float(a_key), n=len(v),
                             ratio_mean=float("nan"), ratio_sd=float("nan"),
                             reason=why))
            continue
        num = np.array([r["plus"] for r in v])
        den = np.array([r["plus"] for r in ctrl])
        if (not np.isfinite(num).all() or not np.isfinite(den).all()
                or not np.all(den > 0)):
            rows.append(dict(status="NOT_DETERMINABLE", seed_class=cls, T=T, eps0=eps,
                             a_seed=float(a_key), n=int(num.size), n_control=int(den.size),
                             ratio_mean=float("nan"), ratio_sd=float("nan"),
                             reason="pump-off band power must be finite and positive"))
            continue
        rows.append(dict(status="ok", seed_class=cls, T=T, eps0=eps,
                         a_seed=float(a_key), n=int(num.size),
                         n_control=int(den.size),
                         ratio_mean=float(num.mean() / den.mean()),
                         ratio_sd=(float(num.std(ddof=1) / den.mean())
                                   if num.size > 1 else float("nan")),
                         files=[r["file"] for r in v],
                         control_files=[r["file"] for r in ctrl]))

    if rows:
        print("")
        print("  band power / pump-off control, ONE ROW PER SEED CLASS "
              "(no mean across classes -- the classes are different "
              "conditions):")
        print("  %-10s %7s %10s %11s %11s %6s %s"
              % ("class", "T [K]", "a_seed", "ratio", "sd", "n", "status"))
        for r in rows:
            print("  %-10s %7.0f %10.3e %11.4f %11.4f %6d %s"
                  % (r["seed_class"], r["T"], r["a_seed"], r["ratio_mean"],
                     r["ratio_sd"], r["n"], r["status"]))
        np.savez(os.path.join(CKPT, "G6_summary.npz"), rows=json.dumps(rows),
                 **_prov(dict(stage="analyze")))
        print("  wrote G6_summary.npz")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="selftest",
                    choices=["selftest", "runs", "analyze"])
    a = ap.parse_args()
    print("=" * 78)
    print("G6 seeding -- band-limited, y-uniform, thermally calibrated, "
          "explicitly seeded")
    print("=" * 78)
    # Same handler and the same exit codes as every other script in runs/
    # (open item 10: G6 had no _gate.GateHalt handler at all).
    sys.exit(H.run_entry("G6." + a.stage,
                         {"selftest": selftest, "runs": runs,
                          "analyze": analyze}[a.stage]))
