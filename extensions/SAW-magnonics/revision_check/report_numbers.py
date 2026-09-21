"""report_numbers.py -- prints EVERY number quoted in model_comparison.md.

Run after `python model_comparison.py --stage all` (which fills out/mc_state.npz
and out/results.json).  Each section header matches a section of the write-up,
so every quoted number has one command behind it:

    python report_numbers.py            > out/report_numbers.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_comparison as mc                                    # noqa: E402
import saw_analysis as sa                                        # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=230)
Z = np.load(os.path.join(mc.OUT, "mc_state.npz"), allow_pickle=True)
FITS = Z["fits"].item()
RES = json.load(open(os.path.join(mc.OUT, "results.json")))


def h(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


h("0. GRID AND PUMP (verification of the brief's numbers)")
print("N_x = %d, dx = %g m, L = %.6f um" % (mc.NX, mc.DX, mc.NX * mc.DX * 1e6))
print("dk            = %.6f um^-1" % sa.to_inv_um(mc.DK))
q = 2 * np.pi * 2 * mc.F0 / mc.V_SAW
print("q_pump        = %.6f um^-1   q/dk = %.6f" % (sa.to_inv_um(q), q / mc.DK))
print("k_SAW/2       = %.6f um^-1   = %.6f dk (off-grid, between bins 4 and 5)"
      % (sa.to_inv_um(q / 2), q / 2 / mc.DK))
print("bin4, bin5    = %.6f, %.6f um^-1" % (sa.to_inv_um(4 * mc.DK), sa.to_inv_um(5 * mc.DK)))
print("bin4+bin5     = %.6f um^-1 = 9 dk (GRID 9th harmonic)" % sa.to_inv_um(9 * mc.DK))
print("9dk - q       = %.6f um^-1 = %.4f%% = %.4f dk"
      % (sa.to_inv_um(9 * mc.DK - q), 100 * (9 * mc.DK - q) / q, (9 * mc.DK - q) / mc.DK))
print("dk/q          = %.4f%%   (the k resolution)" % (100 * mc.DK / q))

h("1. THE ANALYSIS OBJECT AND THE BAND WEIGHTS (eps = 7e-5)")
f = FITS["7e-05"]
A = f["A"]
P = np.abs(A) ** 2
print("blocks start at t = %s ns" % np.round(f["tstart"] * 1e9, 3))
print("|a_j| per block, bins 1..7:"); print(np.abs(A))
print("arg a_j per block (deg), bins 1..7:"); print(np.angle(A, deg=True))
print("pooled power fraction over bins 1..7:", np.round(P.sum(0) / P.sum(), 4))
print("bins 4+5 pooled = %.4f ; bin 6 alone = %.4f ; bin 3 = %.4f"
      % ((P.sum(0)[3] + P.sum(0)[4]) / P.sum(), P.sum(0)[5] / P.sum(), P.sum(0)[2] / P.sum()))
print("measured out-of-band floor (bins %d..%d) rms |a| = %.6g"
      % (mc.FLOOR_BINS[0], mc.FLOOR_BINS[-1],
         np.sqrt((np.abs(f["resid_floor"]) ** 2).mean())))
print("per-bin |a| / floor_rms (blocks x bins):")
print(np.abs(A) / np.sqrt((np.abs(f["resid_floor"]) ** 2).mean()))

h("2. STRUCTURAL DIAGNOSTIC: adjacent-bin phase step (falsifiable)")
for tag in FITS:
    print("eps=%s  core d(arg) over bins 3-4,4-5,5-6,6-7 (deg), per block:" % tag)
    print(np.round(FITS[tag]["phase"]["core_dphi"], 2))
    print("   flips (>90 deg) in core: %d of %d"
          % (FITS[tag]["phase"]["n_flips_core"], FITS[tag]["phase"]["core_dphi"].size))
print("a single off-grid wave REQUIRES exactly one ~180 deg step; control (d)")
print("gives -179.8 deg at the straddled pair and +0.2 deg elsewhere.")
print("ALL adjacent steps (bins 1-2 ... 6-7), deg, and the joint step/peak test:")
for tag in FITS:
    AA = FITS[tag]["A"]
    ph = (np.diff(np.angle(AA, deg=True), axis=1) + 180) % 360 - 180
    print("eps=%s" % tag); print(np.round(ph, 2))
    for r in mc.step_vs_peak(AA):
        print("   block %d: largest step %+7.1f deg at bins %d-%d -> kappa in that"
              " interval; peak bin %d; |a_peak|/|a_step| observed %.3f but a single"
              " wave allows at most %.3f  -> exceeded %.1fx"
              % (r["block"], r["step_deg"], r["step_pair"][0], r["step_pair"][1],
                 r["peak_bin"], r["ratio_observed"], r["ratio_max_single"],
                 r["excess"]))

h("3. FITS, HELD-OUT PREDICTION ERROR, SELECTION")
for tag in FITS:
    ff = FITS[tag]
    print("--- eps0 = %.0e   in-band training power = %.6g   m_rms = %.4g  max|m_y| = %.4g"
          % (ff["eps0"], ff["train_power"], ff["m_rms"], ff["m_max"]))
    for nm in ("preregistered", "amended"):
        e = ff[nm]
        print("  [%s] E1 = %.4f   E2 = %.4f  -> %s"
              % (nm, e["pipe"]["E1"], e["pipe"]["E2"], e["pipe"]["selected"]))
        print("      secondary split C={4,5} E={1,2,3,6,7}: E1 = %.4f  E2 = %.4f -> %s"
              % (e["pipe2"]["E1"], e["pipe2"]["E2"], e["pipe2"]["selected"]))
        for n in (1, 2, 3):
            nn = e["nested"][n]
            print("      N=%d kappa = %s  in-band explained = %.4f  E_held = %.4f  k_real = %d"
                  % (n, np.round(nn["kappa"], 4), nn["frac_explained"],
                     nn["E_held"], nn["n_real_params"]))
print("\nE = 1 is the ZERO-PREDICTION baseline; E < 1 is skill; E > 1 is actively wrong.")

h("4. IN-BAND ADEQUACY AGAINST THE MEASURED FLOOR (secondary criterion)")
for tag in FITS:
    ff = FITS[tag]
    s2 = float((np.abs(ff["resid_floor"]) ** 2).mean())
    nC = len(mc.TRAIN_BLOCKS) * len(mc.BAND)
    print("eps=%s  sigma2_floor = %.4g   n_complex = %d" % (tag, s2, nC))
    for nm in ("preregistered", "amended"):
        r = [ff[nm]["nested"][n]["ss"] / (nC * s2) for n in (1, 2, 3)]
        print("   [%s] adequacy ratio N=1,2,3 : %s   (adequate ~ 1)"
              % (nm, np.round(r, 2)))
    Atr = ff["A"][mc.TRAIN_BLOCKS]
    p = (np.abs(Atr) ** 2).sum(0)
    o = np.argsort(p)[::-1]
    cum = np.cumsum(p[o])
    print("   on-grid (box-eigenmode) ranking: bins %s" % (mc.BAND[o]).tolist())
    print("   cumulative power fraction      : %s" % np.round(cum / ff["train_power"], 4))
    print("   adequacy ratio after N bins    : %s"
          % np.round([(ff["train_power"] - c) / (nC * s2) for c in cum], 2))

h("5. RESIDUAL SPECTRA (is the residual white, or coherent at a k?)")
for tag in ("7e-05",):
    ff = FITS[tag]
    Atr = ff["A"][mc.TRAIN_BLOCKS]
    print("eps=%s  mean |a_j| over training blocks, bins 1..7:" % tag)
    print("   data      ", np.round(np.abs(Atr).mean(0), 5))
    for nm in ("preregistered", "amended"):
        for n in (1, 2):
            k = ff[nm]["nested"][n]["kappa"]
            _, fit, _ = mc._varpro(k, Atr, mc.BAND, 1e-6, full=True)
            print("   %-13s M%d resid" % (nm, n), np.round(np.abs(Atr - fit).mean(0), 5))
    print("   floor rms  %.5f (flat)" % np.sqrt((np.abs(ff["resid_floor"]) ** 2).mean()))

h("6. AUXILIARY INFORMATION CRITERIA (do NOT lean on these)")
n = 2 * len(mc.TRAIN_BLOCKS) * len(mc.BAND)
for nm in ("preregistered", "amended"):
    for N in (1, 2, 3):
        ss = FITS["7e-05"][nm]["nested"][N]["ss"]
        k = FITS["7e-05"][nm]["nested"][N]["n_real_params"]
        print("%-14s N=%d RSS=%.6f k_real=%2d AIC=%8.2f BIC=%8.2f"
              % (nm, N, ss, k, n * np.log(ss / n) + 2 * k,
                 n * np.log(ss / n) + k * np.log(n)))
print("n_real_residual_components = %d.  M2's optimum sits ON the minimum-separation"
      % n)
print("guard, i.e. exactly the boundary where its parameters are unidentifiable;")
print("the usual AIC/BIC asymptotics are not guaranteed there.")

h("7. CONTROL (d): single wave -> FALSE-POSITIVE RATE (kappa scan)")
ps = RES["power"]["d_scan"]
print("  kappa   noise    model    FP     single  indist   E1_med    E1_q95    flips")
for k in sorted(ps):
    v = ps[k]; kk, nz, mo = k.split("|")
    print("  %-7s %-8s %-4s  %.3f  %.3f   %.3f  %9.4f %9.4g   %.2f"
          % (kk, nz, mo, v["fp_rate"], v["single_rate"], v["indist_rate"],
             v["E1_med"], v["E1_q95"], v["core_flips_mean"]))

h("8. CONTROL (e): two components -> DETECTION RATE vs relative amplitude r")
c = RES["controls"]
rv = [0.2, 0.5, 1.0, 1.2, 1.36]
print("OFF-GRID pairs (pre-registered rule):")
for nz in ("floor", "M1resid"):
    for mo in ("R1", "R2", "R3"):
        for case in ("exact_bins_4_5", "offgrid_sep0.5", "offgrid_sep1.0", "offgrid_sep1.5"):
            y = [c["e|%s|%s|%s|%.2f" % (nz, mo, case, r)]["det_rate"] for r in rv]
            print("  %-8s %-3s %-16s det(r=%s) = %s" % (nz, mo, case, rv, np.round(y, 3)))
print("ON-GRID pairs (floor noise, R3) -- pre-registered rule AND the secondary")
print("in-band adequacy ratio:")
for k in sorted(RES["power"]["e_ongrid"]):
    v = RES["power"]["e_ongrid"][k]
    print("  %-22s det=%.3f single=%.3f indist=%.3f  adeq_M1=%8.2f adeq_M2=%6.2f"
          % (k, v["det_rate"], v["single_rate"], v["indist_rate"],
             v["adeq1_med"], v["adeq2_med"]))

h("9. PARAMETRIC BOOTSTRAP -- exceedance UNDER THE ASSUMED RESIDUAL MODEL")
for k, v in RES["boot"].items():
    print("  %-14s p(E1>=obs)=%.3f  p(E1/E2>=obs)=%.3f | E1_obs=%.3f med=%.4f q95=%.4g"
          % (k, v["p_E1_ge_obs"], v["p_ratio_ge_obs"], v["E1_obs"], v["E1_med"], v["E1_q95"]))

h("10. FREQUENCY, NON-CIRCULAR (projection of RAW m(x,t), no f_K pre-slice)")
fr = RES["freq"]
for nm in ("M1_pre", "M1_amd", "M2_amd", "grid"):
    for e in fr[nm]:
        print("  %-7s kappa=%8.4f  f = %.6f GHz  (bin %.2f MHz, sub-bin %+.3f, snr %.0f)"
              % (nm, e["kappa"], e["f_peak"] / 1e9, e["df_bin"] / 1e6,
                 e["subbin_delta"], e["snr"]))
g = {e["kappa"]: e["f_peak"] for e in fr["grid"]}
print("  pair frequency sums (all on-grid pairs summing to bin 9):")
for a, b in ((2, 7), (3, 6), (4, 5)):
    print("    bins %d+%d : %.6f + %.6f = %.6f GHz  -> f_p excess %+6.1f MHz"
          % (a, b, g[str(float(a))] / 1e9 if str(float(a)) in g else g[float(a)] / 1e9,
             g[float(b)] / 1e9, (g[float(a)] + g[float(b)]) / 1e9,
             (g[float(a)] + g[float(b)] - 6e9) / 1e6))
d = fr["unpumped_dispersion_Hz"]
vals = [d[str(k)] for k in (2, 3, 4, 5, 6, 7)]
print("  unpumped sim45 B=50 mT: f(0) = %.6f GHz, f(9) = %.6f GHz"
      % (d["0"] / 1e9, d["9"] / 1e9))
print("  unpumped f(k) bins 2..7 = %s GHz, spread = %.1f MHz, sim45 bin = %.2f MHz"
      % (np.round(np.array(vals) / 1e9, 6), (max(vals) - min(vals)) / 1e6,
         fr["unpumped_df_bin_Hz"] / 1e6))
r = sa.resolution_report(376, 20e-12, "hann", growth_rate=2.93e8)
print("  sim40 record: %d samples x 20 ps -> bin %.2f MHz, Hann -3 dB %.2f MHz,"
      % (r["n_samples"], r["df_bin"] / 1e6, r["fwhm_3dB_Hz"] / 1e6))
print("  growth HWHM %.1f MHz -> effective line width %.1f MHz"
      % (r["growth_HWHM_Hz"] / 1e6, r["effective_width_Hz"] / 1e6))
print("  CONSEQUENCE OF THE 2026-09-18 SIGN FIX (audit sec. 4), stated here so")
print("  the numbers above are not read as physics:")
print("    * before the fix this estimator searched the POSITIVE frequencies of")
print("      an exp(-i 2 pi f t) transform and so returned the COUNTER-")
print("      PROPAGATING branch. bin 4 read 3.034726 GHz, bin 5 read 3.035119,")
print("      sum 6.069845 GHz = +69.8 MHz above 2 f_K. That +70 MHz was an")
print("      artefact of the branch error and MUST NOT be cited as a physical")
print("      mismatch against the candidate mechanism.")
print("    * after the fix the sum is 5.994198 GHz = -5.8 MHz. This MUST NOT be")
print("      cited as evidence FOR a sum rule either. This estimator has no")
print("      error budget: its bias and spread under the actual record length,")
print("      growth envelope, Hann window and parabolic interpolation have not")
print("      been measured on synthetic controls carrying the same envelope and")
print("      window. The 133 MHz fft bin spacing is NOT that error bar, in")
print("      either direction. Until that control-based budget exists the")
print("      frequency sum is a diagnostic and decides nothing.")

h("11. AUXILIARY, EXPLICITLY NON-DISCRIMINATING")
a = RES["aux"]
print("  published-style f slice used by saw_analysis: %.6f GHz" % (a["f_slice_Hz"] / 1e9))
for pad in (1, 8, 16):
    print("  %2dx zero padding: %d local maxima, top = %s"
          % (pad, a["pad%d_npeaks" % pad],
             [[round(x, 4), round(y, 4)] for x, y in a["pad%d_peaks" % pad]]))
print("  arg(M5/M4) = %.3f +- %.3f deg ; |M5/M4| = %.4f, CV = %.3e"
      % (a["arg_M5_over_M4_deg_mean"], a["arg_M5_over_M4_deg_std"],
         a["abs_M5_over_M4_mean"], a["abs_M5_over_M4_cv"]))
for j in range(1, 10):
    print("  Gamma(bin %d) = %+.4e 1/s  (r2 = %.3f)"
          % (j, a["Gamma_bin%d" % j], a["Gamma_bin%d_r2" % j]))

h("12. ROBUSTNESS TO THE ANALYSIS CHOICES (eps = 7e-5)")
for row in RES["robust"]:
    for nm in ("preregistered", "amended"):
        v = row[nm]
        print("  %-28s [%-13s] k1=%8.4f k2=%s expl1=%.3f expl2=%.3f E1=%9.4f E2=%11.4f -> %-22s flips=%d"
              % (row["variant"], nm, v["kappa1"], np.round(v["kappa2"], 3),
                 v["frac_expl_M1"], v["frac_expl_M2"], v["E1"], v["E2"],
                 v["selected"], row["core_flips"]))
print("\n[done]")
