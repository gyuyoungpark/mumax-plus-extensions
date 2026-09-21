"""STEP 1b -- onset, contrast at eps_0 = 1e-4, MR eps^2 slope check, figure.

Consumes out/step1_ipair.npz written by step1_ipair.py.
    python step1b_curves.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rc_common as rc

D = np.load(os.path.join(rc.OUT, "step1_ipair.npz"))
EPS = D["eps"]; LAB = [str(x) for x in D["labels"]]
BANDS = ["a_published", "b_union_nok0", "c_oneside_2dk", "d_bins45"]
SHORT = {"a_published": "(a) published 16dk, overlapping",
         "b_union_nok0": "(b) union 16dk, no k=0",
         "c_oneside_2dk": "(c) +k_pair, 2dk, no k=0",
         "d_bins45": "(d) bins +4,+5 only"}
I = {b: D["I_" + b] for b in BANDS}
i2, iF, iM = LAB.index("full_2fK"), LAB.index("full_fK"), LAB.index("mr_fK")

L = []; P = L.append
P("STEP 1b -- onset, contrast, MR eps^2 slope")
P("=" * 78)
P("command: python step1_ipair.py && python step1b_curves.py")
P("")
P("NORMALISATION.  The published Fig. 6(b) divides by I_pair(mr_fK, eps=1e-5)")
P("in the same band (make_fig6_v2.py:136, 'Inorm = Ip_mrK[0]') and labels it")
P("a NOISE FLOOR.  It is not one: there is no eps_0 = 0 run anywhere in data/,")
P("so no measured floor exists.  It is a driven MR run at eps_0 = 1e-5 whose")
P("f_K slice is 84.4% the k = 0 bin (step1_ipair_raw.txt).  The same divisor is")
P("used here so the curves are comparable with the published panel; the label")
P("must change.")
P("")
rows = []
for b in BANDS:
    Ib = I[b]; norm = Ib[0, iM]
    P("--- %s ---" % SHORT[b])
    P("  I_norm = I(mr_fK, 1e-5) = %.6e (absolute)" % norm)
    P("  eps_0    full_2fK/In      full_fK/In       mr_fK/In    2fK/MR   2fK/own(1e-5)")
    for ei, e in enumerate(EPS):
        P("  %.0e %14.5e %16.5e %15.5e %9.3f %10.4g"
          % (e, Ib[ei, i2] / norm, Ib[ei, iF] / norm, Ib[ei, iM] / norm,
             Ib[ei, i2] / Ib[ei, iM], Ib[ei, i2] / Ib[0, i2]))
    # onset
    n2 = Ib[:, i2] / Ib[0, i2]
    on10 = [e for ei, e in enumerate(EPS) if n2[ei] >= 10]
    on2 = [e for ei, e in enumerate(EPS) if Ib[ei, i2] >= 2 * Ib[ei, iM]]
    o10 = "%.0e" % on10[0] if on10 else "none in range"
    o2 = "%.0e" % on2[0] if on2 else "none in range"
    # the manuscript's "stays at the noise floor for eps<=7e-5" claim
    floor7 = Ib[3, i2] / norm          # eps = 7e-5, full_2fK, in floor units
    rise = Ib[4, i2] / Ib[3, i2]       # 1e-4 / 7e-5
    contrast = Ib[4, i2] / Ib[4, iM]   # eps=1e-4, full_2fK vs mr_fK
    lin = EPS <= 4e-4
    low = EPS <= 1e-4
    s_lin, r_lin, n_lin = rc.loglog_slope(EPS[lin], Ib[lin, iM])
    s_low, r_low, n_low = rc.loglog_slope(EPS[low], Ib[low, iM])
    s_all, r_all, n_all = rc.loglog_slope(EPS, Ib[:, iM])
    s_dec = np.log10(Ib[4, iM] / Ib[0, iM])   # 1e-5 -> 1e-4, exactly one decade
    P("  ONSET, first eps with full_2fK >= 10x its own 1e-5 value : %s" % o10)
    P("  ONSET, first eps with full_2fK >= 2x mr_fK at same eps   : %s" % o2)
    P("  full_2fK at eps=7e-5, in units of the plotted floor      : %.4g x"
      % floor7)
    P("  rise of full_2fK from 7e-5 to 1e-4                       : %.4g x "
      "(%.2f decades)" % (rise, np.log10(rise)))
    P("  CONTRAST at eps_0 = 1e-4, full_2fK / mr_fK               : %.3f"
      % contrast)
    P("  MR baseline log-log slope, eps <= 1e-4 (%d pts)          : %.4f "
      "(R2 = %.4f)   <-- the linear regime" % (n_low, s_low, r_low))
    P("  MR baseline log-log slope, eps <= 4e-4 (%d pts)          : %.4f "
      "(R2 = %.4f)" % (n_lin, s_lin, r_lin))
    P("  MR baseline log-log slope, all 8 pts                     : %.4f "
      "(R2 = %.4f)" % (s_all, r_all))
    P("  MR baseline two-point decade slope 1e-5 -> 1e-4          : %.4f"
      % s_dec)
    P("")
    rows.append((b, o10, o2, floor7, rise, contrast, s_low, r_low, s_lin, s_dec))

P("SUMMARY")
P("  %-6s %10s %10s %10s %10s %10s %9s %9s %9s" %
  ("band", "onset10x", "onset2xMR", "I(7e-5)fl", "7e5->1e4", "contr@1e4",
   "MRsl<=1e-4", "MRsl<=4e-4", "MRdecade"))
for b, o10, o2, f7, ri, c, slo, rlo, sl4, sdec in rows:
    P("  %-6s %10s %10s %10.3g %10.3g %10.2f %9.3f %9.3f %9.3f"
      % (SHORT[b].split(")")[0] + ")", o10, o2, f7, ri, c, slo, sl4, sdec))
P("")
P("PHYSICS CHECK -- MR baseline eps_0^2 scaling")
P("In a band holding only the FORCED finite-k MR response, |M_y|^2 must scale")
P("as eps_0^2 in linear response, i.e. log-log slope 2.  The falsifier is")
P("explicit: a slope that stays far from 2 after k = 0 is removed would mean")
P("the removal did not isolate a linear-response band, and the corrected band")
P("would be no more trustworthy than the published one.")
P("")
P("FIT RANGE IS DECISIVE, AND IT IS NOT THE MANUSCRIPT'S.  The manuscript")
P("calls eps_0 <= 4e-4 the linear regime (main.tex Fig. 6 caption).  The MR")
P("baseline itself refutes that: in band (d) mr_fK rises from 126.7 to 5570")
P("floor units between eps_0 = 1e-4 and 3e-4, a factor 44.0 for a factor 3 in")
P("strain, i.e. a LOCAL slope of 3.44.  Fitted over eps_0 <= 4e-4 all")
P("four bands return ~2.48-2.49 -- further from 2 than the published band, not")
P("closer -- because the fit is contaminated by that nonlinear point.  Over")
P("eps_0 <= 1e-4 the corrected bands return 2.087 / 2.091 / 2.090 while the")
P("published band returns 1.273.  The check therefore PASSES for (b), (c) and")
P("(d) and FAILS for (a), in the one strain range where linear response is")
P("actually expected.  Both fits are reported above; neither is hidden.")
txt = "\n".join(L)
open(os.path.join(rc.OUT, "step1b_curves.txt"), "w").write(txt)
print(txt)

# --------------------------------------------------------------- figure
COL = {"full_2fK": "#D55E00", "full_fK": "#000000", "mr_fK": "#009E73"}
MK = {"full_2fK": "^", "full_fK": "o", "mr_fK": "s"}
LEG = {"full_2fK": r"Full @ $2f_K$ (parametric)", "full_fK": r"Full @ $f_K$",
       "mr_fK": r"MR only @ $f_K$"}
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True)
for ax, b in zip(axes.ravel(), BANDS):
    Ib = I[b]; norm = Ib[0, iM]
    for ci, lab in enumerate(LAB):
        ax.loglog(EPS, Ib[:, ci] / norm, MK[lab] + "-", color=COL[lab],
                  ms=4.5, lw=1.0, label=LEG[lab])
    e_ref = np.array([1e-5, 3e-4])
    ax.loglog(e_ref, (Ib[0, iM] / norm) * (e_ref / EPS[0]) ** 2, ":",
              color=COL["mr_fK"], lw=0.9)
    ax.set_ylabel(r"$I_\mathrm{pair}\,/\,I_\mathrm{pair}(\mathrm{MR},10^{-5})$")
    ax.grid(alpha=0.15, lw=0.4)
for ax in axes[1, :]:
    ax.set_xlabel(r"$\varepsilon_0$")
axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
for ax, t in zip(axes.ravel(), "abcd"):
    ax.text(-0.22, 1.02, "(%s)" % t, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(rc.HERE, "fig6b_corrected.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(rc.OUT, "fig6b_corrected.png"), dpi=200,
            bbox_inches="tight")
print("\nsaved fig6b_corrected.pdf")
