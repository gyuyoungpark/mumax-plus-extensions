"""STEP 1 -- corrected I_pair(eps_0) over all 24 sim40 checkpoints.

Four band definitions (a)-(d), the onset, the contrast at eps_0 = 1e-4,
the MR-baseline eps^2 slope physics check, and fig6b_corrected.pdf.

Conventions: saw_analysis.py / CONVENTIONS.md, via rc_common.py.
Reads data/ only; writes revision_check/ only.

    python step1_ipair.py
"""
from __future__ import annotations
import os, json
import numpy as np
import saw_analysis as sa
import rc_common as rc

EPS = np.array([1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3, 3e-3])
LABELS = ["full_2fK", "full_fK", "mr_fK"]
F_K = 3.0e9
# analyze_sim40.py uses ONE k_pair for all three channels:
#   k_pair = k_SAW(f_K) = 2 pi f_K / v_SAW = k_SAW(2 f_K)/2.
K_PAIR = 2 * np.pi * F_K / rc.V_SAW          # 5.385587 um^-1
K_SAW_2FK = 2 * K_PAIR                       # 10.771175 um^-1

BANDS = ["a_published", "b_union_nok0", "c_oneside_2dk", "d_bins45"]
BAND_TITLE = {
    "a_published":   "(a) as published: sum of the two OVERLAPPING +/-k_pair windows, halfwidth 16 dk",
    "b_union_nok0":  "(b) union of the same two windows, counted once, k = 0 bin removed",
    "c_oneside_2dk": "(c) one-sided +k_pair window, halfwidth 2 dk, k = 0 removed (not in it anyway)",
    "d_bins45":      "(d) the two bins straddling +k_pair only (bins +4 and +5)",
}


def four_bands(k, P, dk):
    """The four band sums on one power slice.  Returns values, bin counts,
    the number of bins the published recipe double counts, and the total."""
    hw16 = 16 * dk
    bp = (k > K_PAIR - hw16) & (k < K_PAIR + hw16)
    bm = (k > -K_PAIR - hw16) & (k < -K_PAIR + hw16)
    j0 = int(np.argmin(np.abs(k)))
    uni = (bp | bm).copy(); uni[j0] = False
    hw2 = 2 * dk
    mc = np.abs(k - K_PAIR) <= hw2 * (1 + 1e-9); mc[j0] = False
    hwd, (j4, j5), _ = rc.straddle(k, K_PAIR)
    md = np.zeros_like(bp); md[j4] = md[j5] = True
    vals = {"a_published": float(P[bp].sum() + P[bm].sum()),
            "b_union_nok0": float(P[uni].sum()),
            "c_oneside_2dk": float(P[mc].sum()),
            "d_bins45": float(P[md].sum())}
    nb = {"a_published": int(bp.sum() + bm.sum()), "b_union_nok0": int(uni.sum()),
          "c_oneside_2dk": int(mc.sum()), "d_bins45": 2}
    extra = {"overlap_bins": int((bp & bm).sum()),
             "k0_in_a": bool(bp[j0] and bm[j0]),
             "bins45": (j4 - j0, j5 - j0),
             "k0_frac_of_a": float(2 * P[j0] / max(P[bp].sum() + P[bm].sum(), 1e-300)),
             "c_bins_k": [float(x) for x in sa.to_inv_um(k[mc])]}
    return vals, nb, extra, float(P.sum())


def main():
    I = {b: np.full((len(EPS), len(LABELS)), np.nan) for b in BANDS}
    FRAC = {b: np.full((len(EPS), len(LABELS)), np.nan) for b in BANDS}
    TOT = np.full((len(EPS), len(LABELS)), np.nan)
    K0 = np.full((len(EPS), len(LABELS)), np.nan)      # k=0 fraction of the f_K slice
    meta = {}
    for ci, lab in enumerate(LABELS):
        for ei, e in enumerate(EPS):
            r = rc.load_sim40(lab, e)
            sl = rc.fk_slice(r["m"], r["dx"], r["dt"], F_K)
            P = sa.power(sl["M"])
            vals, nb, extra, tot = four_bands(sl["k"], P, sl["dk"])
            j0 = int(np.argmin(np.abs(sl["k"])))
            for b in BANDS:
                I[b][ei, ci] = vals[b]; FRAC[b][ei, ci] = vals[b] / tot
            TOT[ei, ci] = tot
            K0[ei, ci] = P[j0] / tot
            if not meta:
                meta = dict(dk=sl["dk"], df=sl["df"], n_t=sl["n_t_used"],
                            f_used=sl["f_used"], f_mismatch=sl["f_mismatch_bins"],
                            n_bins=nb, **extra)
    np.savez(os.path.join(rc.OUT, "step1_ipair.npz"), eps=EPS,
             labels=np.array(LABELS), total_slice=TOT, k0_frac=K0,
             dk=meta["dk"], k_pair=K_PAIR,
             **{"I_" + b: I[b] for b in BANDS},
             **{"frac_" + b: FRAC[b] for b in BANDS})

    # cross-check band (a) against the SHIPPED observables file
    d = np.load(os.path.join(rc.DATA, "sim40_observables.npz"))
    ship = d["I_pair"]; slab = [str(x) for x in d["labels"]]
    ship_re = np.array([[ship[ei, slab.index(l)] for l in LABELS]
                        for ei in range(len(EPS))])
    ratio = ship_re / I["a_published"]

    L = []; P_ = L.append
    P_("STEP 1 -- corrected I_pair over the 24 sim40 checkpoints")
    P_("=" * 78)
    P_("command: python step1_ipair.py   (module saw_analysis.py, CONVENTIONS.md)")
    P_("window : second half of each record, Hann in t, rect in x, detrended, POWER")
    P_("grid   : n_t(used)=%d, n_x=1024, dk=%.7f um^-1, df=%.5f MHz"
       % (meta["n_t"], meta["dk"] / 1e6, meta["df"] / 1e6))
    P_("f-slice: f_K = 3.000000 GHz requested; nearest bin f = %.6f GHz "
       "(%+.3f bins, %+.1f MHz)" % (meta["f_used"] / 1e9, meta["f_mismatch"],
                                    meta["f_mismatch"] * meta["df"] / 1e6))
    P_("k_pair = k_SAW(f_K) = %.6f um^-1 = %.4f dk  -> OFF GRID, straddled by "
       "bins %+d and %+d" % (K_PAIR / 1e6, K_PAIR / meta["dk"], *meta["bins45"]))
    P_("")
    P_("BAND BOOKKEEPING")
    for b in BANDS:
        P_("  %-14s %4d bins   %s" % (b, meta["n_bins"][b], BAND_TITLE[b]))
    P_("  band (a) double counts %d bins (every bin with |k| < 14.25 um^-1), "
       "and the k = 0 bin is in BOTH windows: %s" %
       (meta["overlap_bins"], meta["k0_in_a"]))
    P_("  band (c) bins (um^-1): %s" % np.round(meta["c_bins_k"], 6).tolist())
    P_("")
    P_("REPRODUCTION OF THE PUBLISHED RECIPE")
    P_("  shipped sim40_observables.npz I_pair / band (a) recomputed here:")
    P_("    min %.6e  max %.6e  spread %.3e" %
       (ratio.min(), ratio.max(), ratio.max() / ratio.min() - 1))
    P_("  A CONSTANT ratio means band (a) reproduces the published recipe up to")
    P_("  one overall FFT normalisation; a spread would mean it does not.")
    P_("")
    for b in BANDS:
        P_("--- %s ---" % BAND_TITLE[b])
        P_("  I_pair (absolute, arbitrary FFT normalisation)")
        P_("    eps_0   " + "".join("%16s" % l for l in LABELS))
        for ei, e in enumerate(EPS):
            P_("    %.0e " % e + "".join("%16.6e" % I[b][ei, ci] for ci in range(3)))
        P_("  fraction of the FULL f_K-slice power")
        P_("    eps_0   " + "".join("%16s" % l for l in LABELS))
        for ei, e in enumerate(EPS):
            P_("    %.0e " % e + "".join("%16.6e" % FRAC[b][ei, ci] for ci in range(3)))
        P_("")
    P_("k = 0 BIN as a fraction of the f_K-slice power (why band (a) is wrong)")
    P_("    eps_0   " + "".join("%16s" % l for l in LABELS))
    for ei, e in enumerate(EPS):
        P_("    %.0e " % e + "".join("%16.6e" % K0[ei, ci] for ci in range(3)))
    P_("")
    P_("k = 0 CONTRIBUTION to band (a) itself, 2*P(k=0)/I_a  (it enters twice)")
    P_("    eps_0   " + "".join("%16s" % l for l in LABELS))
    for ei, e in enumerate(EPS):
        row = []
        for ci in range(3):
            row.append(2 * K0[ei, ci] * TOT[ei, ci] / I["a_published"][ei, ci])
        P_("    %.0e " % e + "".join("%16.6e" % v for v in row))
    txt = "\n".join(L)
    open(os.path.join(rc.OUT, "step1_ipair_raw.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
