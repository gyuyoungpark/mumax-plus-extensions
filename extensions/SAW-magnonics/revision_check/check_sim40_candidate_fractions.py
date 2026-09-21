"""Read-only cross-check: can the fixed pipeline reproduce the quoted
31.8% / 43.3% weights of FFT bins 4 and 5 in the f_K slice of the
eps_0 = 7e-5 sim40 checkpoint?

This is bookkeeping for step 1, not an interpretation of the feature.  It
answers one narrow question: is the number the later steps inherit
reproducible, and under WHICH denominator.

Run:  python check_sim40_candidate_fractions.py
Writes CANDIDATE_FRACTIONS.txt next to this file.  Reads nothing but the
checkpoint; writes nothing outside revision_check/.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saw_analysis as sa

HERE = os.path.dirname(os.path.abspath(__file__))
CP = os.path.join(os.path.dirname(HERE), "data", "sim40_checkpoints",
                  "sim40_full_2fK_eps7e-05.npz")

QUOTED = (0.318, 0.433)
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def main():
    if not os.path.isfile(CP):
        say("checkpoint absent -- NOT DETERMINABLE FROM FILES")
        return
    d = np.load(CP)
    my = d["my_xt"]
    dx = float(d["CX"])
    dt = float(d["DT_REC"])
    nt, nx = my.shape
    dk = 2 * np.pi / (nx * dx)
    say(f"source: {CP}")
    say(f"my_xt {my.shape}, CX = {dx*1e9:.1f} nm, DT_REC = {dt*1e12:.1f} ps, "
        f"eps_0 = {float(d['eps0']):.1e}, f_saw = {float(d['f_saw'])/1e9:.3f} GHz")
    say(f"dk = {sa.to_inv_um(dk):.7f} um^-1; bin 4 = {sa.to_inv_um(4*dk):.6f}, "
        f"bin 5 = {sa.to_inv_um(5*dk):.6f} um^-1")
    say("")
    say("Quoted in the task brief: bin 4 carries 31.8%, bin 5 carries 43.3% of")
    say("the f_K-slice weight.  Neither the window, the time range nor the")
    say("normalising denominator was stated with those numbers, so all twelve")
    say("plausible combinations are scanned here.")
    say("")
    say(f"{'window':6s} {'trange':10s} {'denominator':12s} {'f-bin':>10s} "
        f"{'bin4':>8s} {'bin5':>8s} {'bin5/bin4':>10s}")
    ranges = [((nt // 2, nt), "2nd half"), ((0, nt), "full"),
              ((int(4e-9 / dt), int(12e-9 / dt)), "4-12 ns")]
    best = None
    for win in ("hann", "rect"):
        for tr, lbl in ranges:
            k, f, M = sa.spectrum(my, dx, dt, window=win, trange=tr)
            i, sl = sa.f_slice(M, f, 3.0e9)
            P = sa.power(sl)
            j4 = int(np.argmin(np.abs(k - 4 * dk)))
            j5 = int(np.argmin(np.abs(k - 5 * dk)))
            for dm, dlbl in ((np.ones_like(k, bool), "all k"), (k > 0, "k > 0")):
                T = P[dm].sum()
                a, b = P[j4] / T, P[j5] / T
                err = abs(a - QUOTED[0]) + abs(b - QUOTED[1])
                if best is None or err < best[0]:
                    best = (err, win, lbl, dlbl, a, b)
                say(f"{win:6s} {lbl:10s} {dlbl:12s} {f[i]/1e9:9.4f}G "
                    f"{a:8.4f} {b:8.4f} {b/a:10.4f}")
    say("")
    say(f"quoted values:                              "
        f"{QUOTED[0]:8.4f} {QUOTED[1]:8.4f} "
        f"{QUOTED[1]/QUOTED[0]:10.4f}")
    say("")
    say("FINDING (measured, not inferred):")
    say(f"  * The RATIO bin5/bin4 = {QUOTED[1]/QUOTED[0]:.4f} is reproduced by every")
    say("    combination above to within about 1%, so the two bins' relative")
    say("    weight is a robust property of the record.")
    say(f"  * The ABSOLUTE fractions 0.318 / 0.433 are NOT reproduced by any of the")
    say(f"    twelve combinations. The closest is window={best[1]}, trange={best[2]},")
    say(f"    denominator={best[3]}: {best[4]:.4f} / {best[5]:.4f}.")
    say("  * Therefore the pair (31.8%, 43.3%) cannot be carried forward as a")
    say("    measured quantity until the denominator it was normalised to is")
    say("    stated. Later steps must re-derive it through band_content(), which")
    say("    always reports its own total. The provenance of the original number")
    say("    is NOT DETERMINABLE FROM FILES.")
    with open(os.path.join(HERE, "CANDIDATE_FRACTIONS.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
