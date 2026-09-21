"""Re-fit the sim40 bin-4/5/6 growth rates with an EXPLICIT fit interval.

Same quantity, same data, same transform as the published number; only the
interval selection changes.  This is not a new analysis: it is the corrected
version of the fit reported in `model_comparison.md` ("Gamma(bin 4,5,6) = 2.802,
2.927, 2.268 e8 s^-1"), which came from `model_comparison.auxiliary()` with a
hard-coded window t = 7.5-15 ns.

WHAT THE OLD WINDOW DID NOT CHECK (independent audit 2026-09-18, sec.8, P1-2)
  The window was the second half of the record at every strain, with no
  saturation cut and no signal requirement.  Measured here from the arrays
  themselves: max_x|m_y| reaches 0.0144 at eps_0 = 5e-5, 0.1066 at 7e-5 and
  0.9987 at 1e-4 -- the last one is full reversal, and the whole 7.5-15 ns
  window at that strain lies past the 10 % linearity level, which is crossed at
  t = 4.86 ns.

THE MEASURAND (round 2, 2026-09-18).  The quantity reported below is the one
stated once in `growth_interval.MEASURAND` and pre-registered in
PREREGISTRATION.md section 9: the growth rate of the EARLY LINEAR instability,
on a stated interval, above a stated signal criterion, before saturation and
before any turn-over.  Round 1 of this script fitted one side of the peak chosen
from the overall slope, which returns the DECAY branch of a record that grew;
the `branch` and `late` columns below exist so that a record with two branches is
reported as having two, rather than summarised by whichever one is longer.

THE CORRECTED INTERVAL (growth_interval.fit_growth_interval, criteria stated)
  power       a >= 4 x max(measured out-of-band amplitude, 1e-7) where the
              measured floor is the median of |X_j| over the out-of-band bins
              j = 12..21 (the same FLOOR_BINS the model comparison uses)
  saturation  a <= 0.5 x 0.1, AND only samples where max_x|m_y|(t) < 0.1,
              before the FIRST linearity breach or accepted turn-over;
              returning below the ceiling does not reopen the early interval
  adequacy    a contiguous run of >= 5 samples spanning >= 1 e-folding,
              else the verdict is HELD and no rate is reported

Usage
    python growth_refit_sim40.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import growth_interval as GI                                      # noqa: E402
import saw_analysis as SA                                         # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "sim40_checkpoints")
TAGS = ("5e-05", "7e-05", "1e-04")
BINS = (4, 5, 6)
FLOOR_BINS = np.arange(12, 22)
M_LINEAR = 0.1
OLD_WINDOW = (7.5e-9, 15e-9)
PUBLISHED = {4: 2.802e8, 5: 2.927e8, 6: 2.268e8}       # model_comparison.md, 7e-5


def load(tag):
    d = np.load(os.path.join(DATA, "sim40_full_2fK_eps%s.npz" % tag),
                allow_pickle=True)
    m = d["my_xt"].astype(float)
    dt = float(d["DT_REC"])
    return m, dt, float(d["eps0"])


def main():
    print("=" * 78)
    print("sim40 full_2fK: growth rates with an explicit fit interval")
    print("=" * 78)
    for tag in TAGS:
        m, dt, eps = load(tag)
        nt = m.shape[0]
        t = np.arange(nt) * dt
        X = np.fft.fft(m - m.mean(axis=1, keepdims=True), axis=1) / m.shape[1]
        mmax = np.abs(m).max(axis=1)
        lin = mmax < M_LINEAR
        floor = float(np.median(np.abs(X[:, FLOOR_BINS])))
        icross = np.where(~lin)[0]
        print("")
        print("eps_0 = %.0e   record %.2f ns   max|m_y| end = %.4f   "
              "10%% level crossed at %s"
              % (eps, t[-1] * 1e9, mmax[-1],
                 ("t = %.2f ns" % (t[icross[0]] * 1e9)) if icross.size
                 else "never"))
        print("  measured out-of-band floor (bins 12-21, median) = %.3e; "
              "criterion a >= %.3e" % (floor, 4 * max(floor,
                                                      GI.FP_FLOOR_SINGLE)))
        print("  %-4s %13s %13s %11s %8s %7s %7s  %-6s %s"
              % ("bin", "OLD 7.5-15ns", "CORRECTED", "interval", "e-folds",
                 "n", "r2", "branch", "status"))
        for j in BINS:
            a = np.abs(X[:, j])
            g_old, _, r2_old = SA.growth_fit(t, a, t0=OLD_WINDOW[0],
                                             t1=OLD_WINDOW[1])
            r = GI.fit_growth_interval(t, a, floor=floor, sat_level=M_LINEAR,
                                       linear_mask=lin, min_efold=1.0,
                                       min_points=5, label="bin %d" % j)
            interval = ("%.2f-%.2f ns" % (r["t0"] * 1e9, r["t1"] * 1e9)
                        if r["n"] else "-")
            print("  %-4d %13.4e %13s %11s %8.3f %7d %7.4f  %-6s %s"
                  % (j, g_old,
                     ("%.4e" % r["gamma"]) if r["status"] == GI.OK
                     else "NO RATE",
                     interval, r["efolds"], r["n"], r["r2"], r["branch"],
                     r["status"]))
            if r["late"] is not None:
                late = r["late"]
                print("       LATE interval %.2f-%.2f ns: %+.4e 1/s "
                      "(r2 %.4f, %d samples); %s"
                      % (late["t0"] * 1e9, late["t1"] * 1e9, late["gamma"],
                         late["r2"], late["n"], late["note"]))
            if r["status"] != GI.OK:
                print("       reason: %s" % r["reason"])
            elif tag == "7e-05":
                print("       published %.4e -> corrected %.4e (%+.2f %%)"
                      % (PUBLISHED[j], r["gamma"],
                         100 * (r["gamma"] / PUBLISHED[j] - 1)))
    print("")
    print("The published triplet 2.802 / 2.927 / 2.268 e8 s^-1 belongs to the "
          "7e-5 row only.")


if __name__ == "__main__":
    main()
