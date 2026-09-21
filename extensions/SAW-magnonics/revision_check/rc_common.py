"""rc_common -- thin, shared helpers for the revision_check deliverables.

This file decides NOTHING about conventions.  Every transform, every band
partition, every correlator and every null comes from `saw_analysis.py`
(see CONVENTIONS.md).  What lives here is only:

  * where the files are,
  * how a run is loaded,
  * the ONE stated analysis window used by every deliverable,
  * the "two straddling bins" halfwidth used to turn an off-grid target
    wavevector into an exact 2-bin mask for `saw_analysis.band_content`.

Nothing here writes outside revision_check/.
"""
from __future__ import annotations
import os
import numpy as np
import saw_analysis as sa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # extension/SAW-magnonics
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(DATA, "sim40_checkpoints")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

V_SAW = 3500.0                                   # m/s, sim40/sim25 metadata

# ---------------------------------------------------------------- the window
# ONE convention for every table in this directory:
#   time range   : the SECOND HALF of the record,  trange = (n_t//2, n_t)
#   time window  : Hann          (matches np.hanning in the legacy pipeline)
#   space window : rectangular   (the box is periodic)
#   detrend      : block mean removed
#   quantity     : POWER, |M|^2
# Deviations from this are always named explicitly at the call site.
WINDOW_T = "hann"
WINDOW_X = "rect"


def second_half(n_t):
    return (n_t // 2, n_t)


def fk_slice(m_xt, dx, dt, f_target, trange=None, window=WINDOW_T,
             window_x=WINDOW_X):
    """The complex f-slice nearest f_target, in the module's fixed convention.

    Returns dict with k (rad/m), M (complex, n_k), f_used, f_mismatch_bins,
    dk, df, n_t_used.
    """
    m = np.asarray(m_xt, dtype=np.float64)
    if trange is None:
        trange = second_half(m.shape[0])
    k, f, M = sa.spectrum(m, dx, dt, window=window, window_x=window_x,
                          trange=trange, detrend=True, shift=True)
    i, Ms = sa.f_slice(M, f, f_target)
    df = float(f[1] - f[0])
    return dict(k=k, M=Ms, f_used=float(f[i]), i_f=i, df=df,
                f_mismatch_bins=float((f[i] - f_target) / df),
                dk=float(k[1] - k[0]), n_t_used=int(trange[1] - trange[0]))


def straddle(k_axis, target, eps_rel=1e-9):
    """Halfwidth that selects EXACTLY the two bins straddling `target`.

    Returns (halfwidth, (j_lo, j_hi), frac_of_dk) where frac_of_dk =
    target/dk.  Raises if `target` sits within eps of a bin centre, because
    then "the two straddling bins" is not defined and the caller must say
    what it wants instead.
    """
    k = np.asarray(k_axis, dtype=np.float64)
    dk = float(k[1] - k[0])
    below = np.nonzero(k <= target)[0]
    above = np.nonzero(k >= target)[0]
    if below.size == 0 or above.size == 0:
        raise ValueError("target %g outside the k axis" % target)
    j_lo, j_hi = int(below[-1]), int(above[0])
    if j_lo == j_hi:
        raise ValueError("target %g is ON a bin centre; 'straddling bins' "
                         "is undefined" % target)
    far = max(abs(target - k[j_lo]), abs(k[j_hi] - target))
    return far * (1 + eps_rel), (j_lo, j_hi), target / dk


def targets(k_axis, K_pump, k_ref=None):
    """Which k_1 bins to evaluate, and why.  Duplicates merged, order kept.

    `k_ref` is the GEOMETRY reference (the SAW wavevector of the MEL run); it
    is used so the uniform-Suhl control, whose own K_pump is 0, is still
    reported on the same bins as the MEL channel.

    Moved here from step2_coherence.py on 2026-09-18 so that the row-selection
    regression test can exercise the real list rather than a copy of it.
    Body unchanged.
    """
    k = np.asarray(k_axis, dtype=np.float64)
    if k_ref is None:
        k_ref = K_pump
    want = []
    if k_ref != 0:
        _, (jl, jh), _ = straddle(k, k_ref / 2)
        want.append((jl, "lower bin straddling +k_SAW/2"))
        want.append((jh, "upper bin straddling +k_SAW/2"))
        want.append((int(np.argmin(np.abs(k - k_ref / 2))),
                     "nearest bin to +k_SAW/2 = THE PUBLISHED POINT"))
        want.append((int(np.argmin(np.abs(k - k_ref))),
                     "nearest bin to +k_SAW"))
    want.append((int(np.argmin(np.abs(k))), "k = 0"))
    out = []
    for j, lbl in want:
        hit = [i for i, (jj, _) in enumerate(out) if jj == j]
        if hit:
            out[hit[0]] = (j, out[hit[0]][1] + " / " + lbl)
        else:
            out.append((j, lbl))
    return out


def select_target_row(tg, k_axis, j2, k_target, K_pump, atol_bins=0.5):
    """Row index of `tg` that evaluates `k_target`, found by WAVENUMBER.

    `tg` is a `targets()` list of (bin index, label); `j2` is the pair-partner
    index array of `saw_analysis._partner_index(k_axis, K_pump)`.  A row
    qualifies when

        * its bin is the bin of `k_axis` nearest `k_target`, and that bin is
          within `atol_bins` bins of `k_target`, and
        * its pair partner really is the partner of that bin under this pump,
          |(K_pump - k_1) - k_2| <= atol_bins bins.

    Raises LookupError if no row qualifies and LookupError if more than one
    does.  There is deliberately NO fallback: the previous code used the fixed
    position `min(2, n-1)`, and because `targets()` merges duplicate bins that
    position is the near-k_SAW row whenever the half-k_SAW request has merged
    into row 0 or 1.  The percentile of `1/sqrt(M)` was then quoted from the
    near-k_SAW null instead of the half-k_SAW null (independent audit
    2026-09-18, section 5; pinned by
    `test_saw_analysis.py::test_null_row_is_requested_target`).
    """
    k = np.asarray(k_axis, dtype=np.float64)
    dk = abs(float(k[1] - k[0]))
    tol = atol_bins * dk * (1 + 1e-9)
    j_want = int(np.argmin(np.abs(k - k_target)))
    if abs(k[j_want] - k_target) > tol:
        raise LookupError(
            "no bin of the axis is within %g bins of k_target = %.6e rad/m "
            "(nearest bin %.6e)" % (atol_bins, k_target, k[j_want]))
    hits = []
    for n, (j, lbl) in enumerate(tg):
        if j != j_want:
            continue
        if abs((K_pump - k[j]) - k[int(j2[j])]) > tol:
            raise LookupError(
                "row %d (k_1 = %.6e) is not a pair partner row under "
                "K_pump = %.6e: partner snap error %.3e rad/m exceeds %g bins"
                % (n, k[j], K_pump, (K_pump - k[j]) - k[int(j2[j])], atol_bins))
        hits.append(n)
    if len(hits) != 1:
        raise LookupError(
            "k_target = %.6e rad/m (bin %d, k = %.6e) matches %d rows of the "
            "target list %r -- refusing to guess"
            % (k_target, j_want, k[j_want], len(hits),
               [(int(j), lbl) for j, lbl in tg]))
    return hits[0]


def band_six(k_axis, M_slice, k_pair, k_saw, quantity="power"):
    """`saw_analysis.band_content` with the two-straddling-bins halfwidths.

    k0 gets the single k = 0 bin; each of plus/minus/saw_plus/saw_minus gets
    exactly the two bins straddling its target.  Disjointness, the collision
    report and the sum-to-1 check are the module's.
    """
    hw_pair, idx_pair, frac_pair = straddle(k_axis, k_pair)
    hw_saw, idx_saw, frac_saw = straddle(k_axis, k_saw)
    r = sa.band_content(k_axis, M_slice, k_pair, k_saw, hw_pair,
                        quantity=quantity, saw_halfwidth=hw_saw)
    r["bins_pair"] = idx_pair
    r["bins_saw"] = idx_saw
    r["k_pair_over_dk"] = frac_pair
    r["k_saw_over_dk"] = frac_saw
    return r


def load_sim40(label, eps):
    f = os.path.join(CKPT, "sim40_%s_eps%.0e.npz" % (label, eps))
    d = np.load(f)
    return dict(m=d["my_xt"], dx=float(d["CX"]), dt=float(d["DT_REC"]),
                f_saw=float(d["f_saw"]), f_K=float(d["f_K"]),
                eps0=float(d["eps0"]), path=f)


def k_saw_of(f_saw, v=V_SAW):
    return 2 * np.pi * f_saw / v


def loglog_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    lx, ly = np.log10(x[m]), np.log10(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ c
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1 - ((ly - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return float(c[0]), float(r2), int(m.sum())
