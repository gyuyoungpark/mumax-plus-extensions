"""Control suite for saw_analysis.py.

Every test has an answer known before the code runs, and every diagnostic is
stated together with the result that WOULD HAVE REFUTED it.  Where a
diagnostic turns out not to separate two hypotheses, that is recorded as the
finding, not hidden.

Run:  python test_saw_analysis.py          (prints the report, writes
      TEST_RESULTS.txt and DISCRIMINATION_TABLE.md next to this file)
      python -m pytest test_saw_analysis.py -q     also works.

Controls
--------
 (a) two exactly phase-locked waves at k1 and K_pump-k1        -> C = 1
 (a') the same with the WRONG pump-phase sign                  -> C = 0.2357 (analytic)
 (b) the same with independent per-segment phases              -> inside the null
 (c) a spatially uniform k=0 oscillation                       -> finite-k content 0,
                                                                  legacy I_pair = 2x total
 (d) ONE off-grid sinusoid at k = 5.5 um^-1                    -> leakage signature
 (e) TWO on-grid components at bins 4 and 5, same f, same
     growth, fixed relative phase, a4=+sqrt(0.318),
     a5=-sqrt(0.433)                                           -> ONE peak at 5.599 um^-1
                                                                  under 16x zero padding
 (f) two components, different f AND different growth          -> the easy case

Added 2026-09-18 after the independent audit of that date:
 (a'') the (a) pair with k4 read at f1 and k5 read at ITS OWN f2, i.e. two
      genuinely different frequencies extracted independently   -> C = 1, and
      the wrong sign still gives the analytic residual.  Controls (a)/(a')/(b)
      read both members from the single f1 slice, where k5 carries only
      3.0363e-05 of the power it holds at f2, so they test the pump-phase SIGN
      rather than pair extraction (audit section 9).
 (R1) a forward and a backward wave at the SAME |k| with different frequencies
      and amplitudes -> the projected-frequency estimator of
      `model_comparison.project_and_estimate_frequency` must return the FORWARD
      one (audit section 4).
 (R2) a half-k_SAW request against a target list whose duplicates have been
      merged -> the quoted null percentile must come from the half-k_SAW row,
      and an unmatched request must raise (audit section 5).

Item count: this suite reports 49 PASS items (32 before 2026-09-18 -- the
"33 controls" in NUMBERS_FOR_MANUSCRIPT.md was a miscount of that suite, which
emitted 32 checks both statically and at run time -- plus 4 from R1, 8 from R2
and 5 from (a'')).
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saw_analysis as sa

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# ---------------------------------------------------------------- sim40 grid
NX = 1024
DX = 5e-9
DT = 20e-12
LX = NX * DX                      # 5.12 um
DK = 2 * np.pi / LX               # 1.2271846303e6 rad/m  = 1.2271846 "um^-1"
X = (np.arange(NX) + 0.5) * DX    # cell centres, as in chiralsawfield.cu

V_SAW = 3500.0
F_K = 3.0e9
F_SAW = 6.0e9
Q_PUMP = 2 * np.pi * F_SAW / V_SAW          # 10.771175 um^-1

K4 = 4 * DK                                  # 4.908739 um^-1
K5 = 5 * DK                                  # 6.135923 um^-1

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def _check(name, ok, detail=""):
    say(("  PASS  " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    assert ok, name + " " + detail


# ============================================================================
# 0. the convention itself
# ============================================================================

def test_convention_sign():
    say("\n[0a] convention: a +x-travelling wave must land at (+k, +f)")
    k0, f0 = K4, 23 / (376 * DT)
    t = np.arange(376) * DT
    m = np.cos(k0 * X[None, :] - 2 * np.pi * f0 * t[:, None])
    k, f, M = sa.spectrum(m, DX, DT, window="rect")
    P = sa.power(M)
    i, j = np.unravel_index(np.argmax(P), P.shape)
    say(f"       peak at k = {sa.to_inv_um(k[j]):+.6f} um^-1, f = {f[i]/1e9:+.6f} GHz")
    say(f"       expected  k = {sa.to_inv_um(k0):+.6f} um^-1, f = {f0/1e9:+.6f} GHz")
    _check("peak sits at (+k0,+f0)", abs(k[j] - k0) < 1e-6 * DK and abs(f[i] - f0) < 1e-6 * f0)
    say("       REFUTED IF: the peak had appeared at (-k0,+f0) -- that is the")
    say("       fft2 convention and is exactly what the legacy scripts patch with k -> -k.")


def test_fft2_equivalence():
    say("\n[0b] |M_here(k,+f)| == |fft2(-k,+f)|  (proves make_fig5_v3's k_disp=-k_fft is CORRECT)")
    rng = np.random.default_rng(1)
    m = rng.standard_normal((128, 256))
    k, f, M = sa.spectrum(m, DX, DT, window="rect", detrend=False)
    F2 = np.fft.fftshift(np.fft.fft2(m), axes=(0, 1))
    # fft2 axes: f from fftfreq(nt), k from 2pi*fftfreq(nx) -- same grids
    # ::-1 on a shifted axis maps k -> -k up to the Nyquist bin offset
    Mm = np.abs(M)[:, 1:]
    F2m = np.abs(F2)[:, 1:][:, ::-1]
    rel = np.max(np.abs(Mm - F2m)) / np.max(F2m)
    say(f"       max relative difference over all (k,f), Nyquist bin excluded: {rel:.3e}")
    _check("fft2 identity holds", rel < 1e-12)
    say("       REFUTED IF: rel difference were O(1); then the sign flip in")
    say("       make_fig5_v3.py / analyze_sim40.py would be wrong. It is not.")


def test_amplitude_power_not_mixed():
    say("\n[0c] amplitude vs power flag")
    rng = np.random.default_rng(2)
    m = rng.standard_normal((64, 128))
    k, f, M = sa.spectrum(m, DX, DT)
    bp = sa.band_content(k, M[10], 3 * DK, 6 * DK, 1.5 * DK, quantity="power")
    ba = sa.band_content(k, M[10], 3 * DK, 6 * DK, 1.5 * DK, quantity="amplitude")
    say(f"       power total   = {bp['total']:.6e}")
    say(f"       amplitude tot = {ba['total']:.6e}")
    _check("power total equals sum |M|^2", abs(bp["total"] - np.sum(np.abs(M[10]) ** 2)) < 1e-9 * bp["total"])
    _check("amplitude total equals sum |M|", abs(ba["total"] - np.sum(np.abs(M[10]))) < 1e-9 * ba["total"])
    _check("the two totals differ", abs(bp["total"] - ba["total"]) > 0)
    try:
        sa.band_content(k, M[10], 3 * DK, 6 * DK, 1.5 * DK, quantity="magnitude_squared_maybe")
        ok = False
    except ValueError:
        ok = True
    _check("an unknown quantity is rejected, not silently defaulted", ok)


# ============================================================================
# (a)/(a')/(b) the pair correlator
# ============================================================================

SEG_LEN = 64
N_SEG = 6
STRIDE = 67          # deliberately != SEG_LEN, see below
NT_CORR = (N_SEG - 1) * STRIDE + SEG_LEN        # 399
DF_SEG = 1.0 / (SEG_LEN * DT)                   # 781.25 MHz
F1 = 3 * DF_SEG
F2 = 5 * DF_SEG
FP = F1 + F2                                    # 8*DF_SEG = 6.25 GHz
KP = K4 + K5                                    # 9*DK


def _phase_locked_field(phi1_of_seg, phi2_of_seg, nt=NT_CORR):
    """m(x,t) = cos(k4 x - w1 t + phi1_s) + cos(k5 x - w2 t + phi2_s),
    with the phases held constant inside each segment."""
    t = np.arange(nt) * DT
    seg_of = np.zeros(nt, dtype=int)
    for s in range(N_SEG):
        seg_of[s * STRIDE:] = s
    p1 = np.array([phi1_of_seg[s] for s in seg_of])
    p2 = np.array([phi2_of_seg[s] for s in seg_of])
    m = (np.cos(K4 * X[None, :] - 2 * np.pi * F1 * t[:, None] + p1[:, None])
         + np.cos(K5 * X[None, :] - 2 * np.pi * F2 * t[:, None] + p2[:, None]))
    return m


def _corr_from_field(m, phase_sign=+1):
    k, f, Ms, ts = sa.segment_spectra(m, DX, DT, SEG_LEN, N_SEG, stride=STRIDE,
                                      window="hann")
    i1 = int(np.argmin(np.abs(f - F1)))
    sl = Ms[:, i1, :]
    res = sa.pair_correlator(sl, ts, k, KP, 2 * np.pi * FP, phase_sign=phase_sign)
    j4 = int(np.argmin(np.abs(k - K4)))
    return k, f, sl, ts, res, j4


def test_a_phase_locked_correlator():
    say("\n[a] two exactly phase-locked waves at k4 and k5 = K_pump-k4")
    say(f"    f1 = {F1/1e9:.6f} GHz, f2 = {F2/1e9:.6f} GHz, f_pump = {FP/1e9:.6f} GHz")
    say(f"    segments: {N_SEG} x {SEG_LEN} samples, stride {STRIDE} (NOT {SEG_LEN}:")
    say("    with stride == seg_len, f_pump*stride*dt is an integer here and")
    say("    exp(i w_p t_s) == 1 for every segment, i.e. the sign of the pump-phase")
    say("    removal would be UNTESTABLE. The stride is chosen to break that.")
    m = _phase_locked_field([0.3] * N_SEG, [1.1] * N_SEG)
    k, f, sl, ts, res, j4 = _corr_from_field(m, phase_sign=+1)
    C = res["C"][j4]
    say(f"    C(k4) with the CORRECT sign (+1) = {C:.10f}     expected 1")
    _check("C = 1 for a phase-locked pair", abs(C - 1.0) < 1e-9, f"C={C:.12f}")


def test_a_wrong_sign_fails():
    say("\n[a'] the SAME data with the wrong pump-phase sign (the sim39 bug)")
    m = _phase_locked_field([0.3] * N_SEG, [1.1] * N_SEG)
    k, f, sl, ts, res, j4 = _corr_from_field(m, phase_sign=-1)
    C = res["C"][j4]
    # analytic: residual rotation exp(-2 i w_p t_s), t_s = s*STRIDE*dt
    ph = -2 * 2 * np.pi * FP * (np.arange(N_SEG) * STRIDE * DT)
    Cpred = abs(np.exp(1j * ph).mean())
    say(f"    C(k4) with the WRONG sign (-1) = {C:.10f}")
    say(f"    analytic prediction |<exp(-2 i w_p t_s)>| = {Cpred:.10f}")
    _check("wrong sign reproduces the analytic residual", abs(C - Cpred) < 1e-9)
    _check("wrong sign is clearly below 1", C < 0.5)
    say("    REFUTED IF: both signs had given 1. They do not; the sign is a real,")
    say("    falsifiable choice, and control (a) fixes it to +1 in this convention.")


def test_b_random_phases_land_in_null():
    say("\n[b] the same two waves with INDEPENDENT per-segment phases")
    rng = np.random.default_rng(7)
    Cs, inside = [], 0
    trials = 200
    for tr in range(trials):
        p1 = rng.uniform(0, 2 * np.pi, N_SEG)
        p2 = rng.uniform(0, 2 * np.pi, N_SEG)
        m = _phase_locked_field(p1, p2)
        k, f, sl, ts, res, j4 = _corr_from_field(m, +1)
        Cs.append(res["C"][j4])
    Cs = np.array(Cs)
    # null built from ONE of those realisations (amplitudes preserved)
    m = _phase_locked_field(rng.uniform(0, 2 * np.pi, N_SEG),
                            rng.uniform(0, 2 * np.pi, N_SEG))
    k, f, sl, ts, res, j4 = _corr_from_field(m, +1)
    nul = sa.pair_correlator_null(sl, ts, k, KP, 2 * np.pi * FP,
                                  n_real=20000, k_indices=[j4], seed=11)
    q = {kk: float(v[0]) for kk, v in nul["quantiles"].items()}
    lo = float(np.quantile(nul["C_null"][:, 0], 0.005))
    inside = int(((Cs >= lo) & (Cs <= q["q99"])).sum())
    say(f"    measured C over {trials} random-phase realisations:")
    say(f"      mean {Cs.mean():.4f}   median {np.median(Cs):.4f}   max {Cs.max():.4f}")
    say(f"    amplitude-preserving null at the same k:")
    say(f"      mean {float(nul['mean'][0]):.4f}  q50 {q['q50']:.4f}  q90 {q['q90']:.4f}"
        f"  q95 {q['q95']:.4f}  q99 {q['q99']:.4f}")
    say(f"    the single line 1/sqrt(M) = {nul['analytic_1_over_sqrtM']:.4f} sits at "
        f"quantile {float((nul['C_null'][:,0] < nul['analytic_1_over_sqrtM']).mean()):.3f}"
        " of the null -- it is NOT a 95% threshold")
    say(f"    {inside}/{trials} realisations fall inside the null's [0.5%, 99%] interval")
    _check("random-phase data land inside the null", inside >= int(0.95 * trials),
           f"{inside}/{trials}")
    _check("null mean is far below the phase-locked value 1", float(nul["mean"][0]) < 0.6)
    say("    REFUTED IF: the random-phase C had clustered near 1, or the null had")
    say("    been narrower than the scatter of the measured random-phase values.")


# ============================================================================
# (c) uniform k = 0 oscillation, and the published I_pair double-count
# ============================================================================

def legacy_I_pair(k_axis, M_slice, k_pair, dk_window):
    """Verbatim logic of src/analyze_sim40.py::I_pair (the published routine)."""
    s = np.abs(M_slice) ** 2
    band_p = (k_axis > k_pair - dk_window) & (k_axis < k_pair + dk_window)
    band_m = (k_axis > -k_pair - dk_window) & (k_axis < -k_pair + dk_window)
    return float(s[band_p].sum() + s[band_m].sum())


def test_c_uniform_k0():
    say("\n[c] spatially uniform k=0 oscillation")
    nt = 376
    t = np.arange(nt) * DT
    f0 = 23 / (nt * DT)
    m = np.tile(np.cos(2 * np.pi * f0 * t)[:, None], (1, NX))
    k, f, M = sa.spectrum(m, DX, DT, window="rect")
    i, sl = sa.f_slice(M, f, f0)
    k_pair = 2 * np.pi * F_K / V_SAW
    bc = sa.band_content(k, sl, k_pair, Q_PUMP, 1.5 * DK, quantity="power")
    fr = bc["fractions"]
    say(f"    k0 bin fraction        = {fr['k0']:.12f}")
    say(f"    +k_target band         = {fr['plus']:.3e}")
    say(f"    -k_target band         = {fr['minus']:.3e}")
    say(f"    +k_SAW / -k_SAW lobes  = {fr['saw_plus']:.3e} / {fr['saw_minus']:.3e}")
    say(f"    remainder              = {fr['rest']:.3e}")
    say(f"    fractions sum          = {bc['sum_check']:.15f}")
    _check("all power in the k=0 bin", abs(fr["k0"] - 1.0) < 1e-12)
    _check("finite-k band content is zero",
           max(fr["plus"], fr["minus"], fr["saw_plus"], fr["saw_minus"]) < 1e-20)
    _check("fractions sum to 1", abs(bc["sum_check"] - 1.0) < 1e-12)
    # the published bug
    dk_bin = abs(k[1] - k[0])
    leg = legacy_I_pair(k, sl, k_pair, 16 * dk_bin)
    tot = float(np.sum(np.abs(sl) ** 2))
    say(f"    legacy analyze_sim40.I_pair(dk_window = 16 bins = "
        f"{sa.to_inv_um(16*dk_bin):.3f} um^-1) / total = {leg/tot:.12f}")
    say(f"    (its two windows are k in ({sa.to_inv_um(k_pair-16*dk_bin):+.2f},"
        f"{sa.to_inv_um(k_pair+16*dk_bin):+.2f}) and "
        f"({sa.to_inv_um(-k_pair-16*dk_bin):+.2f},{sa.to_inv_um(-k_pair+16*dk_bin):+.2f})"
        f" um^-1 -- they overlap on |k| < {sa.to_inv_um(16*dk_bin-k_pair):.2f} um^-1,"
        " which contains k = 0)")
    _check("legacy I_pair double-counts a pure k=0 signal", abs(leg / tot - 2.0) < 1e-12)
    _check("band_content does not", fr["plus"] + fr["minus"] == 0.0)
    say("    REFUTED IF: band_content had also returned 2x, or had returned any")
    say("    finite-k weight for a field with no finite-k content.")


def test_band_content_disjoint_random():
    say("\n[c'] band_content stays disjoint for overlapping requests")
    rng = np.random.default_rng(3)
    k = sa.grid_k_axis(NX, DX)
    M = rng.standard_normal(NX) + 1j * rng.standard_normal(NX)
    k_pair = 2 * np.pi * F_K / V_SAW
    bc = sa.band_content(k, M, k_pair, Q_PUMP, 16 * DK)   # the legacy halfwidth
    say(f"    halfwidth 16 bins: bins stolen by priority = {bc['stolen']}")
    say(f"    n_bins = {bc['n_bins']}")
    say(f"    sum of fractions = {bc['sum_check']:.15f}")
    _check("sum is exactly 1", abs(bc["sum_check"] - 1.0) < 1e-12)
    _check("collisions are reported, not silent", sum(bc["stolen"].values()) > 0)


# ============================================================================
# (d)/(e)/(f) the hard cases
# ============================================================================

GAMMA = 6.0e8            # 1/s growth rate used by (d) and (e) alike
NT_HARD = 376
K_OFFGRID = 5.5e6        # 5.5 um^-1, 4.4818 bins
F_HARD = 23 / (NT_HARD * DT)


def field_d(growth=True):
    t = np.arange(NT_HARD) * DT
    env = np.exp(GAMMA * t) if growth else np.ones_like(t)
    return env[:, None] * np.cos(K_OFFGRID * X[None, :] - 2 * np.pi * F_HARD * t[:, None])


A4 = np.sqrt(0.318)
A5 = -np.sqrt(0.433)


def field_e(growth=True):
    t = np.arange(NT_HARD) * DT
    env = np.exp(GAMMA * t) if growth else np.ones_like(t)
    ph = -2 * np.pi * F_HARD * t[:, None]
    u = (A4 * np.cos(K4 * X[None, :] + ph) + A5 * np.cos(K5 * X[None, :] + ph))
    return env[:, None] * u


GAMMA_F4, GAMMA_F5 = 4.0e8, 9.0e8
F_F4 = 21 / (NT_HARD * DT)
F_F5 = 27 / (NT_HARD * DT)


def field_f():
    t = np.arange(NT_HARD) * DT
    e4 = np.exp(GAMMA_F4 * t)[:, None]
    e5 = np.exp(GAMMA_F5 * t)[:, None]
    return (e4 * A4 * np.cos(K4 * X[None, :] - 2 * np.pi * F_F4 * t[:, None])
            + e5 * (-A5) * np.cos(K5 * X[None, :] - 2 * np.pi * F_F5 * t[:, None]))


def _spatial_spectrum(u, pad, window_x="rect"):
    w = sa._window(u.size, window_x)
    n = u.size * pad
    U = np.fft.fftshift(np.fft.fft(u * w, n=n))
    kk = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n, d=DX))
    return kk, U


def _count_peaks(P, rel=0.01):
    p = P / P.max()
    idx = [i for i in range(1, p.size - 1)
           if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > rel]
    return idx


def test_d_single_offgrid_leakage():
    say("\n[d] ONE off-grid sinusoid at k = 5.500000 um^-1  (4.48180 bins, dk = "
        f"{sa.to_inv_um(DK):.6f})")
    u = np.cos(K_OFFGRID * X)
    for wx in ("rect", "hann"):
        for pad in (1, 8, 16):
            kk, U = _spatial_spectrum(u, pad, wx)
            P = np.abs(U) ** 2
            P = P * (kk > 0)
            jpk = int(np.argmax(P))
            pk = int(np.argmax(np.abs(kk - K_OFFGRID) < 1e9))
            pks = _count_peaks(P)
            top = np.sort(P[P > 0])[::-1]
            say(f"    window={wx:5s} pad={pad:2d}x : peak at "
                f"{sa.to_inv_um(kk[jpk]):.6f} um^-1, "
                f"{len(pks)} local maxima above 1% of peak, "
                f"2nd/1st = {top[1]/top[0]:.4f}")
    # on-grid bin weights, rectangular, no padding -- the leakage signature
    kk, U = _spatial_spectrum(u, 1, "rect")
    P = np.abs(U) ** 2
    pos = kk > 0
    tot = P[pos].sum()
    j = {n: int(np.argmin(np.abs(kk - n * DK))) for n in range(1, 10)}
    say("    rectangular, no padding, fraction of the positive-k power per bin:")
    for n in range(2, 9):
        say(f"       bin {n} (k = {sa.to_inv_um(n*DK):.6f}): {P[j[n]]/tot:.6f}")
    f4, f5 = P[j[4]] / tot, P[j[5]] / tot
    say(f"    bins 4+5 hold {f4+f5:.6f} of the positive-k power; "
        f"ratio |M5/M4| = {np.sqrt(P[j[5]]/P[j[4]]):.6f}")
    say(f"    relative phase arg(M5/M4) = "
        f"{np.angle(U[j[5]]/U[j[4]], deg=True):+.3f} deg")
    _check("an off-grid sinusoid leaks into many bins", f4 + f5 < 0.98)
    _check("bins 4 and 5 dominate", f4 + f5 > 0.5)
    # Hann suppresses the far tail
    kkh, Uh = _spatial_spectrum(u, 1, "hann")
    Ph = np.abs(Uh) ** 2
    jh = {n: int(np.argmin(np.abs(kkh - n * DK))) for n in range(1, 12)}
    toth = Ph[kkh > 0].sum()
    say(f"    Hann in x: bin 8 holds {Ph[jh[8]]/toth:.3e} vs rectangular "
        f"{P[j[8]]/tot:.3e}  -> the far tail is the discriminating feature,"
        " and a spatial window destroys it")


def test_e_two_bins_one_peak():
    say("\n[e] TWO components at EXACTLY bins 4 and 5, a4 = +sqrt(0.318), "
        "a5 = -sqrt(0.433)")
    u = A4 * np.exp(1j * K4 * X) + A5 * np.exp(1j * K5 * X)
    for pad in (1, 8, 16):
        kk, U = _spatial_spectrum(u, pad, "rect")
        P = np.abs(U) ** 2
        jpk = int(np.argmax(P))
        pks = _count_peaks(P, rel=1e-4)
        vals = np.sort(P[pks])[::-1] / P[jpk]
        say(f"    pad={pad:2d}x : global maximum at {sa.to_inv_um(kk[jpk]):.6f} um^-1"
            f" ; {len(pks)} local maxima above 0.01% ; "
            f"next maxima {', '.join(f'{v:.4f}' for v in vals[1:4])}")
        if pad == 16:
            kpk16 = kk[jpk]
            n_big = int((np.array(vals) > 0.05).sum())
    say(f"    16x zero padding gives ONE dominant maximum at "
        f"{sa.to_inv_um(kpk16):.4f} um^-1")
    say("    (the brief states 5.5990 um^-1 with side maxima at 0.8% and 0.25%)")
    _check("16x padding yields a single dominant maximum near 5.599 um^-1",
           abs(sa.to_inv_um(kpk16) - 5.5990) < 0.01 and n_big == 1,
           f"k={sa.to_inv_um(kpk16):.4f}, dominant maxima={n_big}")
    say("    CONCLUSION: the peak count under zero padding is NOT a mode count.")
    say("    Zero padding interpolates; it adds no resolution.")


def relative_phase_rigidity(m, n_seg=6, seg_len=62, stride=62):
    """std of arg(M(k5)/M(k4)) and of |M(k5)/M(k4)| across segments."""
    k, f, Ms, ts = sa.segment_spectra(m, DX, DT, seg_len, n_seg, stride=stride,
                                      window="hann")
    j4 = int(np.argmin(np.abs(k - K4)))
    j5 = int(np.argmin(np.abs(k - K5)))
    out = []
    for s in range(n_seg):
        P = np.abs(Ms[s]) ** 2
        i4 = int(np.argmax(P[:, j4] * (f > 0)))
        r = Ms[s][i4, j5] / Ms[s][i4, j4]
        out.append(r)
    r = np.array(out)
    ph = np.unwrap(np.angle(r))
    return float(np.std(ph)), float(np.std(np.abs(r)) / np.mean(np.abs(r))), r


def bin_growth_rates(m):
    """Gamma fitted independently to the |bin 4| and |bin 5| envelopes."""
    nt = m.shape[0]
    w = 40
    ts, a4, a5 = [], [], []
    for i0 in range(0, nt - w + 1, 8):
        S = np.fft.fft(m[i0:i0 + w], axis=1)
        a4.append(np.sqrt((np.abs(S[:, 4]) ** 2).mean()))
        a5.append(np.sqrt((np.abs(S[:, 5]) ** 2).mean()))
        ts.append((i0 + w / 2) * DT)
    g4 = sa.growth_fit(np.array(ts), np.array(a4))
    g5 = sa.growth_fit(np.array(ts), np.array(a5))
    return g4, g5


def bin_frequency_centroid(m):
    """peak frequency of bins 4 and 5 over the whole record."""
    k, f, M = sa.spectrum(m, DX, DT, window="hann")
    j4 = int(np.argmin(np.abs(k - K4)))
    j5 = int(np.argmin(np.abs(k - K5)))
    pos = f > 0
    f4 = f[pos][int(np.argmax(np.abs(M[pos, j4]) ** 2))]
    f5 = f[pos][int(np.argmax(np.abs(M[pos, j5]) ** 2))]
    return f4, f5


def single_sinusoid_residual(m, window_x="rect", n_scan=2401):
    """Fit ONE off-grid sinusoid to the peak f-slice of m and return
    (fractional residual power, best k0).

    A single off-grid wave on a rectangular (periodic) box fixes the complex
    amplitude of EVERY bin through the Dirichlet kernel, so the leakage tail is
    a rigid prediction.  The common time-window / growth-envelope factor at a
    fixed f is absorbed into the fitted complex amplitude A, as is the half-cell
    offset of the cell centres.
    """
    k, f, M = sa.spectrum(m, DX, DT, window="hann", window_x=window_x)
    j4 = int(np.argmin(np.abs(k - K4)))
    pos = np.where(f > 0)[0]
    i0 = pos[int(np.argmax(np.abs(M[pos, j4]) ** 2))]
    S = M[i0, :]
    sel = (k > 0.5 * DK) & (k < 12.5 * DK)
    ks, Ss = k[sel], S[sel]
    w = sa._window(NX, window_x)
    xn = np.arange(NX) * DX
    best = (np.inf, np.nan)
    for k0 in np.linspace(2 * DK, 8 * DK, n_scan):
        D = (np.exp(1j * (k0 - ks[:, None]) * xn[None, :]) * w[None, :]).sum(axis=1)
        A = np.vdot(D, Ss) / np.vdot(D, D)
        res = float(np.sum(np.abs(Ss - A * D) ** 2) / np.sum(np.abs(Ss) ** 2))
        if res < best[0]:
            best = (res, float(k0))
    return best


def test_def_discrimination():
    say("\n[d/e/f] every diagnostic run on all three controls")
    md, me, mf = field_d(True), field_e(True), field_f()
    rows = {}
    for name, m in (("d", md), ("e", me), ("f", mf)):
        phstd, ampcv, r = relative_phase_rigidity(m)
        g4, g5 = bin_growth_rates(m)
        f4, f5 = bin_frequency_centroid(m)
        k, fax, M = sa.spectrum(m, DX, DT, window="hann", pad_x=16)
        i, sl = sa.f_slice(M, fax, F_HARD)
        P = np.abs(sl) ** 2 * (k > 0)
        pks = _count_peaks(P, rel=0.05)
        res, k0fit = single_sinusoid_residual(m)
        # band_content on the SAME f-slice, disjoint bands about k = 5.5 um^-1
        kk, ff, MM = sa.spectrum(m, DX, DT, window="hann")
        posf = np.where(ff > 0)[0]
        jj4 = int(np.argmin(np.abs(kk - K4)))
        i_sl = posf[int(np.argmax(np.abs(MM[posf, jj4]) ** 2))]
        bc = sa.band_content(kk, MM[i_sl], K_OFFGRID, Q_PUMP, 1.5 * DK,
                             quantity="power")
        # pair correlator at the same slice, K_pump = k4 + k5
        kc, fc, Ms, ts = sa.segment_spectra(m, DX, DT, 62, 6, stride=62,
                                            window="hann")
        pc4 = int(np.argmin(np.abs(kc - K4)))
        pfs = np.where(fc > 0)[0]
        ic = pfs[int(np.argmax(np.abs(Ms[0][pfs, pc4]) ** 2))]
        # the pump frequency is KNOWN independently (f_SAW), so use the exact
        # value, not the segment bin centre
        f_pump_true = {"d": 2 * F_HARD, "e": 2 * F_HARD,
                       "f": F_F4 + F_F5}[name]
        om_p = 2 * np.pi * f_pump_true
        pcres = sa.pair_correlator(Ms[:, ic, :], ts, kc, K4 + K5, om_p)
        rows[name] = dict(
            band_plus=bc["fractions"]["plus"],
            C_pair=float(pcres["C"][pc4]),
            npeak16=len(pks),
            kpeak16=sa.to_inv_um(k[int(np.argmax(P))]),
            phase_std_deg=np.degrees(phstd),
            amp_ratio_cv=ampcv,
            g4=g4[0], g5=g5[0], dg=abs(g4[0] - g5[0]),
            f4=f4, f5=f5, df=abs(f4 - f5),
            resid=res, k0fit=sa.to_inv_um(k0fit),
        )
        say(f"    [{name}] 16x-pad peaks>5% = {len(pks)} at "
            f"{rows[name]['kpeak16']:.4f} um^-1 | "
            f"arg(M5/M4) std = {rows[name]['phase_std_deg']:.3e} deg | "
            f"|M5/M4| CV = {ampcv:.3e}")
        say(f"         Gamma(bin4) = {g4[0]:.4e} 1/s (r2 {g4[2]:.5f}), "
            f"Gamma(bin5) = {g5[0]:.4e} 1/s (r2 {g5[2]:.5f}), "
            f"|dGamma| = {rows[name]['dg']:.3e}")
        say(f"         f(bin4) = {f4/1e9:.4f} GHz, f(bin5) = {f5/1e9:.4f} GHz, "
            f"|df| = {rows[name]['df']/1e6:.2f} MHz")
        say(f"         single-off-grid-sinusoid fit: residual = {res:.4e} "
            f"at k0 = {rows[name]['k0fit']:.4f} um^-1")
        say(f"         band_content (disjoint, halfwidth 1.5 bins about 5.5 um^-1): "
            f"plus = {bc['fractions']['plus']:.4f}, minus = "
            f"{bc['fractions']['minus']:.3e}, k0 = {bc['fractions']['k0']:.3e}, "
            f"rest = {bc['fractions']['rest']:.4f}, sum = {bc['sum_check']:.12f}")
        say(f"         pair correlator C(bin 4) at K_pump = k4+k5 = "
            f"{float(pcres['C'][pc4]):.6f}  (6 x 62-sample segments, "
            f"exact f_pump = {f_pump_true/1e9:.4f} GHz)")

    say("\n    --- what separates what ---")
    _check("peak count does NOT separate d from e",
           rows["d"]["npeak16"] == rows["e"]["npeak16"] == 1)
    say("    (d) is not at the exact numerical floor: its residual phase wobble of")
    say("    ~3e-3 deg comes from the negative-frequency image leaking through the")
    say("    growth-broadened line. It is orders of magnitude below anything a real")
    say("    measurement could resolve, so it is NOT a usable discriminator.")
    _check("phase rigidity does NOT separate d from e (both rigid to <0.1 deg)",
           rows["d"]["phase_std_deg"] < 0.1 and rows["e"]["phase_std_deg"] < 0.1)
    _check("amplitude proportionality does NOT separate d from e (both CV < 1e-3)",
           rows["d"]["amp_ratio_cv"] < 1e-3 and rows["e"]["amp_ratio_cv"] < 1e-3)
    _check("phase rigidity and amplitude proportionality DO flag f",
           rows["f"]["phase_std_deg"] > 1.0 and rows["f"]["amp_ratio_cv"] > 0.1)
    _check("equal growth rates do NOT separate d from e",
           rows["d"]["dg"] / GAMMA < 5e-3 and rows["e"]["dg"] / GAMMA < 5e-3)
    _check("growth-rate difference DOES flag f",
           rows["f"]["dg"] > 0.2 * GAMMA)
    _check("frequency separation DOES flag f", rows["f"]["df"] > 3 * (1 / (NT_HARD * DT)))
    _check("the single-sinusoid leakage-tail fit DOES separate d from e",
           rows["d"]["resid"] < 1e-3 and rows["e"]["resid"] > 20 * rows["d"]["resid"])
    say("    REFUTED IF: the leakage-tail residual had been small for (e) as well;")
    say("    then NO diagnostic in this module would separate the two hypotheses.")
    _write_table(rows)


DIAG_ROWS = [
    ("8x/16x zero padding: number of peaks", "npeak16"),
    ("inter-bin phase rigidity, std arg(M5/M4) [deg]", "phase_std_deg"),
    ("amplitude proportionality, CV of &#124;M5/M4&#124;", "amp_ratio_cv"),
    ("growth-rate difference &#124;dGamma&#124; [1/s]", "dg"),
    ("bin-frequency difference &#124;df&#124; [Hz]", "df"),
    ("single-off-grid-sinusoid fit residual (**TRAIN-FIT**, synthetic)", "resid"),
    ("band_content: disjoint 'plus' band fraction of the f-slice", "band_plus"),
    ("pair correlator C at bin 4, K_pump = k4+k5", "C_pair"),
]


def _write_table(rows):
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{v:.4g}"
    L = []
    L.append("# DISCRIMINATION TABLE\n")
    # Fit-quality category statement, added 2026-09-18.  model_comparison.md
    # section 0 defines three tags -- TRAIN-FIT, TRANSFER-REFIT, HELD-OUT --
    # because the three had been quoted in one column and compared to one
    # tolerance.  This table is swept and the answer is "none of the three", so
    # say so here rather than leave a reader to infer it.
    L.append("**Fit-quality categories** (swept 2026-09-18). "
             "`model_comparison.md` section 0 defines three tags -- "
             "**TRAIN-FIT**, **TRANSFER-REFIT** and **HELD-OUT** -- because the "
             "three had been quoted interchangeably and compared against the "
             "same tolerance. **No number in this file is a held-out "
             "prediction.** Every value below is a diagnostic's response to "
             "*synthetic data built to a known truth*, which is what lets the "
             "table say 'separates' or 'does not separate' at all. The one fit "
             "residual present -- the single-off-grid-sinusoid row -- is a "
             "**TRAIN-FIT** residual on that synthetic data, not a prediction "
             "and not a statement about sim40. The caveats at the end already "
             "say so; the tag makes it explicit.\n")
    L.append("Hypotheses, all evaluated on the sim40 grid "
             "(N_x = 1024, dx = 5 nm, dt = 20 ps, 376 samples):\n")
    L.append("- **(d)** ONE off-grid wave at k = 5.5 um^-1 -- the *leakage* hypothesis.")
    L.append("- **(e)** TWO components at exactly bins 4 and 5, same frequency, same")
    L.append("  growth rate, fixed relative phase, a4 = +sqrt(0.318), a5 = -sqrt(0.433)")
    L.append("  -- the *two phase-locked components* hypothesis (a single unstable")
    L.append("  Floquet eigenvector with two Fourier components does exactly this).")
    L.append("- **(f)** two components at bins 4 and 5 with DIFFERENT frequencies and")
    L.append("  DIFFERENT growth rates -- the easy case.\n")
    L.append("| diagnostic | (d) one off-grid wave | (e) two locked components | (f) two independent | separates d/e? | separates e/f? |")
    L.append("|---|---|---|---|---|---|")
    verdict = {
        "npeak16": ("NO", "YES"),
        "phase_std_deg": ("NO", "YES"),
        "amp_ratio_cv": ("NO", "YES"),
        "dg": ("NO", "YES"),
        "df": ("NO", "YES"),
        "resid": ("YES", "YES"),
        "band_plus": ("NO", "NO"),
        "C_pair": ("NO", "NO"),
    }
    for label, key in DIAG_ROWS:
        v = verdict[key]
        L.append(f"| {label} | {fmt(rows['d'][key])} | {fmt(rows['e'][key])} | "
                 f"{fmt(rows['f'][key])} | **{v[0]}** | **{v[1]}** |")
    L.append("")
    L.append("## Required negative results\n")
    L.append("Six diagnostics are **non-discriminating for the question that matters**")
    L.append("(is the bin-4/bin-5 weight one leaked wave or two phase-locked")
    L.append("components?). They are recorded here as findings, not as failures:\n")
    L.append("1. **Peak counting under zero padding.** (d) and (e) both give exactly one")
    L.append(f"   dominant maximum under 16x padding, at {rows['d']['kpeak16']:.4f} and")
    L.append(f"   {rows['e']['kpeak16']:.4f} um^-1 respectively. Zero padding interpolates")
    L.append("   the Dirichlet kernel; it adds no resolution and cannot count modes.")
    L.append(f"2. **Fixed inter-bin phase.** std arg(M5/M4) = "
             f"{rows['d']['phase_std_deg']:.2e} deg for (d) and "
             f"{rows['e']['phase_std_deg']:.2e} deg for (e): both rigid far below")
    L.append("   anything a measurement resolves. Two components growing along ONE")
    L.append("   unstable eigenvector hold a rigid relative phase exactly as leakage")
    L.append("   from one wave does. ((d) is not at the exact floor only because the")
    L.append("   negative-frequency image leaks through the growth-broadened line.)")
    L.append(f"3. **Proportional bin amplitudes.** CV of &#124;M5/M4&#124; = "
             f"{rows['d']['amp_ratio_cv']:.2e} for (d), "
             f"{rows['e']['amp_ratio_cv']:.2e} for (e).")
    L.append("4. **Equal growth rates.** Both bins grow at one rate in (d) and in (e).\n")
    L.append(f"5. **The pair correlator itself.** C(bin 4) = {rows['d']['C_pair']:.4f} for (d),")
    L.append(f"   {rows['e']['C_pair']:.4f} for (e), {rows['f']['C_pair']:.4f} for (f) on the same")
    L.append("   f-slice recipe. It measures phase rigidity across segments, which a")
    L.append("   single leaked wave has in full. A high C is evidence AGAINST independent")
    L.append("   thermal populations; it is NOT evidence for two components.")
    L.append(f"6. **band_content.** The disjoint 'plus' fraction is "
             f"{rows['d']['band_plus']:.3f} / {rows['e']['band_plus']:.3f} / "
             f"{rows['f']['band_plus']:.3f} for (d)/(e)/(f). It is the right")
    L.append("   bookkeeping tool -- no bin counted twice -- but it does not count modes.")
    L.append("")
    L.append("These DO separate (e) from (f), i.e. they can still refute the")
    L.append("*independent-modes* reading. They cannot decide leakage vs. locked pair.\n")
    L.append("## The one diagnostic that does separate (d) from (e)\n")
    L.append("The **leakage tail**. A single off-grid sinusoid on a rectangular (periodic)")
    L.append("box fixes the amplitude AND phase of every other bin through the Dirichlet")
    L.append("kernel. Fitting one off-grid sinusoid to the whole positive-k f-slice gives")
    L.append(f"a fractional residual of {rows['d']['resid']:.3e} on (d) and")
    L.append(f"{rows['e']['resid']:.3e} on (e) -- a factor "
             f"{rows['e']['resid']/max(rows['d']['resid'],1e-300):.3g}.\n")
    L.append("Caveats that must be honoured before this is used on sim40 data:\n")
    L.append("- The tail is the *only* lever, so it is exactly the part of the spectrum")
    L.append("  that background, other modes and finite precision contaminate first.")
    L.append("- Any spatial window destroys it. Under a Hann window in x the far tail is")
    L.append("  suppressed by orders of magnitude (see control (d) output), so the fit")
    L.append("  must be done with the rectangular/periodic spatial window only.")
    L.append("- Whether the tail survives above the background in the real sim40")
    L.append("  f_K-slice is NOT established here. That is a separate measurement.")
    L.append("  Until it is made, the honest verdict for the sim40 bin-4/bin-5 feature")
    L.append("  is **indistinguishable with current data**, not 'leakage' and not")
    L.append("  'two-component pair'.\n")
    with open(os.path.join(HERE, "DISCRIMINATION_TABLE.md"), "w") as fh:
        fh.write("\n".join(L))


# ============================================================================
# sim40 record bookkeeping
# ============================================================================

def test_sim40_resolution():
    say("\n[res] sim40 late-record sampling and the frequency resolution it supports")
    cp = os.path.join(DATA, "sim40_checkpoints", "sim40_full_2fK_eps7e-05.npz")
    if os.path.isfile(cp):
        d = np.load(cp)
        nt, nx = d["my_xt"].shape
        dt = float(d["DT_REC"]); dx = float(d["CX"])
        late = nt - nt // 2
        say(f"    MEASURED from {os.path.basename(cp)}: my_xt shape {(nt, nx)}, "
            f"DT_REC = {dt*1e12:.3f} ps, CX = {dx*1e9:.3f} nm")
        say(f"    second half (legacy trange = (nt//2, nt)): {late} samples")
        _check("376 late samples", late == 376, f"got {late}")
        _check("20 ps cadence", abs(dt - 20e-12) < 1e-15)
    else:
        say("    checkpoint not found -- NOT DETERMINABLE FROM FILES")
        nt, dt, late = 751, 20e-12, 376
    r_rect = sa.resolution_report(late, dt, "rect")
    r_hann = sa.resolution_report(late, dt, "hann")
    say(f"    T = {r_rect['T']*1e9:.3f} ns -> df_bin = {r_rect['df_bin']/1e6:.5f} MHz")
    _check("df_bin = 132.98 MHz", abs(r_rect["df_bin"] - 132.978723e6) < 1e3)
    say(f"    rectangular: ENBW {r_rect['enbw_Hz']/1e6:.2f} MHz, "
        f"-3 dB width {r_rect['fwhm_3dB_Hz']/1e6:.2f} MHz")
    say(f"    Hann:        ENBW {r_hann['enbw_Hz']/1e6:.2f} MHz, "
        f"-3 dB width {r_hann['fwhm_3dB_Hz']/1e6:.2f} MHz, "
        f"first zero {r_hann['first_zero_Hz']/1e6:.2f} MHz")
    say("    A Hann window costs a factor 1.44 in resolvable frequency separation:")
    say(f"    two lines closer than ~{r_hann['fwhm_3dB_Hz']/1e6:.0f} MHz are NOT resolved.")
    for G in (1e8, 3e8, 6e8, 1e9):
        rg = sa.resolution_report(late, dt, "hann", growth_rate=G)
        say(f"    growth Gamma = {G:.1e} 1/s -> envelope HWHM "
            f"{rg['growth_HWHM_Hz']/1e6:.2f} MHz = {rg['growth_in_bins']:.2f} bins;"
            f" effective line width {rg['effective_width_Hz']/1e6:.2f} MHz")
    say("    So a growth envelope reaches the one-bin level at Gamma ~ 8e8 1/s; below")
    say("    that the Hann main lobe dominates and the line width stays ~190-270 MHz.")
    say("    Either way the usable frequency resolution of this record is a few times")
    say("    100 MHz, NOT 133 MHz, and any 'omega_1 + omega_2 = omega_p' test must")
    say("    carry that width as its uncertainty.")
    # measured growth rate of the bin-4/bin-5 amplitude in the candidate run
    if os.path.isfile(cp):
        d = np.load(cp)
        my = d["my_xt"].astype(np.float64)
        w = 64
        ts, a4, a5 = [], [], []
        for i0 in range(0, my.shape[0] - w + 1, 16):
            S = np.fft.fft(my[i0:i0 + w], axis=1)
            a4.append(np.sqrt((np.abs(S[:, 4]) ** 2).mean()))
            a5.append(np.sqrt((np.abs(S[:, 5]) ** 2).mean()))
            ts.append((i0 + w / 2) * dt)
        ts = np.array(ts)
        g4 = sa.growth_fit(ts, np.array(a4), t0=2e-9, t1=7e-9)
        g5 = sa.growth_fit(ts, np.array(a5), t0=2e-9, t1=7e-9)
        say("    MEASURED (eps_0 = 7e-5, full_2fK checkpoint), sliding 64-sample")
        say("    windows, ln|bin amplitude| fitted over t = 2-7 ns:")
        say(f"      Gamma(bin 4) = {g4[0]:.3e} 1/s (r2 = {g4[2]:.4f})")
        say(f"      Gamma(bin 5) = {g5[0]:.3e} 1/s (r2 = {g5[2]:.4f})")
        rg = sa.resolution_report(late, dt, "hann", growth_rate=max(g4[0], g5[0]))
        say(f"      -> envelope HWHM {rg['growth_HWHM_Hz']/1e6:.1f} MHz "
            f"({rg['growth_in_bins']:.2f} bins); effective line width "
            f"{rg['effective_width_Hz']/1e6:.0f} MHz")
        say("      This sizes the linewidth ONLY. It is not a claim about how many")
        say("      modes those bins contain.")
    say("    Welch segmentation makes this much worse: 6 segments of 62 samples give")
    say(f"    df_seg = {1/(62*dt)/1e6:.1f} MHz per bin -- f_K = 3 GHz is not on a bin.")


# ============================================================================

# ============================================================================
# R1. REGRESSION -- auditor report 2026-09-18 section 4: the frequency
#     estimator of model_comparison.project_and_estimate_frequency() must read
#     the branch of THIS module's convention, not the counter-propagating one.
#
#     The convention: M(k,f) = sum m exp(-i k x + i 2 pi f t), so a wave
#     exp(i(k0 x - 2 pi f0 t)) sits at (+k0, +f0).  Projecting m(x,t) on
#     exp(-i k0 x) leaves a time series exp(-i 2 pi f0 t); a plain np.fft.fft
#     of that series puts its energy on the NEGATIVE fft frequencies, so a
#     search over POSITIVE fft frequencies returns the wave travelling the
#     OTHER way.  Two amplitudes and two DIFFERENT frequencies are used here so
#     that a sign error cannot survive by symmetry.
# ============================================================================

NT_FREQ = 376


def _counterprop_field(f_fwd, a_fwd, f_bwd, a_bwd, kappa=4, nt=NT_FREQ):
    """m(x,t) = a_fwd cos(k x - 2 pi f_fwd t) + a_bwd cos(k x + 2 pi f_bwd t).

    First term travels toward +x, second toward -x, both with |k| = kappa*dk.
    In this module's convention the forward wave occupies (+kappa, +f_fwd) and
    the backward wave (+kappa, -f_bwd).  An estimator that projects on
    exp(-i k x) and then searches POSITIVE frequencies must therefore return
    f_fwd for any a_bwd, and must never return f_bwd.
    """
    t = np.arange(nt)[:, None] * DT
    k = kappa * DK
    return (a_fwd * np.cos(k * X[None, :] - 2 * np.pi * f_fwd * t)
            + a_bwd * np.cos(k * X[None, :] + 2 * np.pi * f_bwd * t))


def test_freq_estimator_branch():
    say("\n[R1] projected-frequency estimator reads the forward branch "
        "(auditor 2026-09-18 sec. 4)")
    import model_comparison as mc
    tol = 25e6                     # MHz-level; the fft bin here is 133 MHz
    cases = [
        ("forward-strong 3.0 GHz / backward-weak 2.5 GHz",
         dict(f_fwd=3.0e9, a_fwd=1.0, f_bwd=2.5e9, a_bwd=0.01), 3.0e9, 2.5e9),
        ("backward-strong 3.5 GHz / forward-weak 2.2 GHz",
         dict(f_fwd=2.2e9, a_fwd=0.01, f_bwd=3.5e9, a_bwd=1.0), 2.2e9, 3.5e9),
    ]
    for lbl, kw, f_want, f_wrong in cases:
        m = _counterprop_field(**kw)
        r = mc.project_and_estimate_frequency(m, 4.0, DT, t0=0, n=NT_FREQ)
        got = r["f_peak"]
        say("     %s" % lbl)
        say("       returned %.6f GHz ; correct branch %.6f ; wrong branch %.6f"
            % (got / 1e9, f_want / 1e9, f_wrong / 1e9))
        _check("estimator returns the forward component (%s)" % lbl,
               abs(got - f_want) < tol,
               "got %.6f GHz, want %.6f GHz (+-%.0f MHz)"
               % (got / 1e9, f_want / 1e9, tol / 1e6))
        _check("estimator does NOT return the counter-propagating component "
               "(%s)" % lbl, abs(got - f_wrong) > 10 * tol,
               "got %.6f GHz, wrong branch %.6f GHz" % (got / 1e9, f_wrong / 1e9))
    say("     REFUTED IF: the returned frequency had tracked the counter-")
    say("     propagating wave (2.4981 GHz / 3.5 GHz here). That was the state")
    say("     of the code before this regression test was added.")


# ============================================================================
# R2. REGRESSION -- auditor report 2026-09-18 section 5: the null row quoted
#     for `1/sqrt(M)` must be the row of the REQUESTED wavenumber.  The target
#     list merges duplicates, so a fixed position (`min(2, n-1)`) lands on the
#     near-k_SAW row while the published point is the half-k_SAW row.  The two
#     nulls are different distributions, so the quoted percentile was taken from
#     the wrong one.
# ============================================================================

def test_null_row_is_requested_target():
    say("\n[R2] the published null row is found by wavenumber + partner, not "
        "by list position (auditor 2026-09-18 sec. 5)")
    import rc_common as rc
    d37 = np.load(os.path.join(DATA, "sim37_suhl_control.npz"))
    K_saw = float(d37["saw_kSAW"])
    # the REAL step2 axis: sim37 is 2048 cells x CX, so dk = dk_sim40 / 2 and
    # the bin nearest k_SAW/2 is the UPPER straddling bin -- which is why the
    # de-duplicated target list puts the published point at index 1 and the
    # near-k_SAW row at index 2, the two rows the positional rule confused.
    nx37, dx37 = int(d37["saw_my_xt"].shape[1]), float(d37["CX"])
    k = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx37, d=dx37))
    dk = float(k[1] - k[0])
    thr = 1 / np.sqrt(6.0)                                      # M = 6 segments
    for lbl, K_pump in (("MEL / MR channel (K_pump = k_SAW)", K_saw),
                        ("uniform-Suhl control (K_pump = 0)", 0.0)):
        tg = rc.targets(k, K_pump, k_ref=K_saw)
        j2, ok, snap = sa._partner_index(k, K_pump)
        j_half = int(np.argmin(np.abs(k - K_saw / 2)))
        j_near = int(np.argmin(np.abs(k - K_saw)))
        n_half = [n for n, (j, _) in enumerate(tg) if j == j_half]
        n_near = [n for n, (j, _) in enumerate(tg) if j == j_near]
        assert len(n_half) == 1 and len(n_near) == 1
        n_half, n_near = n_half[0], n_near[0]
        # two nulls that differ markedly: thr is below ALL of the half-q null
        # and above ALL of the near-q null
        rng = np.random.default_rng(20260918)
        C_null = np.full((2000, len(tg)), 0.5)
        C_null[:, n_half] = rng.uniform(0.80, 0.90, 2000)
        C_null[:, n_near] = rng.uniform(0.00, 0.10, 2000)
        pct = lambda n: 100.0 * float((C_null[:, n] <= thr).mean())   # noqa: E731
        n = rc.select_target_row(tg, k, j2, K_saw / 2, K_pump)
        say("     %s" % lbl)
        say("       rows: %s" % ", ".join("%d:%.4f um^-1" % (i, sa.to_inv_um(k[j]))
                                          for i, (j, _) in enumerate(tg)))
        say("       requested %.4f um^-1 -> row %d (%.4f um^-1); half-q row is "
            "%d, near-q row is %d" % (sa.to_inv_um(K_saw / 2), n,
                                      sa.to_inv_um(k[tg[n][0]]), n_half, n_near))
        say("       percentile of 1/sqrt(M) = %.4f reported = %.1f ; half-q "
            "null gives %.1f ; near-q null gives %.1f"
            % (thr, pct(n), pct(n_half), pct(n_near)))
        _check("selected row carries the requested wavenumber (%s)" % lbl,
               abs(k[tg[n][0]] - K_saw / 2) <= 0.5 * dk * (1 + 1e-9) and n == n_half,
               "row %d at %.6f um^-1" % (n, sa.to_inv_um(k[tg[n][0]])))
        _check("selected row's pair partner is K_pump - k_1 (%s)" % lbl,
               abs((K_pump - k[tg[n][0]]) - k[j2[tg[n][0]]]) <= 0.5 * dk * (1 + 1e-9))
        _check("quoted percentile comes from the requested target's null (%s)"
               % lbl, abs(pct(n) - pct(n_half)) < 1e-12 and abs(pct(n) - pct(n_near)) > 50,
               "reported %.1f, half-q %.1f, near-q %.1f"
               % (pct(n), pct(n_half), pct(n_near)))
        raised = False
        try:
            rc.select_target_row(tg, k, j2, 0.37 * K_saw, K_pump)
        except (KeyError, ValueError, LookupError) as exc:
            raised = True
            detail = type(exc).__name__
        _check("a target with no matching row RAISES instead of falling back "
               "(%s)" % lbl, raised, detail if raised else "returned silently")
    say("     REFUTED IF: the selector had returned the near-k_SAW row for a")
    say("     half-k_SAW request, which is what `min(2, n-1)` did.")


# ============================================================================
# R3. Non-degenerate pair control with the two components extracted
#     INDEPENDENTLY, each at its own frequency.
#
#     The audit of 2026-09-18 (section 9) notes that controls (a)/(a')/(b) read
#     BOTH members of the pair from the single f1 slice, so the k5 member enters
#     only through Hann leakage and the control tests the pump-phase SIGN rather
#     than pair extraction.  That leakage is measured here (about 3.04e-5 of the
#     power the same bin carries at its own f2) and the control is then repeated
#     with k4 taken from the f1 slice and k5 from the f2 slice.  Answers known
#     before the run: C = 1 for the phase-locked pair with the correct sign, and
#     the same analytic residual |<exp(-2 i w_p t_s)>| for the wrong sign.
# ============================================================================

def _own_frequency_pair_slice(m):
    """(k, t_s, slice, j4, j5, leak_power_ratio, f, i1, i2).

    `slice` is the f1 slice EXCEPT at bin k5, which is read from the f2 slice:
    each member of the pair is extracted at its own frequency.  The correlator
    that consumes it is unchanged.
    """
    k, f, Ms, ts = sa.segment_spectra(m, DX, DT, SEG_LEN, N_SEG, stride=STRIDE,
                                      window="hann")
    i1 = int(np.argmin(np.abs(f - F1)))
    i2 = int(np.argmin(np.abs(f - F2)))
    j4 = int(np.argmin(np.abs(k - K4)))
    j5 = int(np.argmin(np.abs(k - K5)))
    sl = Ms[:, i1, :].copy()
    leak = float(np.mean(np.abs(Ms[:, i1, j5]) ** 2 / np.abs(Ms[:, i2, j5]) ** 2))
    sl[:, j5] = Ms[:, i2, j5]
    return k, ts, sl, j4, j5, leak, f, i1, i2


def test_a2_pair_extracted_at_own_frequencies():
    say("\n[a''] the SAME pair, but k4 read at f1 and k5 read at f2 -- two "
        "genuinely different frequencies, extracted independently")
    m = _phase_locked_field([0.3] * N_SEG, [1.1] * N_SEG)
    k, ts, sl, j4, j5, leak, f, i1, i2 = _own_frequency_pair_slice(m)
    df = float(f[1] - f[0])
    say(f"    segment bin = {df/1e6:.2f} MHz; f1 = {F1/1e9:.6f} GHz is bin {i1}, "
        f"f2 = {F2/1e9:.6f} GHz is bin {i2} -- {abs(i2-i1)} bins apart")
    say(f"    single-slice reading of k5 at f1 holds {leak:.4e} of the power that")
    say("    same bin holds at its own f2: that is what controls (a)/(a')/(b)")
    say("    were actually testing on the k5 member (Hann leakage, audit sec. 9).")
    _check("the pair members come from DIFFERENT frequency bins",
           i1 != i2 and abs(f[i1] - F1) < 1e-9 * df and abs(f[i2] - F2) < 1e-9 * df,
           f"bins {i1} and {i2}")
    _check("reading k5 at its own f2 is not the same measurement as reading it "
           "at f1", leak < 1e-3, f"leakage power ratio {leak:.4e}")
    res_p = sa.pair_correlator(sl, ts, k, KP, 2 * np.pi * FP, phase_sign=+1)
    res_m = sa.pair_correlator(sl, ts, k, KP, 2 * np.pi * FP, phase_sign=-1)
    Cp, Cm = float(res_p["C"][j4]), float(res_m["C"][j4])
    Cpred = float(abs(np.mean(np.exp(-2j * 2 * np.pi * FP * ts))))
    say(f"    C(k4) with independent extraction, correct sign = {Cp:.12f}   expected 1")
    say(f"    C(k4) with the wrong sign = {Cm:.12f}, analytic residual = {Cpred:.10f}")
    _check("C = 1 for the phase-locked pair extracted at its own frequencies",
           abs(Cp - 1.0) < 1e-9, f"C={Cp:.12f}")
    _check("the wrong sign still reproduces the analytic residual under "
           "independent extraction", abs(Cm - Cpred) < 1e-9)
    rng = np.random.default_rng(7)
    mr = _phase_locked_field(rng.uniform(0, 2 * np.pi, N_SEG),
                             rng.uniform(0, 2 * np.pi, N_SEG))
    _, tsr, slr, _, _, _, _, _, _ = _own_frequency_pair_slice(mr)
    Cr = float(sa.pair_correlator(slr, tsr, k, KP, 2 * np.pi * FP,
                                  phase_sign=+1)["C"][j4])
    say(f"    same extraction, INDEPENDENT per-segment phases: C(k4) = {Cr:.6f}")
    _check("independent per-segment phases do NOT give C = 1 under this "
           "extraction", Cr < 0.9, f"C={Cr:.6f}")
    say("    REFUTED IF: C had stayed at 1 for independent phases, or had left 1")
    say("    for the locked pair once the two members were read at their own")
    say("    frequencies. Neither happened.")


def main():
    say("=" * 78)
    say("saw_analysis control suite")
    say("=" * 78)
    say(f"grid: N_x = {NX}, dx = {DX*1e9:.1f} nm, L = {LX*1e6:.2f} um, "
        f"dk = {sa.to_inv_um(DK):.7f} um^-1")
    say(f"bin 4 = {sa.to_inv_um(K4):.6f}, bin 5 = {sa.to_inv_um(K5):.6f}, "
        f"sum = {sa.to_inv_um(K4+K5):.6f} um^-1")
    say(f"physical pump q = 2 pi f_SAW / v_SAW = {sa.to_inv_um(Q_PUMP):.6f} um^-1")
    say(f"difference = {sa.to_inv_um(K4+K5-Q_PUMP):.6f} um^-1 = "
        f"{100*(K4+K5-Q_PUMP)/Q_PUMP:.2f} % -- the bin sum is the GRID's 9th")
    say(f"harmonic, not the pump. q*L/(2 pi) = {Q_PUMP*LX/(2*np.pi):.4f}: the box is")
    say("incommensurate with the pump, so the pump has NO exact bin.")
    say(sa.K_SIGN_DOC)

    test_convention_sign()
    test_freq_estimator_branch()
    test_null_row_is_requested_target()
    test_fft2_equivalence()
    test_amplitude_power_not_mixed()
    test_a_phase_locked_correlator()
    test_a2_pair_extracted_at_own_frequencies()
    test_a_wrong_sign_fails()
    test_b_random_phases_land_in_null()
    test_c_uniform_k0()
    test_band_content_disjoint_random()
    test_d_single_offgrid_leakage()
    test_e_two_bins_one_peak()
    test_def_discrimination()
    test_sim40_resolution()

    # Reconciled 2026-09-18.  The documentation said "33 controls" while the
    # suite yielded 32 result items; the extra came from counting the banner
    # line below, which contains the pass token.  The suite now states its own
    # item count so no document has to hard-code one.
    n_items = sum(1 for s in LOG
                  if s.startswith("  PASS  ") or s.startswith("  FAIL  "))
    say("\n" + "=" * 78)
    say("%d result items." % n_items)
    say("A naive grep for the pass token over this file returns one more than "
        "that, because the banner line below matches as well.")
    say("ALL CONTROLS PASSED")
    say("=" * 78)
    with open(os.path.join(HERE, "TEST_RESULTS.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
