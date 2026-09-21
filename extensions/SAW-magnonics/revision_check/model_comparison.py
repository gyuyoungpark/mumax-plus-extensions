"""model_comparison.py -- ONE vs TWO spatial components in the sim40 f_K feature.

Decides, by model comparison on the FULL COMPLEX SPATIAL PROFILE, whether the
sim40 eps_0 = 7e-5 f_K-slice feature (weight on FFT bins 4 and 5) needs one
off-grid spatial component or two.  Obeys PREREGISTRATION.md, which was written
before any fit was run, and imports saw_analysis.py for all conventions.

NOT decided by: power in two bins, peak counting under zero padding, inter-bin
phase rigidity, proportional amplitudes, equal growth rates, or the pair
correlator -- all shown non-discriminating in DISCRIMINATION_TABLE.md.

Stages (checkpointed to out/mc_state.npz; a finished stage is skipped unless
named explicitly):

    python model_comparison.py --stage all
    python model_comparison.py --stage fit controls bootstrap freq aux robust figs

Writes only into revision_check/.  Never touches src/, data/ or the .tex files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import saw_analysis as sa                                          # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
CKPT = os.path.join(DATA, "sim40_checkpoints")

# ---------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS (PREREGISTRATION.md sections 1-5); do not retune.
# ---------------------------------------------------------------------------
F0 = 3.0e9                 # analysis frequency = f_SAW/2 exactly (not a DFT bin)
V_SAW = 3500.0
NX = 1024
DX = 5e-9
T0_BLOCKS = 375            # published "second half" start
BLOCK_LEN = 94             # 1.88 ns
N_BLOCKS = 4
TRAIN_BLOCKS = [0, 1]
HELD_BLOCKS = [2, 3]
BAND = np.arange(1, 8)             # analysis band A = bins 1..7
PUMP_BINS = np.arange(8, 11)       # bins used to estimate the pump background
INJ_BINS = np.arange(1, 11)        # bins carrying injected residual in controls
FLOOR_BINS = np.arange(12, 22)     # out-of-band bins: measured noise floor
CAL_PRIMARY = np.array([4, 5, 6])
EVAL_PRIMARY = np.array([1, 2, 3, 7])
CAL_SECOND = np.array([4, 5])
EVAL_SECOND = np.array([1, 2, 3, 6, 7])
TAU_ADEQ = 0.25
RHO = 0.50
FP_GATE = 0.05
DET_GATE = 0.80

# Two fit configurations, BOTH reported.
#   PRE  = exactly as pre-registered (kappa in [2,8]).
#   AMD  = documented amendment.  In PRE the M1 optimum runs to the boundary
#          kappa = 8, which lies inside the EXCLUDED pump window and acts only
#          as a smooth 1/(j-kappa) tail-shape generator; AMD confines every
#          component to the analysis band and widens the merge guard.
CFG_PRE = dict(name="preregistered", klo=2.0, khi=8.0, min_sep=0.10, rcond=1e-6)
CFG_AMD = dict(name="amended", klo=2.5, khi=7.5, min_sep=0.25, rcond=1e-6)

DK = 2 * np.pi / (NX * DX)                     # rad/m
KAPPA_Q = 2 * F0 * NX * DX / V_SAW             # q/dk = 8.777142857...


# ---------------------------------------------------------------------------
# loading and the analysis object
# ---------------------------------------------------------------------------

def load_run(eps_tag, label="full_2fK"):
    p = os.path.join(CKPT, "sim40_%s_eps%s.npz" % (label, eps_tag))
    d = np.load(p)
    return dict(m=d["my_xt"].astype(np.float64), dx=float(d["CX"]),
                dt=float(d["DT_REC"]), f_saw=float(d["f_saw"]),
                f_K=float(d["f_K"]), eps0=float(d["eps0"]), path=p)


def block_coeffs(m, dt, f0=F0, t0=T0_BLOCKS, blen=BLOCK_LEN, nb=N_BLOCKS,
                 window="hann"):
    """u_b(x) = sum_t w(t) m(x,t) exp(+i 2 pi f0 t)   (saw_analysis sign), then
    a_j^b = (1/Nx) sum_x u_b(x) exp(-i k_j x).  Returns (nb, Nx) complex.

    The time window and the growth envelope are x-independent: they change only
    the per-block complex scale, never the shape across j.  The one-vs-two
    component test on the SPATIAL profile is therefore immune to the envelope
    and window problems that cripple the frequency test.
    """
    nx = m.shape[1]
    w = np.hanning(blen) if window == "hann" else np.ones(blen)
    out = np.empty((nb, nx), dtype=complex)
    tstart = np.empty(nb)
    for b in range(nb):
        i0 = t0 + b * blen
        tt = np.arange(i0, i0 + blen) * dt
        u = ((m[i0:i0 + blen, :] * w[:, None])
             * np.exp(1j * 2 * np.pi * f0 * tt)[:, None]).sum(0)
        out[b] = np.fft.fft(u) / nx
        tstart[b] = i0 * dt
    return out, tstart


def dirichlet(kappa, bins, N=NX):
    """Exact periodic response of bin j to a continuous wavenumber kappa*dk:
    D_j(kappa) = (1/N) sum_n exp(i 2 pi (kappa - j) n / N)."""
    kap = np.atleast_1d(np.asarray(kappa, dtype=float))
    z = np.pi * (kap[:, None] - np.asarray(bins, dtype=float)[None, :])
    den = N * np.sin(z / N)
    small = np.abs(den) < 1e-12
    val = np.where(small, 1.0, np.sin(z) / np.where(small, 1.0, den))
    r = val * np.exp(1j * z * (N - 1) / N)
    return r[0] if np.ndim(kappa) == 0 else r


def remove_pump(a_full, bins_out, kappa_q=KAPPA_Q, pump_bins=PUMP_BINS, N=NX):
    """Estimate the FIXED-kappa_q pump amplitude on pump_bins only, subtract its
    exact Dirichlet profile from bins_out.  Identical pre-processing for every
    model: no model gets a background parameter another does not."""
    Dp = dirichlet(kappa_q, pump_bins, N)
    Do = dirichlet(kappa_q, bins_out, N)
    c = (np.conj(Dp)[None, :] * a_full[..., pump_bins]).sum(-1) / (np.abs(Dp) ** 2).sum()
    return a_full[..., bins_out] - c[..., None] * Do[None, :], c


# ---------------------------------------------------------------------------
# fits: vectorised exhaustive search + variable projection
# ---------------------------------------------------------------------------

def _varpro(kappas, A, bins, rcond=1e-6, N=NX, full=False):
    G = np.atleast_2d(dirichlet(np.atleast_1d(kappas), bins, N)).T
    coef, *_ = np.linalg.lstsq(G, A.T, rcond=rcond)
    fit = (G @ coef).T
    ss = float((np.abs(A - fit) ** 2).sum())
    return (ss, fit, coef.T) if full else ss


def grid_search_12(A, bins, cfg, n_grid=1201, N=NX):
    """Exhaustive, fully vectorised 1- and 2-component searches.

    ss_1(k)    = ||A||^2 - sum_b |<d_k, A_b>|^2 / <d_k, d_k>
    ss_2(a,b)  = ||A||^2 - sum_b v^H (G^H G)^-1 v,  v = (<d_a,A_b>, <d_b,A_b>)
    solved in closed form for every pair on the grid at once.
    """
    g = np.linspace(cfg["klo"], cfg["khi"], n_grid)
    D = dirichlet(g, bins, N)                          # (ng, nbins)
    tot = float((np.abs(A) ** 2).sum())
    y = np.conj(D) @ A.T                               # (ng, nb)
    Dnn = (np.abs(D) ** 2).sum(1).real                 # (ng,)
    ss1 = tot - (np.abs(y) ** 2).sum(1) / Dnn
    k1 = g[int(np.argmin(ss1))]
    Dab = D.conj() @ D.T                               # (ng, ng)
    det = Dnn[:, None] * Dnn[None, :] - np.abs(Dab) ** 2
    quad = np.zeros((n_grid, n_grid))
    for b in range(A.shape[0]):
        ya, yb = y[:, b][:, None], y[:, b][None, :]
        quad += (Dnn[None, :] * np.abs(ya) ** 2 + Dnn[:, None] * np.abs(yb) ** 2
                 - 2 * np.real(Dab * np.conj(ya) * yb)) / np.where(det > 0, det, np.inf)
    ss2 = tot - quad
    bad = np.abs(g[:, None] - g[None, :]) < cfg["min_sep"]
    ss2 = np.where(bad, np.inf, ss2)
    ij = np.unravel_index(int(np.argmin(ss2)), ss2.shape)
    k2 = np.sort(np.array([g[ij[0]], g[ij[1]]]))
    return dict(k1=np.array([k1]), ss1=float(ss1.min()), k2=k2,
                ss2=float(ss2[ij]), total=tot, grid=g)


def _nelder(f, x0, step, nit=600, tol=1e-11):
    n = x0.size
    sim = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    fv = np.array([f(s) for s in sim])
    for _ in range(nit):
        o = np.argsort(fv); sim, fv = sim[o], fv[o]
        if np.max(np.abs(sim[1:] - sim[0])) < tol:
            break
        cen = sim[:-1].mean(0)
        xr = cen + (cen - sim[-1]); fr = f(xr)
        if fr < fv[0]:
            xe = cen + 2 * (cen - sim[-1]); fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen); fc = f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                fv[1:] = np.array([f(s) for s in sim[1:]])
    return np.sort(sim[int(np.argmin(fv))])


def refine(k, A, bins, cfg, N=NX):
    def pen(kk):
        s = np.sort(kk)
        if s.min() < cfg["klo"] or s.max() > cfg["khi"]:
            return 1e30
        if s.size > 1 and np.min(np.diff(s)) < cfg["min_sep"]:
            return 1e30
        return _varpro(s, A, bins, cfg["rcond"], N)
    return _nelder(pen, np.asarray(k, float), step=1e-3)


def fit_MN(A, bins, ncomp, cfg, N=NX, n_grid=241):
    """N-component fit.  N=1,2 exhaustive; N=3 greedy from the N=2 optimum.
    N>3 is NOT offered: with 7 complex bins those parameters are unidentifiable."""
    gs = grid_search_12(A, bins, cfg, n_grid=max(n_grid, 601) if ncomp == 1 else n_grid, N=N)
    if ncomp == 1:
        k = refine(gs["k1"], A, bins, cfg, N)
    elif ncomp == 2:
        k = refine(gs["k2"], A, bins, cfg, N)
    elif ncomp == 3:
        g = gs["grid"]
        base = refine(gs["k2"], A, bins, cfg, N)
        best, add = np.inf, None
        for gv in g:
            if np.min(np.abs(base - gv)) < cfg["min_sep"]:
                continue
            s = _varpro(np.append(base, gv), A, bins, cfg["rcond"], N)
            if s < best:
                best, add = s, gv
        k = refine(np.sort(np.append(base, add)), A, bins, cfg, N)
    else:
        raise ValueError("N>3 is unidentifiable on 7 complex bins")
    s, fit, coef = _varpro(k, A, bins, cfg["rcond"], N, full=True)
    return dict(kappa=k, ss=s, fit=fit, coef=coef, ncomp=ncomp,
                n_real_params=int(ncomp + 2 * ncomp * A.shape[0]))


# ---------------------------------------------------------------------------
# held-out prediction error (PREREG sec. 4) and the selection rule (sec. 5)
# ---------------------------------------------------------------------------

def heldout_error(kappas, A_held, cal_bins, eval_bins, cfg, N=NX):
    """Wavenumbers FROZEN from the training blocks.  Amplitudes re-estimated on
    cal_bins of the held-out block only; error evaluated on the disjoint
    eval_bins, which are never used to estimate anything for that block.

    E = 1 is the ZERO-PREDICTION baseline (a model with no leakage tail);
    E < 1 means real predictive skill; E > 1 means the model is actively wrong.
    """
    Dc = np.atleast_2d(dirichlet(np.atleast_1d(kappas), cal_bins, N)).T
    De = np.atleast_2d(dirichlet(np.atleast_1d(kappas), eval_bins, N)).T
    ic = np.searchsorted(BAND, cal_bins)
    ie = np.searchsorted(BAND, eval_bins)
    coef, *_ = np.linalg.lstsq(Dc, A_held[:, ic].T, rcond=cfg["rcond"])
    pred = (De @ coef).T
    ye = A_held[:, ie]
    return (float((np.abs(ye - pred) ** 2).sum()) /
            float((np.abs(ye) ** 2).sum())), pred, ye


def select(E1, E2, tau=TAU_ADEQ, rho=RHO):
    if (E2 <= tau) and (E2 <= rho * E1):
        return "two_components_required"
    if E1 <= tau:
        return "single_wave_sufficient"
    return "indistinguishable"


def run_pipeline(a_full, cfg, cal_bins=CAL_PRIMARY, eval_bins=EVAL_PRIMARY,
                 N=NX, fast=False, n_grid=241):
    """FULL end-to-end procedure, re-run in EVERY control/bootstrap realisation:
    pump background removal -> kappa search on the TRAIN blocks -> M1 & M2 ->
    held-out prediction error -> selection rule."""
    A, cpump = remove_pump(a_full, BAND, N=N)
    Atr, Ahe = A[TRAIN_BLOCKS], A[HELD_BLOCKS]
    if fast:
        gs = grid_search_12(Atr, BAND, cfg, n_grid=n_grid, N=N)
        k1, k2, s1, s2 = gs["k1"], gs["k2"], gs["ss1"], gs["ss2"]
        models = {1: dict(kappa=k1, ss=s1), 2: dict(kappa=k2, ss=s2)}
    else:
        models = {n: fit_MN(Atr, BAND, n, cfg, N) for n in (1, 2)}
    E = {}
    for n in (1, 2):
        E[n], pr, ye = heldout_error(models[n]["kappa"], Ahe, cal_bins,
                                     eval_bins, cfg, N)
        models[n]["pred_held"], models[n]["y_held"] = pr, ye
    return dict(A=A, Atr=Atr, Ahe=Ahe, models=models, E=E, E1=E[1], E2=E[2],
                cpump=cpump, selected=select(E[1], E[2]),
                train_power=float((np.abs(Atr) ** 2).sum()))


# ---------------------------------------------------------------------------
# synthetic controls (PREREG sec. 6) and residual models (sec. 7)
# ---------------------------------------------------------------------------

def synth_a(kappas, amps, cpump, N=NX):
    bins = np.arange(N)
    a = np.zeros((amps.shape[0], N), dtype=complex)
    for c, kap in enumerate(np.atleast_1d(kappas)):
        a += amps[:, c][:, None] * dirichlet(kap, bins, N)[None, :]
    return a + cpump[:, None] * dirichlet(KAPPA_Q, bins, N)[None, :]


def noise_draw(rng, model, resid, nb=N_BLOCKS, nbin=len(INJ_BINS)):
    """R1 white (one common variance) / R2 per-bin white (measured per-bin
    variance) / R3 correlated: wild bootstrap of whole measured residual
    bin-vectors, preserving the measured bin-to-bin correlation."""
    if model == "R1":
        s = np.sqrt((np.abs(resid) ** 2).mean() / 2.0)
        return s * (rng.standard_normal((nb, nbin)) + 1j * rng.standard_normal((nb, nbin)))
    if model == "R2":
        s = np.sqrt((np.abs(resid) ** 2).mean(0) / 2.0)[:nbin]
        return s[None, :] * (rng.standard_normal((nb, nbin)) + 1j * rng.standard_normal((nb, nbin)))
    if model == "R3":
        idx = rng.integers(0, resid.shape[0], nb)
        ph = np.exp(2j * np.pi * rng.random(nb))
        return (resid[idx] * ph[:, None])[:, :nbin]
    raise ValueError(model)


def control_realisation(rng, kappas, amps, cpump, resid, model, cfg,
                        cal=CAL_PRIMARY, evl=EVAL_PRIMARY, N=NX, n_grid=241):
    """One synthetic realisation through the IDENTICAL pipeline.  Injection at
    the a_j level is exact: every step upstream of it (time window, projection
    at f0, spatial DFT) is LINEAR, so additive noise there and additive noise in
    m(x,t) are the same thing.  Everything downstream -- pump estimation and
    removal, wavenumber search, M1, M2, held-out error, selection -- is re-run."""
    a = synth_a(kappas, amps, cpump, N)
    a_used = np.zeros((N_BLOCKS, N), dtype=complex)
    a_used[:, INJ_BINS] = a[:, INJ_BINS] + noise_draw(rng, model, resid)
    return run_pipeline(a_used, cfg, cal, evl, N, fast=True, n_grid=n_grid)


def per_block_amps(A, kappas, cfg, N=NX):
    D = np.atleast_2d(dirichlet(np.atleast_1d(kappas), BAND, N)).T
    coef, *_ = np.linalg.lstsq(D, A.T, rcond=cfg["rcond"])
    return coef.T


# ---------------------------------------------------------------------------
# structural diagnostic: the adjacent-bin phase flip (falsifiable)
# ---------------------------------------------------------------------------

def phase_flip_diagnostic(A):
    """A single off-grid component at kappa forces EXACTLY ONE adjacent-bin pair
    (j, j+1) straddling kappa to differ in phase by ~180 deg and every other
    adjacent pair to differ by ~0 deg, because D_j(kappa) ~ sin(pi(kappa-j)) /
    (pi(kappa-j)) alternates sign across kappa.

    REFUTING OUTCOME for 'one component': no adjacent pair inside the signal
    core (bins 3-7) differs by more than 90 deg.  Validated on control (d),
    which shows exactly one -179.8 deg step and 0.2 deg elsewhere.
    """
    ph = np.angle(A, deg=True)
    d = (np.diff(ph, axis=1) + 180) % 360 - 180
    return dict(bins=BAND.tolist(), dphi_deg=d,
                n_flips_all=[int((np.abs(r) > 90).sum()) for r in d],
                core_dphi=d[:, 2:6],
                n_flips_core=int((np.abs(d[:, 2:6]) > 90).sum()))



def step_vs_peak(A):
    """Joint requirement a single off-grid component must satisfy:
    the ~180 deg phase step and the amplitude MAXIMUM must occur at the SAME
    adjacent-bin pair, because |D_j(kappa)| ~ 1/|j-kappa| peaks exactly where
    sin(pi(kappa-j)) changes sign.  If the largest step is at pair (j, j+1) then
    kappa lies in (j, j+1) and, for any such kappa, the ratio |a_p|/|a_j| to the
    observed peak bin p is bounded by max_kappa |kappa-j|/|kappa-p|.
    REFUTING OUTCOME: the observed ratio exceeds that bound.
    """
    ph = (np.diff(np.angle(A, deg=True), axis=1) + 180) % 360 - 180
    out = []
    for b in range(A.shape[0]):
        i = int(np.argmax(np.abs(ph[b])))
        j = int(BAND[i])
        p = int(BAND[int(np.argmax(np.abs(A[b])))])
        kk = np.linspace(j + 1e-6, j + 1 - 1e-6, 20001)
        bound = float(np.max(np.abs(np.sin(np.pi * (kk - p)) / (kk - p))
                             / np.abs(np.sin(np.pi * (kk - j)) / (kk - j))))
        obs = float(np.abs(A[b, p - BAND[0]]) / np.abs(A[b, j - BAND[0]]))
        out.append(dict(block=b, step_deg=float(ph[b, i]), step_pair=(j, j + 1),
                        peak_bin=p, ratio_observed=obs, ratio_max_single=bound,
                        excess=obs / bound))
    return out


# ---------------------------------------------------------------------------
# frequency of each fitted spatial component (NON-circular: no f_K pre-slice)
# ---------------------------------------------------------------------------

def project_and_estimate_frequency(m, kappa, dt, t0=T0_BLOCKS, n=376,
                                   f_lo=1.5e9, f_hi=4.5e9, N=NX):
    """Project the RAW m(x,t) onto the fitted spatial component, then estimate
    that projection's frequency (Hann window, parabolic sub-bin interpolation).
    The f_K slice is NEVER applied first, so the answer is not imposed.

    SIGN.  The time transform is `saw_analysis.time_spectrum`, i.e. the SAME
    kernel exp(+i 2 pi f t) that `saw_analysis.spectrum` uses, so the positive
    frequencies of this transform carry the +x-travelling branch that the
    spatial projection on exp(-i kappa x) selects.  Until 2026-09-18 this
    function performed its own `np.fft.fft` -- kernel exp(-i 2 pi f t) -- and
    then searched POSITIVE frequencies, which reads the COUNTER-PROPAGATING
    component: on a forward 3.0 GHz wave of amplitude 1 plus a backward 2.5 GHz
    wave of amplitude 0.01 it returned 2.4981 GHz.  Independent audit
    2026-09-18 section 4; pinned by
    `test_saw_analysis.py::test_freq_estimator_branch`.

    WHAT THE CORRECTED NUMBER MAY AND MAY NOT BE USED FOR.  The sign fix moves
    the sim40 eps_0 = 7e-5 bin-4 + bin-5 sum from 6.069845 GHz to 5.994198 GHz,
    i.e. from about +70 MHz above 2 f_K to about -5.8 MHz below it.  Neither
    number is evidence about the candidate mechanism.  The old +70 MHz was an
    artefact of this branch error and MUST NOT be cited as a physical mismatch;
    the new -5.8 MHz MUST NOT be cited as confirmation of a sum rule either,
    because this estimator has no error budget: its bias and spread under the
    actual record length, growth envelope, Hann window and parabolic
    interpolation have not been measured on synthetic controls carrying the
    same envelope and window.  The 133 MHz fft bin spacing is NOT that error
    bar -- it is neither an upper nor a lower bound on an interpolated
    estimator's accuracy.  Until such a control-based budget exists, report the
    frequency as a diagnostic only.

    `amp` is on `saw_analysis` scaling (the plain unnormalised sum, a factor n
    larger than the pre-2026-09-18 value); `f_peak`, `subbin_delta` and `snr`
    are unaffected by that scale.
    """
    x = np.arange(N)
    s = m[t0:t0 + n, :] @ (np.exp(-2j * np.pi * kappa * x / N) / N)
    f, S = sa.time_spectrum(s, dt, window="hann")
    idx = np.where((f > f_lo) & (f < f_hi))[0]
    j = idx[int(np.argmax(np.abs(S[idx])))]
    a0, a1, a2 = np.abs(S[j - 1]), np.abs(S[j]), np.abs(S[j + 1])
    den = np.log(a0) - 2 * np.log(a1) + np.log(a2)
    delta = 0.5 * (np.log(a0) - np.log(a2)) / den if den != 0 else 0.0
    df = f[1] - f[0]
    return dict(f_peak=float(f[j] + delta * df), df_bin=float(df),
                subbin_delta=float(delta), amp=float(a1),
                snr=float(a1 / np.median(np.abs(S[idx]))))


def unpumped_dispersion(bins):
    """f(k) from sim45 B = 50 mT: unpumped free ring-down, same 1024 x 5 nm grid,
    no SAW.  sim40 runs at B0 = 50.629 mT, so the k-DEPENDENCE is the reference,
    not the absolute frequency; the k=0 offset is reported so the shift is
    visible."""
    d = np.load(os.path.join(DATA, "sim45_checkpoints", "sim45_B50mT.npz"))
    m = d["my_xt"].astype(np.float64)
    dt = float(d["DT_REC"])
    nt = m.shape[0]
    F = np.fft.fft(np.fft.fft(m * np.hanning(nt)[:, None], axis=1), axis=0)
    f = np.fft.fftfreq(nt, d=dt)
    pos = np.where(f > 0.5e9)[0]
    out = {}
    for j in bins:
        col = np.abs(F[pos, int(j)])
        i = int(np.argmax(col))
        a0, a1, a2 = col[i - 1], col[i], col[i + 1]
        den = np.log(a0) - 2 * np.log(a1) + np.log(a2)
        dd = 0.5 * (np.log(a0) - np.log(a2)) / den if den != 0 else 0.0
        out[int(j)] = float(f[pos][i] + dd * (f[1] - f[0]))
    return out, float(f[1] - f[0])


# ---------------------------------------------------------------------------
# auxiliary, explicitly NON-discriminating diagnostics
# ---------------------------------------------------------------------------

def auxiliary(m, dx, dt):
    aux = {}
    for pad in (1, 8, 16):
        kp, fp, Mp = sa.spectrum(m, dx, dt, window="hann", trange=(375, 751),
                                 pad_x=pad)
        ip, slp = sa.f_slice(Mp, fp, F0)
        sel = (sa.to_inv_um(kp) > 2.0) & (sa.to_inv_um(kp) < 9.5)
        kk, pp = sa.to_inv_um(kp)[sel], np.abs(slp[sel])
        loc = [j for j in range(1, len(pp) - 1) if pp[j] > pp[j - 1] and pp[j] > pp[j + 1]]
        loc.sort(key=lambda j: -pp[j])
        aux["f_slice_Hz"] = float(fp[ip])
        aux["pad%d_npeaks" % pad] = len(loc)
        aux["pad%d_peaks" % pad] = [[float(kk[j]), float(pp[j] / pp[loc[0]])]
                                    for j in loc[:5]]
    X = np.fft.fft(m - m.mean(axis=1, keepdims=True), axis=1) / m.shape[1]
    t = np.arange(m.shape[0]) * dt
    win = slice(375, 751)
    ratio = X[win, 5] / X[win, 4]
    aux["arg_M5_over_M4_deg_mean"] = float(np.angle(ratio).mean() * 180 / np.pi)
    aux["arg_M5_over_M4_deg_std"] = float(np.angle(ratio).std() * 180 / np.pi)
    aux["abs_M5_over_M4_mean"] = float(np.abs(ratio).mean())
    aux["abs_M5_over_M4_cv"] = float(np.abs(ratio).std() / np.abs(ratio).mean())
    for j in range(1, 10):
        g, b, r2 = sa.growth_fit(t, np.abs(X[:, j]), t0=7.5e-9, t1=15e-9)
        aux["Gamma_bin%d" % j] = float(g)
        aux["Gamma_bin%d_r2" % j] = float(r2)
    q = 2 * np.pi * 2 * F0 / V_SAW
    aux["momentum"] = dict(
        k4_um=float(sa.to_inv_um(4 * DK)), k5_um=float(sa.to_inv_um(5 * DK)),
        sum_um=float(sa.to_inv_um(9 * DK)), q_um=float(sa.to_inv_um(q)),
        mismatch_um=float(sa.to_inv_um(9 * DK - q)),
        mismatch_pct=float(100 * (9 * DK - q) / q),
        mismatch_in_dk=float((9 * DK - q) / DK),
        dk_over_q_pct=float(100 * DK / q),
        q_over_dk=float(q / DK))
    return aux


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_fit(eps_tags=("5e-05", "7e-05", "1e-04")):
    res = {}
    for tag in eps_tags:
        r = load_run(tag)
        a_full, tstart = block_coeffs(r["m"], r["dt"])
        A, cpump = remove_pump(a_full, BAND)
        Ain, _ = remove_pump(a_full, INJ_BINS)
        Atr, Ahe = A[TRAIN_BLOCKS], A[HELD_BLOCKS]
        tot = float((np.abs(Atr) ** 2).sum())
        entry = dict(a_full=a_full, tstart=tstart, A=A, cpump=cpump,
                     train_power=tot, eps0=r["eps0"],
                     resid_floor=a_full[:, FLOOR_BINS],
                     phase=phase_flip_diagnostic(A),
                     m_rms=float(np.sqrt((r["m"][375:] ** 2).mean())),
                     m_max=float(np.abs(r["m"]).max()))
        for cfg in (CFG_PRE, CFG_AMD):
            nm = cfg["name"]
            pipe = run_pipeline(a_full, cfg)
            pipe2 = run_pipeline(a_full, cfg, CAL_SECOND, EVAL_SECOND)
            nested = {}
            for n in (1, 2, 3):
                mdl = fit_MN(Atr, BAND, n, cfg)
                e, _, _ = heldout_error(mdl["kappa"], Ahe, CAL_PRIMARY,
                                        EVAL_PRIMARY, cfg)
                nested[n] = dict(kappa=mdl["kappa"], ss=mdl["ss"],
                                 frac_explained=1 - mdl["ss"] / tot, E_held=e,
                                 n_real_params=mdl["n_real_params"],
                                 resid_bins=np.abs(Atr - mdl["fit"]).mean(0))
            entry[nm] = dict(pipe=pipe, pipe2=pipe2, nested=nested)
        # conservative residual level: after the AMENDED single-wave fit
        k1 = entry["amended"]["nested"][1]["kappa"]
        c1 = per_block_amps(A, k1, CFG_AMD)
        entry["resid_M1"] = Ain - (np.atleast_2d(dirichlet(k1, INJ_BINS)).T @ c1.T).T
        # descriptive on-grid (box-eigenmode) decomposition: bins are orthogonal
        p = (np.abs(A) ** 2).sum(0)
        o = np.argsort(p)[::-1]
        entry["ongrid"] = dict(order=(BAND[o]).tolist(),
                               cum_frac=np.cumsum(p[o] / p.sum()).tolist())
        res[tag] = entry
    return res


def run_controls(fitdata, cfgname="preregistered", nreal=200,
                 models=("R1", "R2", "R3"), noise_sets=("floor", "M1resid"),
                 cal=CAL_PRIMARY, evl=EVAL_PRIMARY, seed=20260917, n_grid=241):
    cfg = CFG_PRE if cfgname == "preregistered" else CFG_AMD
    A = fitdata["A"]
    cpump = fitdata["cpump"]
    k_d = fitdata["amended"]["nested"][1]["kappa"]      # the fitted single wave
    A1 = per_block_amps(A, k_d, cfg)
    resids = {"floor": fitdata["resid_floor"], "M1resid": fitdata["resid_M1"]}
    cases = {"exact_bins_4_5": np.array([4.0, 5.0]),
             "offgrid_sep1.0": np.array([4.30, 5.30]),
             "offgrid_sep0.5": np.array([4.55, 5.05]),
             "offgrid_sep1.5": np.array([4.05, 5.55])}
    rvals = [0.2, 0.5, 1.0, 1.2, 1.36]
    out = {}
    for nz in noise_sets:
        resid = resids[nz]
        for model in models:
            rng = np.random.default_rng(seed)
            sel, E1s, E2s, flips = [], [], [], []
            for _ in range(nreal):
                p = control_realisation(rng, k_d, A1, cpump, resid, model, cfg,
                                        cal, evl, n_grid=n_grid)
                sel.append(p["selected"]); E1s.append(p["E1"]); E2s.append(p["E2"])
                flips.append(phase_flip_diagnostic(p["A"])["n_flips_core"])
            sel = np.array(sel)
            out["d|%s|%s" % (nz, model)] = dict(
                fp_rate=float((sel == "two_components_required").mean()),
                single_rate=float((sel == "single_wave_sufficient").mean()),
                indist_rate=float((sel == "indistinguishable").mean()),
                E1_med=float(np.median(E1s)), E1_q95=float(np.quantile(E1s, .95)),
                E2_med=float(np.median(E2s)), E1=np.array(E1s), E2=np.array(E2s),
                core_flips_mean=float(np.mean(flips)), n=nreal)
            for cname, kaps in cases.items():
                for rr in rvals:
                    rng = np.random.default_rng(seed + 7717)
                    amps = np.zeros((N_BLOCKS, 2), dtype=complex)
                    amps[:, 0] = A1[:, 0]
                    amps[:, 1] = A1[:, 0] * rr * np.exp(1j * np.pi)
                    sel, flips = [], []
                    for _ in range(nreal):
                        p = control_realisation(rng, kaps, amps, cpump, resid,
                                                model, cfg, cal, evl, n_grid=n_grid)
                        sel.append(p["selected"])
                        flips.append(phase_flip_diagnostic(p["A"])["n_flips_core"])
                    sel = np.array(sel)
                    out["e|%s|%s|%s|%.2f" % (nz, model, cname, rr)] = dict(
                        det_rate=float((sel == "two_components_required").mean()),
                        single_rate=float((sel == "single_wave_sufficient").mean()),
                        indist_rate=float((sel == "indistinguishable").mean()),
                        core_flips_mean=float(np.mean(flips)), n=nreal)
    return out


def bootstrap_exceedance(pipe, controls, noise_sets=("floor", "M1resid"),
                         models=("R1", "R2", "R3")):
    """Exceedance UNDER THE ASSUMED RESIDUAL MODEL -- not an unconditional
    p-value.  Fraction of ONE-component realisations reaching an E1, or an
    E1/E2 ratio, at least as extreme as the measured one."""
    out = {}
    ratio_obs = pipe["E1"] / max(pipe["E2"], 1e-300)
    for nz in noise_sets:
        for model in models:
            c = controls["d|%s|%s" % (nz, model)]
            ratio = c["E1"] / np.maximum(c["E2"], 1e-300)
            out["%s|%s" % (nz, model)] = dict(
                p_E1_ge_obs=float((c["E1"] >= pipe["E1"]).mean()),
                p_ratio_ge_obs=float((ratio >= ratio_obs).mean()),
                E1_med=c["E1_med"], E1_q95=c["E1_q95"],
                ratio_med=float(np.median(ratio)),
                ratio_q95=float(np.quantile(ratio, 0.95)),
                ratio_obs=float(ratio_obs), E1_obs=float(pipe["E1"]), n=c["n"])
    return out



def adequacy_ratio(ss, n_complex, sigma2_floor):
    """In-band goodness of fit measured against the MEASURED out-of-band floor.
    For a correct model whose only residual is that floor, E[ss] = n_complex *
    sigma2_floor, so the ratio is ~1.  This is the SECONDARY criterion; it is
    not pre-registered and is calibrated on the same controls."""
    return float(ss / (n_complex * sigma2_floor))


def control_d_scan(fitdata, cfgname="preregistered", nreal=200,
                   kappas=(4.0, 4.3886, 4.5, 4.5622, 5.0001),
                   models=("R1", "R2", "R3"), noise_sets=("floor", "M1resid"),
                   cal=CAL_PRIMARY, evl=EVAL_PRIMARY, seed=424242, n_grid=241):
    """Control (d) over a SCAN of single-wave wavenumbers.  kappa = 4.0 and
    kappa = 5.0001 are essentially ON-grid (box-commensurate: no leakage tail);
    4.3886 = q/2dk, 4.5622 = the 16x zero-padded maximum, 4.5 = midpoint."""
    cfg = CFG_PRE if cfgname == "preregistered" else CFG_AMD
    A, cpump = fitdata["A"], fitdata["cpump"]
    resids = {"floor": fitdata["resid_floor"], "M1resid": fitdata["resid_M1"]}
    sig2 = {k: float((np.abs(v) ** 2).mean()) for k, v in resids.items()}
    out = {}
    for kd in kappas:
        amps = per_block_amps(A, np.array([kd]), cfg)
        for nz in noise_sets:
            for model in models:
                rng = np.random.default_rng(seed)
                sel, E1s, E2s, ad1, ad2, fl = [], [], [], [], [], []
                for _ in range(nreal):
                    p = control_realisation(rng, np.array([kd]), amps, cpump,
                                            resids[nz], model, cfg, cal, evl,
                                            n_grid=n_grid)
                    sel.append(p["selected"]); E1s.append(p["E1"]); E2s.append(p["E2"])
                    ad1.append(adequacy_ratio(p["models"][1]["ss"],
                                              len(TRAIN_BLOCKS) * len(BAND), sig2[nz]))
                    ad2.append(adequacy_ratio(p["models"][2]["ss"],
                                              len(TRAIN_BLOCKS) * len(BAND), sig2[nz]))
                    fl.append(phase_flip_diagnostic(p["A"])["n_flips_core"])
                sel = np.array(sel)
                out["%.4f|%s|%s" % (kd, nz, model)] = dict(
                    fp_rate=float((sel == "two_components_required").mean()),
                    single_rate=float((sel == "single_wave_sufficient").mean()),
                    indist_rate=float((sel == "indistinguishable").mean()),
                    E1_med=float(np.median(E1s)), E2_med=float(np.median(E2s)),
                    E1_q95=float(np.quantile(E1s, .95)),
                    adeq1_med=float(np.median(ad1)), adeq2_med=float(np.median(ad2)),
                    adeq1_q95=float(np.quantile(ad1, .95)),
                    core_flips_mean=float(np.mean(fl)), n=nreal)
    return out


def control_e_ongrid(fitdata, cfgname="preregistered", nreal=200,
                     models=("R3",), noise_sets=("floor",),
                     pairs=((4.0, 5.0), (4.0, 6.0), (3.0, 5.0)),
                     rvals=(0.2, 0.5, 1.0, 1.2, 1.36),
                     cal=CAL_PRIMARY, evl=EVAL_PRIMARY, seed=515151, n_grid=241):
    """Control (e) restricted to ON-GRID pairs, with the SECONDARY in-band
    adequacy criterion recorded alongside the pre-registered held-out rule."""
    cfg = CFG_PRE if cfgname == "preregistered" else CFG_AMD
    A, cpump = fitdata["A"], fitdata["cpump"]
    resids = {"floor": fitdata["resid_floor"], "M1resid": fitdata["resid_M1"]}
    sig2 = {k: float((np.abs(v) ** 2).mean()) for k, v in resids.items()}
    out = {}
    for pair in pairs:
        for rr in rvals:
            amps = np.zeros((N_BLOCKS, 2), dtype=complex)
            amps[:, 0] = per_block_amps(A, np.array([pair[0]]), cfg)[:, 0]
            amps[:, 1] = amps[:, 0] * rr * np.exp(1j * np.pi)
            for nz in noise_sets:
                for model in models:
                    rng = np.random.default_rng(seed)
                    sel, ad1, ad2 = [], [], []
                    for _ in range(nreal):
                        p = control_realisation(rng, np.array(pair), amps, cpump,
                                                resids[nz], model, cfg, cal, evl,
                                                n_grid=n_grid)
                        sel.append(p["selected"])
                        ad1.append(adequacy_ratio(p["models"][1]["ss"],
                                                  len(TRAIN_BLOCKS) * len(BAND), sig2[nz]))
                        ad2.append(adequacy_ratio(p["models"][2]["ss"],
                                                  len(TRAIN_BLOCKS) * len(BAND), sig2[nz]))
                    sel = np.array(sel)
                    out["%.0f+%.0f|%s|%s|%.2f" % (pair[0], pair[1], nz, model, rr)] = dict(
                        det_rate=float((sel == "two_components_required").mean()),
                        single_rate=float((sel == "single_wave_sufficient").mean()),
                        indist_rate=float((sel == "indistinguishable").mean()),
                        adeq1_med=float(np.median(ad1)), adeq2_med=float(np.median(ad2)),
                        n=nreal)
    return out


def stage_robust():
    r = load_run("7e-05")
    rows = []
    variants = [("preregistered", dict()),
                ("rect time window", dict(window="rect")),
                ("f0 = 2.9255 GHz (DFT bin)", dict(f0=2.9255319e9)),
                ("f0 = 3.0585 GHz (DFT bin)", dict(f0=3.0585106e9)),
                ("blocks of 62 samples", dict(blen=62)),
                ("blocks of 125 samples", dict(blen=125)),
                ("start t0 = 250 (5.0 ns)", dict(t0=250)),
                ("start t0 = 500 (10.0 ns), 62", dict(t0=500, blen=62))]
    for name, kw in variants:
        a, _ = block_coeffs(r["m"], r["dt"], **kw)
        row = dict(variant=name)
        for cfg in (CFG_PRE, CFG_AMD):
            p = run_pipeline(a, cfg)
            A, _ = remove_pump(a, BAND)
            tot = float((np.abs(A[TRAIN_BLOCKS]) ** 2).sum())
            row[cfg["name"]] = dict(
                kappa1=float(p["models"][1]["kappa"][0]),
                kappa2=p["models"][2]["kappa"].tolist(),
                frac_expl_M1=float(1 - p["models"][1]["ss"] / tot),
                frac_expl_M2=float(1 - p["models"][2]["ss"] / tot),
                E1=p["E1"], E2=p["E2"], selected=p["selected"])
            row["core_flips"] = phase_flip_diagnostic(A)["n_flips_core"]
        rows.append(row)
    return rows


def stage_figs(state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in"})
    f7 = state["fits"]["7e-05"]
    A = f7["A"]
    ku = sa.to_inv_um(BAND * DK)
    cfg = CFG_AMD
    n1 = f7["amended"]["nested"][1]
    n2 = f7["amended"]["nested"][2]
    Atr = A[TRAIN_BLOCKS]
    f1 = _varpro(n1["kappa"], Atr, BAND, cfg["rcond"], full=True)[1]
    f2 = _varpro(n2["kappa"], Atr, BAND, cfg["rcond"], full=True)[1]

    fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for b in range(N_BLOCKS):
        ax[0, 0].plot(ku, np.abs(A[b]), "o-", ms=3, lw=0.9,
                      label="block %d" % b)
    ax[0, 0].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    ax[0, 0].set_ylabel(r"$|a_j|$")
    ax[0, 0].set_yscale("log"); ax[0, 0].legend(frameon=False, fontsize=6)

    for b in range(N_BLOCKS):
        ax[0, 1].plot(ku, np.angle(A[b], deg=True), "o-", ms=3, lw=0.9)
    ax[0, 1].plot(ku, np.angle(dirichlet(n1["kappa"][0], BAND), deg=True),
                  "k--", lw=0.9, label="one off-grid wave")
    ax[0, 1].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    ax[0, 1].set_ylabel(r"$\arg a_j$ (deg)")
    ax[0, 1].legend(frameon=False, fontsize=6)

    ax[1, 0].plot(ku, np.abs(Atr).mean(0), "ko-", ms=3, lw=0.9, label="data")
    ax[1, 0].plot(ku, np.abs(f1).mean(0), "s--", ms=3, lw=0.9, label="M1 fit")
    ax[1, 0].plot(ku, np.abs(f2).mean(0), "^--", ms=3, lw=0.9, label="M2 fit")
    ax[1, 0].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    ax[1, 0].set_ylabel(r"$|a_j|$"); ax[1, 0].set_yscale("log")
    ax[1, 0].legend(frameon=False, fontsize=6)

    ax[1, 1].plot(ku, np.abs(Atr - f1).mean(0), "s-", ms=3, lw=0.9,
                  label="M1 residual")
    ax[1, 1].plot(ku, np.abs(Atr - f2).mean(0), "^-", ms=3, lw=0.9,
                  label="M2 residual")
    ax[1, 1].plot(ku, np.sqrt((np.abs(f7["resid_floor"]) ** 2).mean())
                  * np.ones_like(ku), "k:", lw=0.9, label="out-of-band floor")
    ax[1, 1].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    ax[1, 1].set_ylabel(r"$|a_j - \hat a_j|$"); ax[1, 1].set_yscale("log")
    ax[1, 1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_profile_and_residual.pdf"))
    plt.close(fig)

    # operating characteristic
    ctrl = state["controls"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    rvals = [0.2, 0.5, 1.0, 1.2, 1.36]
    for nz, mk in (("floor", "o-"), ("M1resid", "s--")):
        for cname in ("exact_bins_4_5", "offgrid_sep1.0", "offgrid_sep0.5"):
            y = [ctrl["e|%s|R3|%s|%.2f" % (nz, cname, rr)]["det_rate"] for rr in rvals]
            ax[0].plot(rvals, y, mk, ms=3, lw=0.9, label="%s, %s" % (nz, cname))
    ax[0].set_xlabel("second-component relative amplitude $r$")
    ax[0].set_ylabel("detection rate")
    ax[0].set_ylim(-0.05, 1.05); ax[0].legend(frameon=False, fontsize=5)
    if "power" in state:
        for pr in ("4+5", "4+6", "3+5"):
            y = [state["power"]["e_ongrid"]["%s|floor|R3|%.2f" % (pr, rr)]["det_rate"]
                 for rr in rvals]
            ax[0].plot(rvals, y, "^:", ms=3, lw=0.9, label="floor, on-grid %s" % pr)
        ax[0].legend(frameon=False, fontsize=5)
        ks = sorted({k.split("|")[0] for k in state["power"]["d_scan"]})
        xx = np.arange(len(ks))
        w = 0.26
        for i, mo in enumerate(("R1", "R2", "R3")):
            fp = [state["power"]["d_scan"]["%s|floor|%s" % (k, mo)]["fp_rate"] for k in ks]
            sr = [state["power"]["d_scan"]["%s|floor|%s" % (k, mo)]["single_rate"] for k in ks]
            ax[1].bar(xx + (i - 1) * w, sr, w * 0.9, color="0.72",
                      edgecolor="0.3", lw=0.4,
                      label="single-wave rate" if i == 0 else None)
            ax[1].plot(xx + (i - 1) * w, fp, "kx", ms=4, mew=0.9,
                       label="false-positive rate" if i == 0 else None)
        ax[1].set_xticks(xx)
        ax[1].set_xticklabels([r"$\kappa$=" + k for k in ks], rotation=30,
                              ha="right", fontsize=6)
        ax[1].set_xlabel("single-wave control wavenumber")
        ax[1].set_ylabel("rate")
        ax[1].set_ylim(-0.05, 1.08)
        ax[1].legend(frameon=False, fontsize=6, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_operating_characteristic.pdf"))
    plt.close(fig)
    return ["fig_profile_and_residual.pdf", "fig_operating_characteristic.pdf"]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        if np.iscomplexobj(o):
            return {"abs": np.abs(o).tolist(), "deg": np.angle(o, deg=True).tolist()}
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, complex):
        return {"abs": abs(o), "deg": float(np.angle(o, deg=True))}
    return o


def _save(ck, state):
    np.savez(ck, **{k: np.array(v, dtype=object) for k, v in state.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["all"])
    ap.add_argument("--nreal", type=int, default=200)
    args = ap.parse_args()
    do = lambda s: ("all" in args.stage) or (s in args.stage)     # noqa: E731

    ck = os.path.join(OUT, "mc_state.npz")
    state = {}
    if os.path.isfile(ck):
        z = np.load(ck, allow_pickle=True)
        state = {k: (z[k].item() if z[k].dtype == object and z[k].ndim == 0 else z[k])
                 for k in z.files}

    if do("fit") or "fits" not in state:
        print("[stage] fit", flush=True)
        state["fits"] = stage_fit(); _save(ck, state)
    f7 = state["fits"]["7e-05"]

    if do("controls") or "controls" not in state:
        print("[stage] controls nreal=%d" % args.nreal, flush=True)
        state["controls"] = run_controls(f7, nreal=args.nreal); _save(ck, state)
    if do("bootstrap") or "boot" not in state:
        print("[stage] bootstrap", flush=True)
        state["boot"] = bootstrap_exceedance(f7["preregistered"]["pipe"],
                                             state["controls"]); _save(ck, state)
    if do("freq") or "freq" not in state:
        print("[stage] freq", flush=True)
        r = load_run("7e-05")
        fr = {}
        for name, kaps in (("M1_pre", f7["preregistered"]["nested"][1]["kappa"]),
                           ("M1_amd", f7["amended"]["nested"][1]["kappa"]),
                           ("M2_amd", f7["amended"]["nested"][2]["kappa"]),
                           ("grid", np.array([2., 3., 4., 5., 6., 7.]))):
            fr[name] = [dict(kappa=float(k),
                             **project_and_estimate_frequency(r["m"], float(k), r["dt"]))
                        for k in np.atleast_1d(kaps)]
        disp, dfd = unpumped_dispersion([0, 2, 3, 4, 5, 6, 7, 9])
        fr["unpumped_dispersion_Hz"] = disp
        fr["unpumped_df_bin_Hz"] = dfd
        state["freq"] = fr; _save(ck, state)
    if do("aux") or "aux" not in state:
        print("[stage] aux", flush=True)
        r = load_run("7e-05")
        state["aux"] = auxiliary(r["m"], r["dx"], r["dt"]); _save(ck, state)
    if do("power") or "power" not in state:
        print("[stage] power", flush=True)
        state["power"] = dict(d_scan=control_d_scan(f7, nreal=args.nreal),
                              e_ongrid=control_e_ongrid(f7, nreal=args.nreal))
        _save(ck, state)
    if do("robust") or "robust" not in state:
        print("[stage] robust", flush=True)
        state["robust"] = stage_robust(); _save(ck, state)
    if do("figs"):
        print("[stage] figs", flush=True)
        state["figs"] = stage_figs(state); _save(ck, state)

    dump = {k: v for k, v in state.items() if k != "fits"}
    dump["fits_summary"] = {}
    for tag, f in state["fits"].items():
        d = dict(eps0=f["eps0"], train_power=f["train_power"],
                 m_rms=f["m_rms"], m_max=f["m_max"],
                 core_dphi_deg=f["phase"]["core_dphi"].tolist(),
                 n_flips_core=f["phase"]["n_flips_core"],
                 band_abs=np.abs(f["A"]).tolist(),
                 band_deg=np.angle(f["A"], deg=True).tolist(),
                 ongrid=f["ongrid"],
                 floor_rms=float(np.sqrt((np.abs(f["resid_floor"]) ** 2).mean())),
                 M1resid_rms=float(np.sqrt((np.abs(f["resid_M1"]) ** 2).mean())))
        for nm in ("preregistered", "amended"):
            e = f[nm]
            d[nm] = dict(E1=e["pipe"]["E1"], E2=e["pipe"]["E2"],
                         selected=e["pipe"]["selected"],
                         E1_sec=e["pipe2"]["E1"], E2_sec=e["pipe2"]["E2"],
                         selected_sec=e["pipe2"]["selected"],
                         nested={n: dict(kappa=e["nested"][n]["kappa"].tolist(),
                                         frac_explained=e["nested"][n]["frac_explained"],
                                         E_held=e["nested"][n]["E_held"],
                                         n_real_params=e["nested"][n]["n_real_params"],
                                         resid_bins=e["nested"][n]["resid_bins"].tolist())
                                 for n in e["nested"]})
        dump["fits_summary"][tag] = d
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(_jsonable(dump), fh, indent=1)
    print("[done] ->", OUT)


if __name__ == "__main__":
    main()
