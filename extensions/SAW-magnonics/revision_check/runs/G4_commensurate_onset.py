"""G4 -- STEP 2: commensurate reproduction of the onset.

Question this run answers, and the only one it answers:
    With the physical q and Omega unchanged but the box chosen so that
    q L = 2 pi n with n EVEN, does the finite-momentum feature survive, and
    where does its weight sit?

Why it has to exist.  In the sim40 box (NX=1024, dx=5 nm, L=5.12 um) the pump
is incommensurate: q L / 2 pi = 8.7771.  Discretising the pump's own strain
eps_xx = eps_0 sin(q x) on that grid puts 42.40 % of its power on bin +-9
(11.044662 um^-1), 3.42 % on bin +-8, and 15.2 % outside the top two bins
(revision_check/out/pump_bin_control.txt).  The surviving candidate pair, bins
4 and 5, sums to exactly that same bin 9.  So in that box "k_1 + k_2 = pump"
cannot be distinguished from "k_1 + k_2 = the pump's own aliased bin".  Making
the box commensurate removes the alias and puts q on bin n and q/2 on bin n/2.

Stages (each independently checkpointed; run all, or one by name):
    pump   engine-side verification that the SAW really sits on bin n
    disp   SAW off: measure omega(k) on the grid and tune B0 so that
           omega(q/2) = Omega/2 exactly, using the UNPUMPED dispersion as the
           reference (CONVENTIONS: a driven mode is NOT required to sit on it)
    onset  the strain sweep, two parameter sets, two rng seeds
    analyze  band-resolved Gamma(k), band_content, resolution report

Usage
    python G4_commensurate_onset.py                 # all stages
    python G4_commensurate_onset.py pump
    python G4_commensurate_onset.py onset --arm YIGlit
    python G4_commensurate_onset.py analyze
Resumes automatically from whatever checkpoints exist; delete a checkpoint to
force a rerun.  Nothing outside revision_check/ is written.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _conditions as _cond                                       # noqa: E402
import _gate                                                      # noqa: E402
import _harness as H                                              # noqa: E402
import growth_interval as GI                                      # noqa: E402
import saw_analysis as SA                                         # noqa: E402

BOX      = H.BOX_PRIMARY          # n=12, NX=1400, dx=5 nm, L=7.000000 um
F_SAW    = BOX["f_saw"]           # 6.000 GHz, unchanged from sim40
OMEGA    = 2 * np.pi * F_SAW
Q        = BOX["q"]               # 10.771175 um^-1, unchanged from sim40
K_HALF   = Q / 2                  # now EXACTLY bin 6 of this grid

CKPT   = os.path.join(H.OUT_DIR, "G4")
os.makedirs(CKPT, exist_ok=True)

DT_REC_RUN  = 20e-12              # 25 GHz Nyquist; same cadence as sim40/43-49
DT_REC_DISP = 5e-12               # 100 GHz Nyquist; exchange branch included
T_DISP      = 100e-9              # -> df_bin = 10.0 MHz (sim45 used 20 ns = 50 MHz
                                  #    and its delta column was quantised to one bin)
B0_START    = H.kittel_field(3.0e9, H.YIG_LIT["MS"])   # 50.629 mT, the sim40 value
B0_TOL      = 10e6                # Hz, one dispersion bin
B0_MAXIT    = 3

# --- k = 0 / Kittel diagnostic (audit 2026-09-18 sec.8a) --------------------
# The prescribed seed is band-limited with modes 1..floor(2q/dk); it contains NO
# k = 0 component, so the k = 0 column of a dispersion run is a numerical
# residual ~1e-15 of the main component and peak-picking on it is not a
# measurement.  The dispersion run therefore carries an EXPLICIT small uniform
# excitation, and the gate REQUIRES it to be visible before the k = 0 column is
# read at all.
K0_DIAG_AMP     = 1e-3      # uniform m_y canting added to the dispersion seed
                            # (10x the per-mode seed amplitude, 1e-3 of Ms, so
                            # the run stays linear; it is a diagnostic tone, not
                            # a physical claim)
K0_REL_MIN      = 1e-6      # k=0 column peak POWER, as a fraction of the
                            # strongest column: the residual the auditor
                            # measured is ~1e-15 and must fail this
K0_SNR_OFFBAND  = 100.0     # and >= 20 dB above the median out-of-band column
M_LINEAR        = 0.1       # max_x|m_y| above which the record is no longer a
                            # linear-growth measurement (same 10% level the
                            # runtime planner uses as m_sat_level)

# Two arms.  Arm A reproduces the published benchmark; arm B is the
# observability arm and is the one any manuscript claim must rest on.
ARMS = {
    "sim40set": dict(mat=H.SIM40,
                     eps=(3e-5, 5e-5, 7e-5, 1e-4),
                     note="reproduces sim40's parameters (|B1| 25x literature "
                          "YIG, K_mr on). Mechanism/box test ONLY."),
    "YIGlit":   dict(mat=H.YIG_LIT,
                     eps=(3e-5, 7e-5, 1e-4, 2e-4),
                     note="literature YIG. Headline evidence must come from "
                          "here (CLAUDE.md sec.4). The sweep BRACKETS the "
                          "predicted threshold eps_0 = 5.92e-5: 3e-5 is "
                          "predicted to DECAY (-9.58e6 1/s) and 7e-5 to grow "
                          "(+3.56e6 1/s). The old list started at 7e-5, i.e. "
                          "above the plan's own threshold, and could not "
                          "bracket a sign change (audit 2026-09-18 sec.8b). "
                          "3e-4 was dropped to pay for 3e-5; 1e-4 and 2e-4 are "
                          "the measured sim48 anchors and stay."),
}
# ---------------------------------------------------------------------------
# WHAT EACH STAGE REQUIRES OF ITS PREREQUISITE  (runs/_conditions.py)
#
# One mechanism, declared by name, compared field by field.  The old code
# compared the recorded gate STATE and nothing else, so a PASS pump certificate
# measured in the sim40 box (NX = 1024, pump on bin 9) authorised this NX = 1400
# bin-12 campaign.  These lists say which conditions each certificate must have
# been produced under; anything not listed is deliberately NOT required, because
# a gate must refuse what invalidates the run at hand and no more.
# ---------------------------------------------------------------------------
# G4.1 -- "the pump sits on bin n".  Where the pump's power lands is set by the
# grid and by the drive geometry, and by the build that evaluated the kernel.
# NOT by eps_0 (the criterion is a power FRACTION, linear in the amplitude) and
# not by the material (the analytic leg does not touch it).
NEED_PUMP_CERT = ("grid", "pump.f_saw", "pump.q", "pump.direction",
                  "pump.phase", "pump.xi", "pump.enable_mel",
                  "pump.enable_barnett", "build")
ALLOW_PUMP_CERT = ("pump.eps_0",)
CRITERIA_PUMP_CERT = dict(expected_bin=BOX["bin_q"], offbin_frac_max=1e-6,
                          engine_leg_required=True)
# G4.2 -- "B0* puts omega(q/2) at Omega/2".  The dispersion is measured with the
# SAW OFF, so B1/Kmr/eps_0/xi do not enter it; omega(k) does depend on the grid,
# on Msat/Aex/alpha, on the bath temperature and on the solver timestep.
# material.B0 is the RESULT and is therefore allowed to differ from the search's
# starting field.
NEED_B0_CERT = ("grid", "pump.f_saw", "pump.q", "material.Msat",
                "material.Aex", "material.alpha", "material.temperature",
                "numerics.dt_step", "numerics.dt_rec", "numerics.nt",
                "numerics.seed_kind", "numerics.seed_k_cut",
                "numerics.rng_seed", "build")
ALLOW_B0_CERT = ("material.B0",)
CRITERIA_B0_CERT = dict(B0_TOL=B0_TOL, B0_MAXIT=B0_MAXIT, T_DISP=T_DISP,
                        DT_REC_DISP=DT_REC_DISP, B0_START=B0_START,
                        K0_DIAG_AMP=K0_DIAG_AMP, K0_REL_MIN=K0_REL_MIN,
                        K0_SNR_OFFBAND=K0_SNR_OFFBAND)

RNG_SEEDS = (20260917, 20260918)   # explicit integers, recorded in every npz
K_CUT     = 2 * Q                  # seed band limit
V_TOT     = BOX["NX"] * H.NY * BOX["dx"] * H.CY * H.CZ


# --------------------------------------------------------------------------
def _prov(params):
    return H.prov_kw(__file__, params)


# ---------------------------------------------------------------- conditions --
def pump_conditions(eps0=1e-4, a_seed=1e-6):
    """Conditions the pump-bin check runs under (and is valid for).

    Built through the ONE builder in _harness, so the harness constants that
    build() applies -- NY, NZ, CY, CZ, PBC, XI, phase, direction, enable_mel,
    enable_barnett, DT_STEP -- are in it whether or not this stage mentions them.
    """
    return H.conditions(
        dict(box=BOX, material=H.SIM40, B0=B0_START, eps0=eps0, f_saw=F_SAW,
             temperature=0.0, rng_seed=RNG_SEEDS[0], a_seed=a_seed,
             k_cut=K_CUT),
        source="G4.1 pump-bin check", nt=8, dt_rec=1.0 / F_SAW / 8,
        block_records=8)


def consumer_conditions():
    """The conditions a later G4 stage will run under, so far as they are fixed
    by the box and the drive.  Material and strain vary per arm and per point and
    are therefore not part of what a certificate has to match."""
    return H.conditions(dict(box=BOX, f_saw=F_SAW),
                        source="G4 downstream consumer")


def _disp_conditions():
    """Conditions of the dispersion measurement, as it would be performed NOW."""
    nt = int(T_DISP / DT_REC_DISP) + 1
    return H.conditions(
        dict(box=BOX, material=H.YIG_LIT, B0=B0_START, eps0=0.0, f_saw=F_SAW,
             temperature=0.0, rng_seed=RNG_SEEDS[0], a_seed=1e-4,
             k_cut=K_CUT, saw="OFF"),
        source="G4.2 dispersion / B0* tuning",
        nt=nt, dt_rec=DT_REC_DISP, block_records=500)


DISP_VERIFIED = ("grid", "pump.f_saw", "pump.q", "material", "numerics", "build")


def _disp_stamp(**extra):
    return _disp_conditions().stamp(verified=DISP_VERIFIED,
                                    criteria=CRITERIA_B0_CERT,
                                    extra=extra or None)


def _disp_identity():
    """The dispersion certificate's condition stamp (kept under its old name).

    It is the SAME object the certificate carries -- not a second description of
    the same run -- so a field cannot be in one and missing from the other.  It
    now includes numerics.dt_step, the solver timestep that changes omega(k) and
    that the round-1 version omitted.
    """
    return json.loads(_disp_stamp()[_cond.KEY])


def _fpeak(f_axis, col, f_lo=0.5e9, f_hi=None):
    """Sub-bin peak frequency by parabolic interpolation of log|M| over f>0."""
    a = np.abs(col).astype(float)
    m = f_axis > f_lo
    if f_hi:
        m &= f_axis < f_hi
    idx = np.where(m)[0]
    j = idx[np.argmax(a[idx])]
    if j <= 0 or j >= a.size - 1 or a[j - 1] <= 0 or a[j + 1] <= 0:
        return float(f_axis[j]), float(a[j])
    y0, y1, y2 = np.log(a[j - 1]), np.log(a[j]), np.log(a[j + 1])
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    df = f_axis[1] - f_axis[0]
    return float(f_axis[j] + d * df), float(a[j])


# ------------------------------------------------------------ stage: pump ---
def stage_pump():
    """Is the pump on bin n?  Two independent checks.

    (1) CPU, analytic, always runs: FFT of eps_xx = eps_0 sin(q x) sampled at
        the cell centres the kernel uses, coord = (i + 0.5) dx.
    (2) engine: evaluate the chiral SAW effective field the GPU kernel actually
        produces and FFT that.  `chiral_saw_field` is exported at C++ module
        level (src/bindings/wrap_ferromagnet.cpp:223) but has no Python property
        in mumaxplus/ferromagnet.py, so it is reached through the same `_cpp`
        module that file uses.  If that fails on this build, it is recorded as
        NOT DETERMINABLE -- not guessed, and not silently skipped.

    REFUTES the box if: the top bin is not bin n, or more than 1e-6 of the
    positive-k power sits off bin n.  The same test run on the sim40 box is the
    positive control and MUST fail it (42.40 % on bin 9, 15.2 % scattered).

    TWO GATES, NOT ONE (corrected 2026-09-18, audit section 8, P0-1).  The two
    checks used to be summarised by one boolean computed from the ANALYTIC
    result alone, so with mumaxplus absent the stage returned PASS while the
    engine entry read "NOT DETERMINABLE".  They are now separate states:
        G4.1a  analytic, always evaluable
        G4.1b  engine; NOT_DETERMINABLE when the engine cannot be reached,
               which BLOCKS the campaign instead of passing it
    The record also says which boxes were measured ON THE ENGINE and which were
    computed analytically.  Only the primary commensurate box is put on the
    engine here; the NX = 1024 sim40 box (top bin 9, 0.8477 of the power) is an
    ANALYTIC control and its numbers must not be quoted as engine measurements.
    """
    boxes = {}
    for lbl, nx, dx in (("commensurate_n12", BOX["NX"], BOX["dx"]),
                        ("sim40_incommensurate", 1024, 5e-9)):
        x = (np.arange(nx) + 0.5) * dx
        S = np.abs(np.fft.rfft(np.sin(Q * x))) ** 2
        S = S / S.sum()
        dk = 2 * np.pi / (nx * dx)
        top = int(np.argmax(S))
        boxes[lbl] = dict(analytic=dict(
            method="analytic: rfft of sin(q x) at cell centres, power "
                   "normalised by the rfft total",
            NX=int(nx), dx=float(dx), top_bin=top, top_frac=float(S[top]),
            offbin_frac=float(1.0 - S[top]), dk=float(dk),
            k_top=float(top * dk)))
        print("  [analytic] %-22s top bin %d (k=%.6f um^-1) frac=%.6f  "
              "off-bin=%.3e" % (lbl, top, top * dk / 1e6, S[top], 1 - S[top]))

    # -- the engine check, attempted for the PRIMARY box only -----------------
    engine_target = "commensurate_n12"
    eng = {"status": "not attempted"}
    try:
        mag, meta = H.make_seed(RNG_SEEDS[0], BOX["NX"], BOX["dx"],
                                1e-6, K_CUT)
        world, magnet, saw_meta = H.build(BOX, H.SIM40, B0_START, 1e-4,
                                          F_SAW, mag,
                                          conditions=pump_conditions())
        eng["saw_meta"] = saw_meta
        from mumaxplus import _cpp                                # noqa: PLC0415
        from mumaxplus.fieldquantity import FieldQuantity         # noqa: PLC0415
        fq = FieldQuantity(_cpp.chiral_saw_field(magnet._impl))
        rows = []
        for _ in range(8):                    # 8 samples over ~1 SAW period
            hx = np.asarray(fq.eval())[2].mean(axis=(0, 1))   # H_z: the MR channel
            P = np.abs(np.fft.rfft(hx - hx.mean())) ** 2
            if P.sum() > 0:
                rows.append(P / P.sum())
            world.timesolver.run(1.0 / F_SAW / 8)
        P = np.mean(rows, axis=0)
        top = int(np.argmax(P))
        eng = dict(status="ok", method="engine: chiral_saw_field H_z, 8 phases "
                                       "over one SAW period",
                   top_bin=top, top_frac=float(P[top]),
                   offbin_frac=float(1 - P[top]),
                   k_top=float(top * BOX["dk"]), saw_meta=saw_meta)
        print("  [engine]   chiral_saw_field top bin %d (k=%.6f um^-1) "
              "frac=%.6f" % (top, top * BOX["dk"] / 1e6, P[top]))
    except Exception as exc:                                      # noqa: BLE001
        eng = {"status": "NOT DETERMINABLE (%s)" % exc}
        print("  [engine]   NOT DETERMINABLE: %s" % exc)
    boxes[engine_target]["engine"] = eng

    on_engine = [lbl for lbl, rec in boxes.items()
                 if rec.get("engine", {}).get("status") == "ok"]
    analytic_only = [lbl for lbl in boxes if lbl not in on_engine]

    # -- the two gates -------------------------------------------------------
    a = boxes["commensurate_n12"]["analytic"]
    g_analytic = _gate.Gate(
        "G4.1a_analytic_pump_bin",
        _gate.PASS if (a["top_bin"] == BOX["bin_q"] and
                       a["offbin_frac"] < 1e-6) else _gate.FAIL,
        "analytic top bin %d (expected %d), off-bin power %.3e (< 1e-6 "
        "required)" % (a["top_bin"], BOX["bin_q"], a["offbin_frac"]),
        detail=a)
    if eng.get("status") != "ok":
        g_engine = _gate.Gate(
            "G4.1b_engine_pump_bin", _gate.NOT_DETERMINABLE,
            "the engine-side check could not run (%s), so whether the GPU "
            "kernel puts the pump on bin %d is UNMEASURED here"
            % (eng.get("status"), BOX["bin_q"]), detail=eng)
    else:
        g_engine = _gate.Gate(
            "G4.1b_engine_pump_bin",
            _gate.PASS if (eng["top_bin"] == BOX["bin_q"] and
                           eng["offbin_frac"] < 1e-6) else _gate.FAIL,
            "engine top bin %d (expected %d), off-bin power %.3e"
            % (eng["top_bin"], BOX["bin_q"], eng["offbin_frac"]), detail=eng)
    gate = _gate.combine("G4.1_pump_on_bin", g_analytic, g_engine)

    H.stage_artifact(
        os.path.join(CKPT, "G4_pump_bin.npz"),
        pump_conditions(),
        verified=("grid", "pump", "build"),
        criteria=CRITERIA_PUMP_CERT,
        done=True,
             boxes=json.dumps(boxes, default=str),
             analytic=json.dumps({k: v["analytic"] for k, v in boxes.items()},
                                 default=str),
             engine=json.dumps(eng, default=str),
             engine_target=engine_target,
             measured_on_engine=json.dumps(on_engine),
             analytic_only=json.dumps(sorted(analytic_only)),
             analytic_state=g_analytic.state, engine_state=g_engine.state,
             gate_state=gate.state, gate=gate.to_json(),
             gate_pass=bool(gate.ok), expected_bin=BOX["bin_q"],
             **_prov(dict(box=BOX)))
    print("  measured on the engine : %s" % (on_engine or "NONE"))
    print("  analytic only          : %s" % sorted(analytic_only))
    print("  GATE G4.1a analytic %s | G4.1b engine %s"
          % (g_analytic.state, g_engine.state))
    print("  GATE G4.1 %s -- %s" % (gate.state, gate.reason))
    if gate.blocks:
        print("  CAMPAIGN BLOCKED at G4.1: no later stage may run.")
    return gate


# ------------------------------------------- k = 0 / Kittel diagnostic ------
def k0_signal_check(k_ax, f_ax, M, k_offband, f_lo=0.5e9,
                    rel_min=K0_REL_MIN, snr_offband=K0_SNR_OFFBAND,
                    uniform_initial_amplitude=None):
    """Is the k = 0 column a MEASUREMENT, or the residual of a seed that never
    excited k = 0?

    Audit 2026-09-18 sec.8: "the dispersion seed injects no k = 0 component, the
    residual there is ~1e-15 of the main component, and peak-picking on it is not
    a measurement".  Two stated criteria, both required:

      rel_power   peak power of the k = 0 column, as a fraction of the peak power
                  of the strongest column, >= rel_min (default 1e-6).  The
                  ~1e-15 residual the audit measured fails this by nine orders.
      snr_offband the same peak power >= snr_offband (default 100, i.e. 20 dB)
                  times the MEDIAN peak power of the out-of-band columns
                  |k| > k_offband.  If those columns are exactly zero (only a
                  synthetic record does that) the ratio is infinite and the
                  relative criterion is the binding one; that is recorded.

    Returns a dict with status "ok" or "NOT_EVALUABLE" -- never a frequency.
    The caller reads f(0) only when this says "ok".
    """
    P = np.abs(np.asarray(M)) ** 2
    fm = np.asarray(f_ax) > f_lo
    if not fm.any():
        return dict(status="NOT_EVALUABLE", rel_power=float("nan"),
                    snr_offband=float("nan"),
                    reason="no f > %.3g Hz in the record" % f_lo)
    colpk = P[fm].max(axis=0)
    j0 = int(np.argmin(np.abs(np.asarray(k_ax))))
    p0, pmax = float(colpk[j0]), float(colpk.max())
    off = np.abs(np.asarray(k_ax)) > k_offband
    floor = float(np.median(colpk[off])) if off.any() else 0.0
    rel = (p0 / pmax) if pmax > 0 else 0.0
    snr = (p0 / floor) if floor > 0 else float("inf")
    known_uniform = (uniform_initial_amplitude is not None
                     and np.isfinite(uniform_initial_amplitude)
                     and uniform_initial_amplitude > 0)
    ok = bool(rel >= rel_min and snr >= snr_offband and known_uniform)
    out = dict(status=("ok" if ok else "NOT_EVALUABLE"),
               k0_index=j0, k0_k=float(k_ax[j0]), p_k0=p0, p_max=pmax,
               rel_power=rel, offband_floor=floor,
               offband_floor_zero=bool(floor == 0.0), snr_offband=snr,
               uniform_initial_amplitude=uniform_initial_amplitude,
               criteria=dict(rel_min=float(rel_min),
                             snr_offband=float(snr_offband), f_lo=float(f_lo),
                             k_offband=float(k_offband)),
               reason="")
    if not known_uniform:
        out["reason"] = ("uniform diagnostic origin is not established; spectral "
                         "power/SNR alone cannot distinguish a uniform diagnostic "
                         "from finite-window leakage")
    elif not ok:
        out["reason"] = ("k = 0 column is %.3e of the strongest column "
                         "(criterion %.1e) and %.3e x the out-of-band floor "
                         "(criterion %.1e): there is no uniform excitation to "
                         "measure. Add the explicit uniform diagnostic "
                         "(K0_DIAG_AMP)." % (rel, rel_min, snr, snr_offband))
    return out


def disp_gate(resid_Hz, f_k0, f_kittel, df_bin, k0):
    """GATE G4.2, as three explicit sub-gates.

    The plan always said the k = 0 column had to agree with Kittel to within one
    bin, but the code decided the gate on the B0 residual ALONE, and the k = 0
    column it would have read carried no signal (audit sec.8a).  Now:

      G4.2a  the k = 0 column is a measurement at all  -> else NOT_EVALUABLE
      G4.2b  |f(0) - f_Kittel| <= one dispersion bin   -> else FAIL
      G4.2c  |f(q/2) - Omega/(4 pi)| < B0_TOL          -> else FAIL

    A NOT_EVALUABLE result BLOCKS the onset stage exactly as a FAIL does: there
    is no state in which a missing diagnostic passes.
    """
    ga = _gate.Gate("G4.2a_k0_signal",
                    _gate.PASS if k0.get("status") == "ok"
                    else _gate.NOT_EVALUABLE,
                    k0.get("reason") or
                    ("k = 0 column at %.3e of the strongest column, %.3e x the "
                     "out-of-band floor"
                     % (k0.get("rel_power", float("nan")),
                        k0.get("snr_offband", float("nan")))),
                    detail=dict(k0=k0))
    if ga.blocks:
        gb = _gate.Gate("G4.2b_kittel", _gate.NOT_EVALUABLE,
                        "f(0) cannot be compared with Kittel while the k = 0 "
                        "column carries no measurable signal")
    else:
        dk0 = abs(float(f_k0) - float(f_kittel))
        gb = _gate.Gate("G4.2b_kittel",
                        _gate.PASS if dk0 <= float(df_bin) else _gate.FAIL,
                        "|f(0) - f_Kittel| = %.3e Hz vs one bin %.3e Hz"
                        % (dk0, df_bin))
    gc = _gate.Gate("G4.2c_resid",
                    _gate.PASS if abs(float(resid_Hz)) < B0_TOL else _gate.FAIL,
                    "|f(q/2) - Omega/(4 pi)| = %.3e Hz vs tolerance %.3e Hz"
                    % (abs(float(resid_Hz)), B0_TOL))
    return _gate.combine("G4.2_B0_star", ga, gb, gc)


# ------------------------------------------------------------ stage: disp ---
def stage_disp():
    """Unpumped dispersion in THIS box, and the B0 that puts omega(q/2)=Omega/2.

    SAW off, so nothing here can be contaminated by the drive.  q/2 is bin n/2
    exactly, so omega(q/2) is read from one column with no interpolation in k.

    REFUTES the operating point if: after B0_MAXIT Newton steps the residual
    |f(q/2) - Omega/(4 pi)| still exceeds B0_TOL.  A dispersion run in which the
    k=0 column does NOT sit within one bin of the Kittel prediction refutes the
    whole setup and must stop the campaign at this stage.
    """
    nt = int(T_DISP / DT_REC_DISP) + 1
    B0 = B0_START
    hist = []
    for it in range(B0_MAXIT + 1):
        tag = "it%d_B%.4fmT" % (it, B0 * 1e3)
        path = os.path.join(CKPT, "G4_disp_%s.npz" % tag)
        mag, smeta = H.make_seed(RNG_SEEDS[0], BOX["NX"], BOX["dx"],
                                 1e-4, K_CUT)
        # EXPLICIT k = 0 diagnostic tone (audit sec.8a).  make_seed starts at
        # mode 1, so without this line there is nothing at k = 0 to measure and
        # the Kittel comparison reads a ~1e-15 residual.  inject_plane_wave at
        # k1 = 0 adds a uniform m_y canting and renormalises |m| = 1.
        mag = H.inject_plane_wave(mag, 0.0, K0_DIAG_AMP, 0.0, BOX["dx"])
        smeta = dict(smeta, k0_diag_amp=K0_DIAG_AMP,
                     k0_diag="uniform m_y canting, inject_plane_wave(k1=0)")
        extra = dict(B0=B0, it=it, seed_meta=json.dumps(smeta, default=str),
                     **_prov(dict(box=BOX, B0=B0, T_DISP=T_DISP,
                                  DT_REC=DT_REC_DISP, saw="OFF",
                                  seed=smeta)))
        br = H.BlockRun(path, nt, BOX["NX"], DT_REC_DISP,
                        manifest=dict(box=BOX, material=H.YIG_LIT, B0=B0,
                                      eps0=0.0, f_saw=F_SAW, temperature=0.0,
                                      stage="disp", T_DISP=T_DISP,
                                      seed=smeta))
        if not br.done:
            world, magnet, _ = H.build(BOX, H.YIG_LIT, B0, 0.0, F_SAW, mag,
                                       conditions=br.conditions)
            br.integrate(world, magnet, extra)
        my = br.my_xt

        k_ax, f_ax, M = SA.spectrum(my, BOX["dx"], DT_REC_DISP, window="hann")
        j0 = int(np.argmin(np.abs(k_ax - 0.0)))
        jh = int(np.argmin(np.abs(k_ax - K_HALF)))
        jq = int(np.argmin(np.abs(k_ax - Q)))
        f0, _ = _fpeak(f_ax, M[:, j0])
        fh, _ = _fpeak(f_ax, M[:, jh])
        fq_, _ = _fpeak(f_ax, M[:, jq])
        fK_th = H.kittel_freq(B0, H.YIG_LIT["MS"])
        resid = fh - F_SAW / 2
        k0chk = k0_signal_check(k_ax, f_ax, M, 2 * Q,
                                uniform_initial_amplitude=float(np.abs(mag[1].mean())))
        hist.append(dict(it=it, B0=B0, f_k0=f0, f_qhalf=fh, f_q=fq_,
                         kittel=fK_th, resid=resid, k0=k0chk,
                         k_qhalf=float(k_ax[jh]), snap=float(k_ax[jh] - K_HALF)))
        print("      k=0 diagnostic: %s (rel_power=%.3e, offband SNR=%.3e)"
              % (k0chk["status"], k0chk["rel_power"], k0chk["snr_offband"]))
        print("  it%d B0=%.4f mT  f(0)=%.4f GHz (Kittel %.4f)  f(q/2)=%.4f GHz "
              " resid=%+.1f MHz" % (it, B0 * 1e3, f0 / 1e9, fK_th / 1e9,
                                    fh / 1e9, resid / 1e6))
        if abs(resid) < B0_TOL or it == B0_MAXIT:
            break
        # Newton step on the Kittel relation (the k-dependent part of
        # omega(k) is nearly B0-independent at fixed k in this regime).
        Ms = H.YIG_LIT["MS"]
        dfdB = (H.GAMMA / (2 * np.pi)) * (2 * B0 + H.MU0 * Ms) / \
               (2 * np.sqrt(B0 * (B0 + H.MU0 * Ms)))
        B0 = B0 - resid / dfdB

    gate = disp_gate(resid_Hz=hist[-1]["resid"], f_k0=hist[-1]["f_k0"],
                     f_kittel=hist[-1]["kittel"], df_bin=1.0 / T_DISP,
                     k0=hist[-1]["k0"])
    gate.detail = dict(history=hist, newton_steps=len(hist) - 1)
    ok = gate.ok
    cert = os.path.join(CKPT, "G4_B0star.npz")
    stamp = _disp_stamp(newton_steps=len(hist) - 1)
    np.savez(cert, done=True,
             B0_star=hist[-1]["B0"], history=json.dumps(hist),
             gate_pass=ok, gate_state=gate.state, gate=gate.to_json(),
             run_identity_sha256=_disp_conditions().digest,
             tol_Hz=B0_TOL,
             df_bin=1.0 / T_DISP, **dict(_prov(dict(box=BOX)), **stamp))
    print("  B0* = %.4f mT   GATE G4.2 %s (|omega(q/2)-Omega/2| < %.0f MHz)"
          % (hist[-1]["B0"] * 1e3, gate.state, B0_TOL / 1e6))
    if gate.blocks:
        print("  CAMPAIGN BLOCKED at G4.2: the onset stage must not run.")
    return gate


def _require_stage(filename, gate_name, need=None, current=None, allow=(),
                   criteria=None, require_build_link=True):
    """Load a recorded gate and refuse anything that is not an explicit PASS
    **produced under the conditions this stage needs**.

    Round 1 (audit sec.8, P0-2) removed the fallback value.  Round 2 removes the
    remaining half of the defect: checking the recorded STATE and nothing else
    let a PASS certificate measured in the sim40 box (NX = 1024, pump on bin 9)
    authorise the NX = 1400 bin-12 campaign.  The second refusal is the shared
    mechanism in runs/_conditions.py -- not a bespoke comparison of one field --
    and `need` must be stated: a prerequisite whose required conditions nobody
    named cannot be checked, so that is refused too.

    Corrected 2026-09-18 (audit section 8, P0-2): a B0 file carrying
    gate_pass = False used to be accepted, a missing file fell back to the
    sim40 field, and the onset stage never looked at either.  There is now no
    fallback: a missing, unreadable, failed or NOT_DETERMINABLE prerequisite
    raises _gate.GateHalt naming the file and the reason.
    """
    p = os.path.join(CKPT, filename)
    if not os.path.isfile(p):
        raise _gate.GateHalt(
            "%s: prerequisite record %s is MISSING -- run that stage first. "
            "There is no fallback value." % (gate_name, filename))
    try:
        d = np.load(p, allow_pickle=True)
        _ = d.files                     # a truncated npz fails HERE, by name
    except Exception as exc:                                      # noqa: BLE001
        raise _gate.GateHalt("%s: prerequisite record %s is unreadable (%s)"
                             % (gate_name, filename, exc))
    if need is None or current is None:
        raise _gate.GateHalt(
            "%s: prerequisite %s was required without stating WHICH conditions "
            "it must have been produced under; refusing to accept it on its "
            "recorded state alone." % (gate_name, filename))
    state = _gate.state_from_record(d)
    if state != _gate.PASS:
        reason = ""
        if "gate" in d:
            try:
                reason = json.loads(str(d["gate"])).get("reason", "")
            except Exception:                                     # noqa: BLE001
                reason = ""
        raise _gate.GateHalt(
            "%s: prerequisite %s did not pass (state=%s%s). No downstream "
            "stage may use its output."
            % (gate_name, filename, state,
               "; reason: %s" % reason if reason else ""))
    # ... and the conditions it was produced under.  ONE comparison, the same
    # one every checkpoint and every other stage uses.
    _cond.require(gate_name, filename, d, current, need, allow=allow,
                  criteria=criteria, require_build_link=require_build_link)
    return d


def b0star_for_dependents():
    """B0* for OTHER stage scripts (G4b, G6, G9), with the same checks.

    They used to carry their own `if os.path.isfile(...) else Kittel fallback`,
    which is the P0-2 defect in three more places: a failed or absent
    dispersion stage silently became "the sim40 field".  There is one accessor
    now and it raises _gate.GateHalt instead of inventing a field.
    """
    return _B0star()


def _B0star():
    """B0* from the dispersion stage, or a halt.  Never a fallback.

    Round 2: the condition comparison moved into _require_stage, so there is one
    mechanism rather than a local _gate.diff here; and a PASSED, condition-
    matched record that does not actually carry B0_star halts with a NAMED state
    instead of raising a bare KeyError (open item 15).
    """
    d = _require_stage("G4_B0star.npz", "G4.2", need=NEED_B0_CERT,
                       current=_disp_conditions(), allow=ALLOW_B0_CERT,
                       criteria=CRITERIA_B0_CERT)
    if "B0_star" not in getattr(d, "files", []):
        raise _gate.GateHalt(
            "G4.2: NOT_DETERMINABLE -- G4_B0star.npz passed its gate and its "
            "conditions match, but it carries no B0_star field (keys: %s), so "
            "there is no field to adopt."
            % ", ".join(sorted(getattr(d, "files", []))[:12]))
    try:
        return float(d["B0_star"])
    except Exception as exc:                                      # noqa: BLE001
        raise _gate.GateHalt(
            "G4.2: NOT_DETERMINABLE -- G4_B0star.npz carries a B0_star that is "
            "not a number (%s)" % exc)


# ----------------------------------------------------------- stage: onset ---
def stage_onset(only_arm=None):
    # Both earlier stages must have PASSED, and B0* must have been measured for
    # THIS box.  Either check failing halts before any engine work.
    _require_stage("G4_pump_bin.npz", "G4.1", need=NEED_PUMP_CERT,
                   current=consumer_conditions(), allow=ALLOW_PUMP_CERT,
                   criteria=CRITERIA_PUMP_CERT)
    B0 = _B0star()
    for arm, cfg in ARMS.items():
        if only_arm and arm != only_arm:
            continue
        mat = cfg["mat"]
        print("\n---- arm %s : %s" % (arm, cfg["note"]))
        a_phys = H.thermal_mode_amplitude(K_HALF, 300.0, B0, mat["MS"],
                                          mat["AEX"], V_TOT)
        for eps in cfg["eps"]:
            g = H.gamma_estimate(mat, eps)
            win = H.auto_window(g)
            a_seed, why = H.choose_seed_amplitude(mat, eps, a_phys,
                                                  t_window=win)
            plan = H.plan_runtime(mat, eps, a_seed, t_window=win, t_min=40e-9)
            # The planner's status participates HERE, before any engine work:
            # INSUFFICIENT_LINEAR_WINDOW and NO_LINEAR_WINDOW are blocking, and
            # a blocked plan has no run length to read (H.RunPlan refuses it).
            if plan["status"] in H.PLAN_BLOCKING:
                raise _gate.GateHalt(_gate.Gate(
                    "G4.plan_%s_eps%.0e" % (arm, eps), _gate.FAIL,
                    "run plan blocked: %s -- %s" % (plan["status"],
                                                    plan["note"])))
            nt = H.nt_records(plan, DT_REC_RUN,
                              "G4 onset arm %s eps=%.3e" % (arm, eps))
            T_eq = H.seed_temperature(a_seed, K_HALF, B0, mat["MS"],
                                      mat["AEX"], V_TOT)
            for s in RNG_SEEDS:
                tag = "%s_eps%.0e_seed%d" % (arm, eps, s)
                path = os.path.join(CKPT, "G4_onset_%s.npz" % tag)
                mag, smeta = H.make_seed(s, BOX["NX"], BOX["dx"], a_seed, K_CUT)
                params = dict(box=BOX, arm=arm, material=mat, B0=B0, eps0=eps,
                              f_saw=F_SAW, T_run=plan["t_run"],
                              DT_REC=DT_REC_RUN, DT_STEP=H.DT_STEP,
                              rng_seed=s, seed=smeta, a_seed=a_seed,
                              a_phys_300K=a_phys, seed_equiv_temperature_K=T_eq,
                              seed_rule=why, gamma_expected=g,
                              freq_window=win, temperature=0.0)
                extra = dict(eps0=eps, B0=B0, arm=arm, rng_seed=s,
                             a_seed=a_seed, gamma_expected=g,
                             seed_equiv_temperature_K=T_eq,
                             seed_meta=json.dumps(smeta, default=str),
                             **_prov(params))
                br = H.BlockRun(path, nt, BOX["NX"], DT_REC_RUN,
                                manifest=params)
                if br.done:
                    print("  [skip] %s" % tag)
                    continue
                print("  ---- %s  T=%.0f ns  a_seed=%.3e (T_eq=%.3g K)  "
                      "G_exp=%.2e 1/s" % (tag, plan["t_run"] * 1e9, a_seed,
                                          T_eq, g))
                world, magnet, smeta2 = H.build(BOX, mat, B0, eps, F_SAW,
                                                mag,
                                                conditions=br.conditions)
                extra["saw_meta"] = json.dumps(smeta2, default=str)
                br.integrate(world, magnet, extra)


# --------------------------------------------------------- stage: analyze ---
def _amp_norm(nx, dx, dt, seg, stride, f0, k_ref):
    """Kept as a name for the tests that call it; the implementation is the one
    in growth_interval, so there is a single calibration as well as a single
    fitter."""
    return GI._amp_norm(nx, dx, dt, seg, stride, f0, k_ref)


def _gamma_map(my, dx, dt, f0, seg=None, m_linear=M_LINEAR, k_offband=None,
               snr_amp=4.0, min_efold=1.0, min_points=5):
    """Band-resolved Gamma(k) at f0 -- a thin wrapper on THE one fitter.

    Corrected 2026-09-18 (audit sec.8, P1-2) and REBUILT the same day after the
    correction was found to report post-turn-over damping as the growth rate.
    The interval rules, the signal criterion, the turn-over cut and every status
    now live in `growth_interval`, which states the measurand once
    (`GI.MEASURAND`) and is called by G4, G4b and G9 alike.  This function exists
    only so the G4 call sites and the existing regression tests keep their names;
    it must contain no fitting logic of its own, because a second implementation
    of the map is a second measurand, and that is precisely how the dead-bin
    defect survived in G4b and G9 after G4 was fixed.

    Returns (k_ax, G, R, STAT, t0s, jf, f_ax, meta); read STAT before G.
    """
    return GI.gamma_map(my, dx, dt, f0, seg=seg, m_linear=m_linear,
                        k_offband=(2 * Q) if k_offband is None else k_offband,
                        snr_amp=snr_amp, min_efold=min_efold,
                        min_points=min_points)


def onset_gate(rows, arm):
    """GATE G4.3 for one arm: a growing finite-k band AND a bracketed threshold.

    The plan asked for "Gamma > 0 with r^2 > 0.9 in at least one finite-k band,
    for both rng seeds, at a strain where the same quantity is negative at lower
    strain".

    Three things this gate got wrong before, all of them the same mistake -- it
    compared the evidence against ITSELF instead of against the DECLARED plan:

      * "for both rng seeds" was `len(seeds_grow) == len(seeds)` with `seeds`
        taken from the rows PRESENT, so one seed satisfied a clause that names
        two and a missing or unanalysed seed passed instead of blocking.  The
        comparison is now against `RNG_SEEDS` and `ARMS[arm]["eps"]`, the
        declared design, and absence is NOT_DETERMINABLE.
      * a point whose early growth interval could not be read entered the
        bracket as a NEGATIVE-growth point.  Points are now built with
        `GI.point_from_fit` and passed to `GI.bracket_from_fits`, which refuses
        any point that is not a MEASURED rate: "not measurable" is not "measured
        negative", and a bracket built on the latter is not a bracket.
      * an unfinished run could supply a gate point.  A row with done = False is
        excluded from the gate and named.
    """
    mine = [r for r in rows if r["arm"] == arm]
    if not mine:
        return _gate.Gate("G4.3_%s" % arm, _gate.NOT_DETERMINABLE,
                          "no analysed runs for this arm")
    subs = []

    uncertified = [r for r in mine if r.get("artifact_state") != _gate.PASS]
    if uncertified:
        return _gate.Gate("G4.3_inputs_%s" % arm, _gate.NOT_EVALUABLE,
                          "input conditions/build/completion are not verified for every row")

    # ---- partial runs are evidence of nothing: name them and drop them.
    partial = [r for r in mine if not bool(r.get("done", True))]
    if partial:
        subs.append(_gate.Gate(
            "G4.3_0_complete_%s" % arm, _gate.NOT_DETERMINABLE,
            "%d of %d runs are UNFINISHED (done = False) and are excluded from "
            "this gate: %s"
            % (len(partial), len(mine),
               ", ".join(sorted(str(r.get("file", "<row with no file>"))
                                for r in partial)))))
    mine = [r for r in mine if bool(r.get("done", True))]
    if not mine:
        return _gate.combine("G4.3_%s" % arm, *subs) if subs else _gate.Gate(
            "G4.3_%s" % arm, _gate.NOT_DETERMINABLE, "no finished runs")

    # ---- coverage against the DECLARED design, not against what is present.
    want_seeds = sorted(int(x) for x in RNG_SEEDS)
    want_eps = sorted(float(x) for x in ARMS[arm]["eps"])
    have = {(int(r["rng_seed"]), float(r["eps0"])) for r in mine}
    missing = [(s_, e_) for s_ in want_seeds for e_ in want_eps
               if (s_, e_) not in have]
    subs.append(_gate.Gate(
        "G4.3_1_coverage_%s" % arm,
        _gate.PASS if not missing else _gate.NOT_DETERMINABLE,
        ("all %d declared (seed, eps_0) configurations are analysed"
         % (len(want_seeds) * len(want_eps))) if not missing else
        ("%d of %d declared configurations are MISSING, so the arm's claims "
         "cannot be evaluated: %s (declared seeds %s, declared strains %s)"
         % (len(missing), len(want_seeds) * len(want_eps),
            ", ".join("seed %d eps=%.3e" % m for m in missing[:8]),
            want_seeds, ["%.3e" % e for e in want_eps]))))

    grow = [r for r in mine
            if r["gamma_qhalf_status"] == GI.OK and r["gamma_qhalf"] > 0
            and r["r2_qhalf"] > 0.9]
    seeds_grow = sorted({int(r["rng_seed"]) for r in grow})
    subs.append(_gate.Gate(
        "G4.3a_growth_%s" % arm,
        _gate.PASS if seeds_grow == want_seeds else _gate.FAIL,
        "Gamma(q/2) > 0 with r2 > 0.9 for seeds %s of the %s DECLARED in "
        "RNG_SEEDS" % (seeds_grow, want_seeds)))
    for s_ in want_seeds:
        pts_rows = sorted([r for r in mine if int(r["rng_seed"]) == s_],
                          key=lambda r: r["eps0"])
        if not pts_rows:
            subs.append(_gate.Gate(
                "G4.3b_bracket_%s_seed%s" % (arm, s_), _gate.NOT_DETERMINABLE,
                "seed %d is declared in RNG_SEEDS but has no analysed run: an "
                "absent seed cannot bracket a threshold" % s_))
            continue
        pts = [GI.point_from_fit(
            r["eps0"], dict(status=r["gamma_qhalf_status"],
                            gamma=r["gamma_qhalf"], r2=r["r2_qhalf"],
                            branch=r.get("gamma_qhalf_branch", ""),
                            measurable=r["gamma_qhalf_status"] == GI.OK,
                            reason=r.get("gamma_qhalf_reason", "")),
            note=str(r.get("file", ""))) for r in pts_rows]
        br = GI.bracket_from_fits(pts)
        st = {GI.OK: _gate.PASS, "NOT_BRACKETED": _gate.FAIL,
              "NOT_EVALUABLE": _gate.NOT_EVALUABLE}[br["status"]]
        subs.append(_gate.Gate(
            "G4.3b_bracket_%s_seed%s" % (arm, s_), st,
            br["reason"] or ("sign change of Gamma(q/2) bracketed in "
                             "eps_0 = [%.3e, %.3e]"
                             % tuple(br["eps_threshold_interval"])),
            detail=br))
    return _gate.combine("G4.3_%s" % arm, *subs)


def stage_analyze():
    rows = []
    for fn in sorted(os.listdir(CKPT)):
        if not fn.startswith("G4_onset_") or not fn.endswith(".npz"):
            continue
        d = np.load(os.path.join(CKPT, fn), allow_pickle=True)
        done = bool(d["done"]) if "done" in d else False
        if not done:
            print("  [partial] %s -- analysed anyway, flagged on the row "
                  "(done = False) and EXCLUDED from gate G4.3" % fn)
        my = d["my_xt"].astype(float)
        eps = float(d["eps0"])
        arm = str(d["arm"])
        mat = H.SIM40 if arm == "sim40set" else H.YIG_LIT
        input_gate = H.analysis_input_gate(d, BOX, mat, F_SAW, DT_REC_RUN)
        f_half = F_SAW / 2
        k_ax, G, R, STAT, t0s, jf, f_ax, gmeta = _gamma_map(
            my, BOX["dx"], DT_REC_RUN, f_half)

        # second half, f = Omega/2 slice, disjoint band bookkeeping
        _, f2, M2 = SA.spectrum(my, BOX["dx"], DT_REC_RUN, window="hann",
                                trange=(my.shape[0] // 2, my.shape[0]))
        _, sl = SA.f_slice(M2, f2, f_half)
        bc = SA.band_content(k_ax, sl, K_HALF, Q, halfwidth=BOX["dk"] / 2,
                             quantity="power")
        best = gmeta["best"]
        jbest, g_best = int(best["j"]), float(best["gamma"])
        k_best, best_state = float(best["k_um"]), str(best["state"])
        gmax = g_best if best_state == GI.OK else None
        res = SA.resolution_report(my.shape[0] // 2, DT_REC_RUN, "hann",
                                   growth_rate=gmax)
        jh = int(np.argmin(np.abs(k_ax - K_HALF)))
        jq = int(np.argmin(np.abs(k_ax - Q)))
        j0 = int(np.argmin(np.abs(k_ax)))
        rows.append(dict(
            file=fn, arm=arm, eps0=eps, n_rec=int(my.shape[0]),
            done=done, partial=(not done),
            artifact_state=input_gate.state, artifact_reason=input_gate.reason,
            rng_seed=int(d["rng_seed"]) if "rng_seed" in d else -1,
            gamma_qhalf=float(G[jh]), r2_qhalf=float(R[jh]),
            gamma_qhalf_status=str(STAT[jh]),
            gamma_qhalf_branch=str(gmeta["branch"][jh]),
            gamma_qhalf_reason=str(gmeta["held_reasons"][jh])[:400],
            gamma_qhalf_late=gmeta["gamma_late"][jh],
            gamma_qhalf_late_diagnostic=gmeta["late"][jh],
            growth_segment_duration_s=gmeta["segment_duration"],
            growth_segment_df_Hz=gmeta["segment_df"],
            gamma_qhalf_stderr=float(gmeta["gamma_se"][jh]),
            gamma_q=float(G[jq]), gamma_q_status=str(STAT[jq]),
            gamma_k0=float(G[j0]), gamma_k0_status=str(STAT[j0]),
            gamma_max_kpos=g_best, gamma_max_kpos_status=best_state,
            k_argmax_gamma_um=k_best,
            n_bins_evaluable=int(gmeta["n_bins_ok"]),
            fit_floor_amplitude=gmeta["floor_amplitude"],
            n_segments_linear=gmeta["n_segments_linear"],
            max_my_last=gmeta["max_my_last"],
            frac_plus=float(bc["fractions"]["plus"]),
            frac_minus=float(bc["fractions"]["minus"]),
            frac_k0=float(bc["fractions"]["k0"]),
            frac_saw_plus=float(bc["fractions"]["saw_plus"]),
            frac_rest=float(bc["fractions"]["rest"]),
            df_bin_MHz=res["df_bin"] / 1e6,
            eff_width_MHz=res.get("effective_width_Hz", np.nan) / 1e6))
        print("  %-42s G(q/2)=%+.3e [%s] (r2 %.3f)  argmax_k=%s  plus=%.4f"
              % (fn, G[jh], STAT[jh], R[jh],
                 ("%+.3f um^-1" % k_best) if best_state == GI.OK
                 else best_state, bc["fractions"]["plus"]))
        if STAT[jh] != GI.OK:
            print("      q/2 HELD: %s" % gmeta["held_reasons"][jh])
    if rows:
        gates = [onset_gate(rows, a) for a in sorted({r["arm"] for r in rows})]
        for g in gates:
            print("  GATE %s %s -- %s" % (g.name, g.state, g.reason))
        np.savez(os.path.join(CKPT, "G4_summary.npz"),
                 rows=json.dumps(rows), box=json.dumps(BOX),
                 measurand=GI.MEASURAND,
                 gates=json.dumps([g.as_dict() for g in gates], default=str),
                 gate_state=_gate.combine("G4.3", *gates).state,
                 **_prov(dict(stage="analyze")))
        with open(os.path.join(CKPT, "G4_summary.txt"), "w") as fh:
            fh.write(json.dumps(rows, indent=1))
        print("  wrote G4_summary.npz / .txt  (%d runs)" % len(rows))
    else:
        print("  nothing to analyse")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "pump", "disp", "onset", "analyze"])
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    a = ap.parse_args()
    print("=" * 78)
    print("G4 commensurate onset -- box %s  L=%.6f um  dk=%.7f um^-1"
          % (BOX["tag"], BOX["L"] * 1e6, BOX["dk"] / 1e6))
    print("  q = %.6f um^-1 = bin %d exactly ; q/2 = %.6f um^-1 = bin %d"
          % (Q / 1e6, BOX["bin_q"], K_HALF / 1e6, BOX["bin_qhalf"]))
    print("=" * 78)
    # Stage sequencing.  EVERY stage's gate must PASS, whether it was reached
    # through 'all' or invoked on its own: a per-stage invocation exists to
    # produce a usable certificate, and a blocked one is not usable.  The exit
    # code comes from H.run_entry, which is the same handler in every script
    # (open item 10: this path used to exit 0 after printing CAMPAIGN BLOCKED).
    def sequence():
        if a.stage in ("all", "pump"):
            stage_pump().require()
        if a.stage in ("all", "disp"):
            stage_disp().require()
        if a.stage in ("all", "onset"):
            stage_onset(a.arm)
        if a.stage in ("all", "analyze"):
            return stage_analyze()
        return None

    sys.exit(H.run_entry("G4." + a.stage, sequence))
