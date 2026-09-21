"""Matched controls for the recovered finite-q parametric mechanism.

Existing engine sources, old trajectories, and formal G4/G9 verdicts are not
modified. The uniform control is an external magnetic field, not a q=0 SAW.
All production traces retain the audited BlockRun identity/resume mechanism.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

import direct_check as DC
import spatial_linear_check as SL
from check_dispersion_reference import checkpoint, resume, sha
from compare_trace_growth import estimate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "controls"
H = SL.H
sys.path.insert(0, str(ROOT.parent))
import saw_analysis as SA

DT = 20e-12
DT_STEP = 4e-13
T_RUN = 40e-9
EPS = 2e-4
SEED = 20260917
A_SEED = 1e-4
FREQ = 6e9
K_SHAPE = H.MU0 * H.YIG_LIT["MS"]**2 / 2
CASES = (
    dict(name="reverse_mel", kind="saw", direction=-1, mel=True, kmr=0., reflected=True),
    dict(name="uniform_field", kind="uniform_field", direction=1, mel=False, kmr=0., reflected=False),
    dict(name="mr_only", kind="saw", direction=1, mel=False, kmr=K_SHAPE, reflected=False),
    dict(name="mel_and_mr", kind="saw", direction=1, mel=True, kmr=K_SHAPE, reflected=False),
)


def write(name, data):
    OUT.mkdir(exist_ok=True)
    dest = OUT / name
    temp = dest.with_suffix(dest.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temp, dest)


def setup():
    report, _ = DC.setup()
    H.DT_STEP = DT_STEP
    H.SEED_KIND = "bandlimited_yflat_randphase"
    H.SAW_PHASE = 0.
    if H.ENABLE_BARNETT:
        raise RuntimeError("This suite does not exercise the unresolved Barnett path")
    completed = json.loads((ROOT / "direct/analysis.json").read_text(encoding="utf-8"))
    if completed["source_sha256"] != sha(ROOT / "direct_check.py"):
        raise RuntimeError("The baseline producer changed")
    for row in completed["rows"]:
        path = ROOT / "direct" / ("MEL_eps%.0e_seed%d.npz" % (row["strain"], row["seed"]))
        if sha(path) != row["file_sha256"]:
            raise RuntimeError("The baseline data changed")
    return report


def plan(report):
    result = dict(source_sha256=sha(__file__), reference_sha256=sha(ROOT / "spatial_linear.json"),
                  baseline_analysis_sha256=sha(ROOT / "direct/analysis.json"),
                  duration_s=T_RUN, dt_record=DT, dt_step=DT_STEP, strain=EPS,
                  B0=report["Bstar"], material=H.YIG_LIT, cases=CASES,
                  seed=SEED, seed_amplitude=A_SEED, box=H.BOX_PRIMARY, pbc=report["pbc"],
                  uniform_amplitude_T=-2*H.YIG_LIT["B1"]/H.YIG_LIT["MS"]*EPS,
                  mr_coefficient=dict(value_J_m3=K_SHAPE, derivation="mu0 Ms^2 / 2; Ku=0 in this isotropic-film model"),
                  growth_window_ns=[10, 40], growth_segment_ns=2, max_linear_my=.05,
                  preflight_tolerance=dict(field_relative=5e-4, covariance_relative=5e-4,
                                           resume_absolute=1e-7),
                  interpretation="Matched mechanism controls, not an ensemble-coherence, threshold or detector certificate. The reflected seed tests covariance, not independent-seed robustness.")
    path = OUT / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(result)):
        raise RuntimeError("Control plan changed; do not adopt old checkpoints")
    write("plan.json", result)
    return result


def initial_state(case):
    initial, _ = H.make_seed(SEED, H.BOX_PRIMARY["NX"], H.BOX_PRIMARY["dx"],
                             A_SEED, 2*H.BOX_PRIMARY["q"])
    if case["reflected"]:
        initial = initial[..., ::-1].copy()
        initial[1:] *= -1
    return initial


def declaration(case, report, duration):
    mat = dict(H.YIG_LIT, KMR=case["kmr"])
    return dict(box=H.BOX_PRIMARY, material=mat, B0=report["Bstar"], eps0=EPS,
                f_saw=FREQ, q=0. if case["kind"] == "uniform_field" else H.BOX_PRIMARY["q"],
                saw_direction=case["direction"], enable_mel=case["mel"],
                temperature=0., rng_seed=SEED, a_seed=A_SEED, k_cut=2*H.BOX_PRIMARY["q"],
                seed_kind=H.SEED_KIND, initial_transform="axial_x_reflection" if case["reflected"] else "none",
                T_run=duration, DT_REC=DT, DT_STEP=DT_STEP, pump_kind=case["kind"],
                uniform_amplitude_T=(-2*mat["B1"]/mat["MS"]*EPS if case["kind"] == "uniform_field" else 0.),
                source_sha256=sha(__file__), reference_sha256=sha(ROOT / "spatial_linear.json"),
                mr_basis="Ku=0; Kmr=mu0 Ms^2/2 when MR is enabled",
                purpose="matched deterministic mechanism control, not thermal detectability")


def build(case, report, decl, cs):
    cs.require_execution("matched control", nx=H.BOX_PRIMARY["NX"])
    initial = initial_state(case)
    if case["kind"] == "uniform_field":
        # Allocate an explicitly unpumped world, then apply and verify the
        # separately declared uniform field before any time integration.
        base = dict(decl, eps0=0., q=H.BOX_PRIMARY["q"], pump_kind="none")
        base_cs = H.conditions(base, source="uniform-control unpumped allocation",
                               nt=round(decl["T_run"]/DT)+1, dt_rec=DT, block_records=100)
        world, magnet, meta = H.build(H.BOX_PRIMARY, decl["material"], decl["B0"], 0.,
                                      FREQ, initial, enable_mel=False, conditions=base_cs)
        b = decl["uniform_amplitude_T"]
        magnet.bias_magnetic_field.add_time_term(
            lambda t: (b*np.sin(-2*np.pi*FREQ*t), 0., 0.))
        meta.update(kind="uniform_external_field", amplitude_T=b, q=0.)
    else:
        world, magnet, meta = H.build(H.BOX_PRIMARY, decl["material"], decl["B0"], EPS,
                                      FREQ, initial, enable_mel=case["mel"],
                                      saw_direction=case["direction"], conditions=cs)
    verify_applied(case, decl, world, magnet)
    return world, magnet, meta


def verify_applied(case, decl, world, magnet):
    if not np.isclose(float(world.timesolver.timestep), DT_STEP, rtol=2e-6):
        raise RuntimeError("Applied timestep differs from the declaration")
    if bool(world.timesolver.adaptive_timestep):
        raise RuntimeError("Adaptive timesteps are not part of this declaration")
    if case["kind"] == "uniform_field":
        if bool(magnet.enable_saw) or decl["q"] != 0.:
            raise RuntimeError("Uniform reference must have SAW off and declared q=0")
    else:
        expected = dict(saw_frequency=2*np.pi*FREQ, saw_wavevector=case["direction"]*decl["q"],
                        saw_amplitude=EPS, saw_ellipticity=H.XI, saw_phase=0., saw_direction=0.)
        if not bool(magnet.enable_saw) or bool(magnet.saw_enable_barnett):
            raise RuntimeError("SAW/Barnett enable state differs from the declaration")
        if bool(magnet.saw_enable_mel) != case["mel"]:
            raise RuntimeError("MEL enable state differs from the declaration")
        if any(not np.isclose(float(getattr(magnet, k)), v, rtol=2e-6, atol=1e-12)
               for k, v in expected.items()):
            raise RuntimeError("Applied SAW wavevector/frequency/phase differs from the declaration")
        if not np.allclose(magnet.Kmr.eval(), case["kmr"], rtol=2e-6, atol=1e-12):
            raise RuntimeError("Applied MR coefficient differs from the declaration")
    if not np.allclose(magnet.msat.eval(), decl["material"]["MS"], rtol=2e-6):
        raise RuntimeError("Applied material differs from the declaration")


def field_check(case, decl, world, magnet):
    from mumaxplus import _cpp
    from mumaxplus.fieldquantity import FieldQuantity

    m = magnet.magnetization.eval().astype(float)
    x = (np.arange(H.BOX_PRIMARY["NX"])+.5)*H.BOX_PRIMARY["dx"]
    errors = []
    for t in (0., .037e-9, .117e-9):
        world.timesolver.time = t
        if case["kind"] == "uniform_field":
            actual = magnet.bias_magnetic_field.eval().astype(float)
            expected = np.zeros_like(actual)
            expected[0] = decl["B0"]+decl["uniform_amplitude_T"]*np.sin(-2*np.pi*FREQ*t)
            scale = decl["uniform_amplitude_T"]*np.sqrt(expected[0].size)
        else:
            phase = case["direction"]*decl["q"]*x-2*np.pi*FREQ*t
            expected = np.zeros_like(m)
            if case["mel"]:
                expected[0] += -2*decl["material"]["B1"]/decl["material"]["MS"]*EPS*np.sin(phase)*m[0]
            if case["kmr"]:
                rot = case["direction"]*H.XI*EPS/2*np.cos(phase)
                expected[0] += case["kmr"]/decl["material"]["MS"]*rot*m[2]
                expected[2] += case["kmr"]/decl["material"]["MS"]*rot*m[0]
            actual = FieldQuantity(_cpp.chiral_saw_field(magnet._impl)).eval().astype(float)
            scale = np.linalg.norm(expected)
        errors.append(float(np.linalg.norm(actual-expected)/scale))
    world.timesolver.time = 0.
    if max(errors) > 5e-4:
        raise RuntimeError("Actual pump field failed its SI analytic check: %r" % errors)
    return errors


def preflight():
    report = setup()
    p = plan(report)
    checks = []
    for case in CASES:
        decl = declaration(case, report, T_RUN)
        cs = H.conditions(decl, source="control preflight", nt=2001, dt_rec=DT, block_records=100)
        world, magnet, _ = build(case, report, decl, cs)
        checks.append(dict(case=case["name"], field_relative_errors=field_check(case, decl, world, magnet),
                           conditions=cs.stamp()))
        if case["name"] == "reverse_mel":
            reverse_torque = magnet.llg_torque.eval().astype(float)
            forward = dict(case, direction=1, reflected=False)
            fd = declaration(forward, report, T_RUN)
            fc = H.conditions(fd, source="forward covariance check", nt=2001, dt_rec=DT, block_records=100)
            fw, fm, _ = build(forward, report, fd, fc)
            expected = fm.llg_torque.eval().astype(float)[..., ::-1].copy()
            expected[1:] *= -1
            covariance = float(np.linalg.norm(reverse_torque-expected)/np.linalg.norm(expected))
            if covariance > 5e-4:
                raise RuntimeError("x-reflection covariance failed: %g" % covariance)
            checks[-1]["torque_covariance_relative_error"] = covariance
            magnet.saw_wavevector = abs(float(magnet.saw_wavevector))
            try:
                verify_applied(case, decl, world, magnet)
            except RuntimeError:
                checks[-1]["wrong_sign_refused"] = True
            else:
                raise AssertionError("Wrong applied pump sign was accepted")
        print("checked", case["name"], flush=True)
    uniform = next(c for c in CASES if c["kind"] == "uniform_field")
    resume_error = resume_check(uniform, report)
    write("preflight.json", dict(status="PASS", source_sha256=sha(__file__),
                                 plan_sha256=sha(OUT/"plan.json"), checks=checks,
                                 uniform_resume_max_absolute_error=resume_error))
    print("preflight PASS, uniform resume error", resume_error, flush=True)


def resume_check(case, report):
    duration = .4e-9
    decl = declaration(case, report, duration)
    nt = round(duration/DT)+1
    uninterrupted = H.BlockRun(str(OUT/"smoke_uniform_continuous.npz"), nt,
                                H.BOX_PRIMARY["NX"], DT, block_records=100, manifest=decl)
    w, m, meta = build(case, report, decl, uninterrupted.conditions)
    uninterrupted.integrate(w, m, dict(saw_meta=json.dumps(meta), **H.prov_kw(__file__, decl)))
    split_path = OUT/"smoke_uniform_resumed.npz"
    split = H.BlockRun(str(split_path), nt, H.BOX_PRIMARY["NX"], DT,
                       block_records=5, manifest=decl)
    w, m, meta = build(case, report, decl, split.conditions)
    extra = dict(saw_meta=json.dumps(meta), **H.prov_kw(__file__, decl))
    original = H.sample_my
    calls = [0]
    def interrupted(magnet):
        calls[0] += 1
        if calls[0] == 6:
            raise InterruptedError("intentional checkpoint/resume test")
        return original(magnet)
    if split.i0 == 0:
        H.sample_my = interrupted
        try:
            split.integrate(w, m, extra)
        except InterruptedError:
            pass
        finally:
            H.sample_my = original
    split = H.BlockRun(str(split_path), nt, H.BOX_PRIMARY["NX"], DT,
                       block_records=5, manifest=decl)
    if split.i0 == 0:
        raise AssertionError("No partial checkpoint was saved")
    w, m, meta = build(case, report, decl, split.conditions)
    split.integrate(w, m, dict(saw_meta=json.dumps(meta), **H.prov_kw(__file__, decl)))
    error = float(np.max(abs(split.my_xt-uninterrupted.my_xt)))
    if error > 1e-7:
        raise RuntimeError("Uniform callback phase was not reproduced on resume")
    return error


def require_preflight():
    report = setup()
    p = plan(report)
    check = json.loads((OUT/"preflight.json").read_text(encoding="utf-8"))
    if (check["status"] != "PASS" or check["source_sha256"] != sha(__file__)
            or check["plan_sha256"] != sha(OUT/"plan.json")
            or {row["case"] for row in check["checks"]} != {c["name"] for c in CASES}):
        raise RuntimeError("A complete matching engine preflight is required")
    return report, p


def predict():
    report, p = require_preflight()
    path = OUT/"uniform_prediction.npz"
    identity = dict(source_sha256=sha(__file__), plan_sha256=sha(OUT/"plan.json"))
    if bool(resume(path, identity).get("done", False)):
        return
    # Reopen the static operator under its original cache declaration.
    # This does not change the timestep of any driven trajectory.
    H.DT_STEP = 2e-13
    try:
        A, info = SL.operator(tuple(report["pbc"]), 6)
    finally:
        H.DT_STEP = DT_STEP
    J = SL.field_matrix(info["gamma"])
    A = A+(report["Bstar"]-SL.B_REF)*J
    d = len(A)
    period = 1/FREQ
    b = p["uniform_amplitude_T"]
    def rhs(s, y):
        return (period*(A+b*np.sin(-2*np.pi*s)*J)@y.reshape(d,d)).ravel()
    solution = solve_ivp(rhs, (0,1), np.eye(d,dtype=complex).ravel(), method="DOP853",
                         rtol=1e-10, atol=1e-12, t_eval=np.arange(26)/25)
    if not solution.success:
        raise RuntimeError(solution.message)
    propagators = solution.y.T.reshape(26,d,d)
    vals, vectors = np.linalg.eig(propagators[-1])
    initial = initial_state(next(c for c in CASES if c["kind"] == "uniform_field"))
    nx = H.BOX_PRIMARY["NX"]
    u0 = (np.fft.fft(initial[1:,0], axis=-1)[:,:,6]/nx*np.exp(-1j*np.pi*6/nx)).ravel()
    weights = np.linalg.solve(vectors,u0)
    trace = np.empty(2001,complex)
    for i in range(2001):
        cycles, phase = divmod(3*i,25)
        u = propagators[phase]@(vectors@(vals**cycles*weights))
        trace[i] = u.reshape(2,H.NY)[0].mean()
    checkpoint(path, identity, done=True, coefficient=trace, mode=6, t=np.arange(2001)*DT,
               floquet_amplitude_rate=float(np.log(abs(vals).max())/period))
    print("uniform prediction saved without reading uniform driven data", flush=True)


def run():
    report, p = require_preflight()
    prediction = resume(OUT/"uniform_prediction.npz",
                        dict(source_sha256=sha(__file__), plan_sha256=sha(OUT/"plan.json")))
    if not bool(prediction.get("done", False)):
        raise RuntimeError("The parameter-free uniform prediction is required before integration")
    for case in CASES:
        write("status.json", dict(state="running", case=case["name"], pid=os.getpid(),
                                   source_sha256=sha(__file__), plan_sha256=sha(OUT/"plan.json")))
        decl = declaration(case, report, T_RUN)
        br = H.BlockRun(str(OUT/(case["name"]+".npz")), 2001, H.BOX_PRIMARY["NX"], DT,
                        block_records=100, manifest=decl)
        if not br.done:
            world, magnet, meta = build(case, report, decl, br.conditions)
            br.integrate(world, magnet, dict(saw_meta=json.dumps(meta), **H.prov_kw(__file__,decl)),
                         progress_every=100)
    write("status.json", dict(state="complete", pid=os.getpid(), analysis="pending"))


def load_complete(path):
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["done"]) or int(data["next_index"]) != 2001:
            raise RuntimeError("Complete control data are required: "+str(path))
        my = data["my_xt"].astype(float)
        if not np.isfinite(my).all() or np.max(abs(my)) > .05:
            raise RuntimeError("Small-angle/finiteness bound failed: "+str(path))
    return my


def analyze():
    report, p = require_preflight()
    fit_plan = json.loads((ROOT/"direct/plan.json").read_text(encoding="utf-8"))
    baseline_path = ROOT/"direct/MEL_eps2e-04_seed20260917.npz"
    baseline = load_complete(baseline_path)
    off_path = ROOT/"direct/MEL_eps0e+00_seed20260917.npz"
    off = load_complete(off_path)
    rows = []
    for case in CASES:
        decl = declaration(case, report, T_RUN)
        br = H.BlockRun(str(OUT/(case["name"]+".npz")), 2001, H.BOX_PRIMARY["NX"], DT,
                        block_records=100, manifest=decl)
        if not br.done:
            raise RuntimeError("No combined report may use an unfinished control")
        path = OUT/(case["name"]+".npz")
        my = load_complete(path)
        spatial = np.fft.fft(my,axis=1)/my.shape[1]
        measured = {}
        for mode in (-6,6):
            fit = estimate(spatial[:,mode % my.shape[1]],fit_plan)
            measured[str(mode)] = fit
        k, frequencies, spectrum = SA.spectrum(my[500:],H.BOX_PRIMARY["dx"],DT,
                                               window="hann",window_x="rect",
                                               detrend=False,shift=False)
        sub = (frequencies >= 2.5e9)&(frequencies <= 3.5e9)
        power = np.sum(abs(spectrum[sub])**2,axis=0)
        shares = dict(positive=float(power[k>0].sum()/power.sum()),
                      negative=float(power[k<0].sum()/power.sum()),
                      zero=float(power[0]/power.sum()))
        row = dict(case=case["name"],data_sha256=sha(path),max_my=float(abs(my).max()),
                   growth_at_half_pump_modes=measured,subharmonic_band_directional_power=shares,
                   directional_power_note="2.5-3.5 GHz, 10-40 ns, all k bins; no spatial mask or smoothing. Not C(K).")
        if case["name"] == "reverse_mel":
            expected = -baseline[:,::-1]
            row["full_trace_reflection_relative_l2"] = float(np.linalg.norm(my-expected)/np.linalg.norm(expected))
            row["reflection_note"] = "The initial axial magnetization is also reflected; this tests covariance, not an independent-seed ensemble."
        if case["name"] == "uniform_field":
            with np.load(OUT/"uniform_prediction.npz",allow_pickle=False) as prediction:
                expected = prediction["coefficient"]
            observed = spatial[:,6]*np.exp(-1j*np.pi*6/my.shape[1])
            row["unfitted_qhalf_trace_relative_l2"] = float(np.linalg.norm(observed-expected)/np.linalg.norm(observed))
            row["predicted_finite_window"] = {"6":estimate(expected,fit_plan),"-6":estimate(expected.conj(),fit_plan)}
        if case["name"] in ("mr_only","mel_and_mr"):
            reference = off if case["name"] == "mr_only" else baseline
            ref_mode = np.fft.fft(reference,axis=1)[:,6]/reference.shape[1]
            row["qhalf_trace_difference_from_reference"] = float(np.linalg.norm(spatial[:,6]-ref_mode)/np.linalg.norm(ref_mode))
            row["reference"] = "pump_off" if case["name"] == "mr_only" else "MEL_only"
        rows.append(row)
    write("analysis.json", dict(rows=rows,source_sha256=sha(__file__),plan_sha256=sha(OUT/"plan.json"),
                                 baseline_sha256=sha(baseline_path),pump_off_sha256=sha(off_path),
                                 status="matched_control_results_not_full_pairing_certificate",
                                 limitations=["single commensurate box", "no ensemble coherence null",
                                              "no threshold bracket", "no detector/thermal visibility calculation"]))
    write("status.json", dict(state="complete",analysis="complete",result="analysis.json",
                               physical_scope="matched deterministic mechanism controls only"))
    print(json.dumps(rows,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("preflight","predict","run","analyze"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"preflight":preflight,"predict":predict,"run":run,"analyze":analyze}[args.stage]()
