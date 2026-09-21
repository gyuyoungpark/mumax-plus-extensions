"""Bounded nonlinear-engine test of the engine-derived MEL prediction.

This is not a G4/G9 campaign certificate. All traces use literature YIG,
commensurate q, explicit weak initial perturbations, and a linked clock64 build.
The existing campaign results and its failing certificate are not overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

import spatial_linear_check as SL
from check_dispersion_reference import sha

H = SL.H
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direct"
sys.path.insert(0, str(ROOT.parent))
import saw_analysis as SA
import growth_interval as GI

DT_REC = 20e-12
RUN_NS = 40
STRAIN = 2e-4
SEEDS = (20260917, 20260918)
AMPLITUDE = 1e-4


def write(name, data):
    OUT.mkdir(exist_ok=True)
    dest = OUT / name
    tmp = dest.with_suffix(dest.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
    os.replace(tmp,dest)


def setup():
    report = json.loads((ROOT/"spatial_linear.json").read_text())
    if report["script_sha256"] != sha(SL.__file__):
        raise RuntimeError("Spatial-reference producer changed")
    if report["convergence_Hz"] > 2e6 or report["perturbation_convergence_Hz"] > 1e5:
        raise RuntimeError("Spatial reference is not converged")
    if report["pump_tangent_rel_error"] > 5e-4:
        raise RuntimeError("Pump tangent failed its engine check")
    H.PBC = tuple(report["pbc"])
    H.DT_STEP = 2e-13
    # Reopening through the producer revalidates source/build/grid/material
    # identities; an unstamped or different-build matrix is never adopted.
    A,info = SL.operator(H.PBC,H.BOX_PRIMARY["bin_qhalf"])
    A = A+(report["Bstar"]-SL.B_REF)*SL.field_matrix(info["gamma"])
    if abs(SL.fundamental(A)-3e9)>1e5:
        raise RuntimeError("Reference Bstar does not reproduce the declared resonance")
    return report,A


def integrate(label, initial, B0, strain, seed, dt, duration, mode_kind):
    H.DT_STEP=dt
    H.SEED_KIND="engine_eigenmode" if mode_kind.startswith("engine") else "bandlimited_yflat_randphase"
    bx=H.BOX_PRIMARY
    nt=round(duration/DT_REC)+1
    decl=dict(box=bx,material=H.YIG_LIT,B0=B0,eps0=strain,f_saw=6e9,
              temperature=0.,rng_seed=seed,a_seed=AMPLITUDE,k_cut=2*bx["q"],
              seed_kind=H.SEED_KIND,initialization=mode_kind,
              T_run=duration,DT_REC=DT_REC,DT_STEP=dt,
              reference_sha256=sha(ROOT/"spatial_linear.json"),
              driver_sha256=sha(__file__),
              purpose="deterministic weak-seed mechanism check; not thermal detectability")
    path=OUT/(label+".npz")
    br=H.BlockRun(str(path),nt,bx["NX"],DT_REC,block_records=100,manifest=decl)
    if not br.done:
        world,magnet,meta=H.build(bx,H.YIG_LIT,B0,strain,6e9,initial,conditions=br.conditions)
        br.integrate(world,magnet,dict(saw_meta=json.dumps(meta),**H.prov_kw(__file__,decl)),
                     progress_every=100)
    return br.my_xt


def harmonic_fit(my, mode):
    s=np.fft.fft(my,axis=1)[:,mode]/my.shape[1]
    t=np.arange(len(s))*DT_REC
    phase=np.unwrap(np.angle(s))
    p=np.polyfit(t,phase,1)
    g=np.polyfit(t,np.log(abs(s)),1)
    return dict(frequency_Hz=float(-p[0]/(2*np.pi)),growth_per_s=float(g[0]),
                phase_residual_rms=float(np.sqrt(np.mean((phase-np.polyval(p,t))**2))))


def smoke():
    OUT.mkdir(exist_ok=True)
    report,A=setup()
    values,vectors=np.linalg.eig(A)
    candidates=np.flatnonzero(values.imag<0)
    j=candidates[np.argmin(abs(values[candidates].imag+2*np.pi*3e9))]
    v=vectors[:,j].reshape(2,H.NY)
    v/=np.max(abs(v[0]))
    nx=H.BOX_PRIMARY["NX"]
    carrier=np.exp(2j*np.pi*H.BOX_PRIMARY["bin_qhalf"]*(np.arange(nx)+.5)/nx)
    initial=np.zeros((3,1,H.NY,nx))
    initial[0]=1
    initial[1:,0]=AMPLITUDE*np.real(v[...,None]*carrier)
    initial/=np.linalg.norm(initial,axis=0)
    results=[]
    for dt in (2e-13,4e-13):
        label="unpumped_eigenmode_dt%.0efs" % (dt*1e15)
        my=integrate(label,initial,report["Bstar"],0.,0,dt,2e-9,"engine eigenmode, amplitude 1e-4")
        fit=harmonic_fit(my,H.BOX_PRIMARY["bin_qhalf"])
        fit["dt_step"]=dt
        results.append(fit)
    ok=all(abs(r["frequency_Hz"]-3e9)<2e6 and r["phase_residual_rms"]<.001
           and abs(r["growth_per_s"]-values[j].real)<2e5 for r in results)
    ok=ok and abs(results[0]["frequency_Hz"]-results[1]["frequency_Hz"])<2e5
    summary=dict(status="PASS" if ok else "FAIL",results=results,
                 reference_frequency_Hz=float(-values[j].imag/(2*np.pi)),
                 reference_growth_per_s=float(values[j].real),
                 driver_sha256=sha(__file__),reference_sha256=sha(ROOT/"spatial_linear.json"),
                 criteria=dict(frequency_error_Hz=2e6,growth_error_per_s=2e5,
                               phase_rms_rad=.001,dt_frequency_difference_Hz=2e5))
    write("smoke.json",summary)
    print(json.dumps(summary),flush=True)
    if not ok:
        raise RuntimeError("Short spatial-engine integration failed")


def run():
    report,_=setup()
    check=json.loads((OUT/"smoke.json").read_text())
    if (check["status"]!="PASS" or check["driver_sha256"]!=sha(__file__)
            or check["reference_sha256"]!=sha(ROOT/"spatial_linear.json")):
        raise RuntimeError("A passing, matching short integration is required")
    jobs=[(0.,SEEDS[0]),(STRAIN,SEEDS[0]),(STRAIN,SEEDS[1])]
    plan=dict(jobs=jobs,run_ns=RUN_NS,dt_step=4e-13,dt_rec=DT_REC,
              seed_amplitude=AMPLITUDE,pbc=report["pbc"],B0=report["Bstar"],
              material=H.YIG_LIT,claim="weak-seed amplification only; no threshold or detector certificate",
              growth_analysis_window_ns=[10,40],growth_segment_ns=2,
              max_linear_my=.05,driver_sha256=sha(__file__))
    write("plan.json",plan)
    for strain,seed in jobs:
        label="MEL_eps%.0e_seed%d"%(strain,seed)
        write("status.json",dict(state="running",case=label,pid=os.getpid()))
        initial,_=H.make_seed(seed,H.BOX_PRIMARY["NX"],H.BOX_PRIMARY["dx"],
                              AMPLITUDE,2*H.BOX_PRIMARY["q"])
        integrate(label,initial,report["Bstar"],strain,seed,4e-13,RUN_NS*1e-9,
                  "explicit band-limited small perturbation; not a thermal bath")
    write("status.json",dict(state="complete",pid=os.getpid(),physical_verdict="analysis_pending"))


def analyze():
    report,_=setup()
    rows=[]
    half=H.BOX_PRIMARY["bin_qhalf"]
    for strain,seed in [(0.,SEEDS[0]),(STRAIN,SEEDS[0]),(STRAIN,SEEDS[1])]:
        path=OUT/("MEL_eps%.0e_seed%d.npz"%(strain,seed))
        with np.load(path,allow_pickle=False) as data:
            if not bool(data["done"]):
                raise RuntimeError("An unfinished trace cannot support a verdict")
            my=data["my_xt"].astype(float)
        if np.max(abs(my))>.05:
            raise RuntimeError("Declared small-angle limit breached; do not fit a saturated trace")
        spatial=np.fft.fft(my,axis=1)/my.shape[1]
        seg=round(2e-9/DT_REC)
        amplitudes=[]
        times=[]
        for start in range(0,len(my)-seg+1,round(.2e-9/DT_REC)):
            f,s=SA.time_spectrum(spatial[start:start+seg,half],DT_REC,window="hann")
            amplitudes.append(abs(s[np.argmin(abs(f-3e9))])/np.hanning(seg).sum())
            times.append((start+(seg-1)/2)*DT_REC)
        times,amplitudes=np.array(times),np.array(amplitudes)
        interval=(times>=10e-9)&(times<=40e-9)
        fit=GI.fit_growth_interval(times[interval],amplitudes[interval],
                                   floor_abs=1e-8,min_efold=1.,label="declared 10-40 ns window")
        sl=np.polyfit(times[interval],np.log(amplitudes[interval]),1)
        f,S=SA.time_spectrum(spatial[round(10e-9/DT_REC):,half],DT_REC,window="hann",pad_t=8)
        ids=np.flatnonzero((f>1e9)&(f<5e9))
        peak=float(f[ids[np.argmax(abs(S[ids]))]])
        pred=next(r["amplitude_growth_per_s"] for r in report["floquet"]
                  if r["strain"]==strain and r["harmonics"]==6 and r["rtol"]==1e-8)
        rows.append(dict(strain=strain,seed=seed,file_sha256=sha(path),
                         max_my=float(np.max(abs(my))),frequency_peak_Hz=peak,
                         fitted_growth=fit,descriptive_slope_per_s=float(sl[0]),
                         predicted_growth_per_s=pred,
                         note="A held rate is not a negative-rate threshold point. FFT padding is not extra resolution."))
    result=dict(rows=rows,status="limited_mechanism_check_not_full_pairing_certificate",
                residual_requirements=["nondegenerate idler control", "other commensurate box",
                                       "onset bracket", "physical thermal background / detector model"],
                source_sha256=sha(__file__))
    write("analysis.json",result)
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("smoke","run","analyze"))
    args=parser.parse_args()
    {"smoke":smoke,"run":run,"analyze":analyze}[args.stage]()
