"""Raw-time phase measurements; no f_K slice or enforced frequency sum.

Descriptive diagnostics only. Phase stability in one deterministic trajectory
is not a pairing certificate or an independently sampled correlation test.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import saw_analysis as SA
from check_dispersion_reference import sha

DT=20e-12
F_PUMP=6e9
FIT_START,FIT_END=10e-9,35e-9


def forward_branch(s):
    f,S=SA.time_spectrum(s,DT,window="rect")
    # Inverse of the canonical positive-frequency time transform.
    return np.fft.fft(np.where(f>0,S,0))/len(s)


def measure_pair(a,b):
    a,b=forward_branch(a),forward_branch(b)
    t=np.arange(len(a))*DT
    mask=(t>=FIT_START)&(t<=FIT_END)
    fa=-np.polyfit(t[mask],np.unwrap(np.angle(a))[mask],1)[0]/(2*np.pi)
    fb=-np.polyfit(t[mask],np.unwrap(np.angle(b))[mask],1)[0]/(2*np.pi)
    relative=np.unwrap(np.angle(a*b*np.exp(2j*np.pi*F_PUMP*t)))
    return dict(f_a_Hz=float(fa),f_b_Hz=float(fb),sum_Hz=float(fa+fb),
                sum_residual_Hz=float(fa+fb-F_PUMP),
                pump_relative_phase_std_rad=float(np.std(relative[mask])),
                amplitude_a_min=float(abs(a[mask]).min()),
                amplitude_b_min=float(abs(b[mask]).min()),
                caveat="descriptive; branch-filter end effects checked on synthetic signals; not a correlation significance test")


def controls():
    t=np.arange(2001)*DT
    records=[]
    for fa,fb,gain in ((2.8e9,3.2e9,0.),(2.8e9,3.2e9,5.65e7),
                       (2.9e9,3.25e9,5.65e7),(3e9,3e9,5.65e7)):
        a=np.exp(gain*t)*(np.exp(-2j*np.pi*fa*t+.3j)+.01*np.exp(2j*np.pi*2.5e9*t))
        b=np.exp(gain*t)*(np.exp(-2j*np.pi*fb*t-.8j)+.01*np.exp(2j*np.pi*2.2e9*t))
        result=measure_pair(a,b)
        result.update(true_f_a_Hz=fa,true_f_b_Hz=fb,growth_per_s=gain)
        result["pass_check"]=bool(abs(result["f_a_Hz"]-fa)<2e6 and abs(result["f_b_Hz"]-fb)<2e6)
        records.append(result)
    dest=ROOT/"phase_controls.json"
    dest.write_text(json.dumps(dict(script_sha256=sha(__file__),records=records),indent=2),encoding="utf-8")
    if not all(r["pass_check"] for r in records):
        raise AssertionError("Raw-time phase estimator failed controls")
    print(json.dumps(records,indent=2))


def analyze():
    test=json.loads((ROOT/"phase_controls.json").read_text())
    if test["script_sha256"]!=sha(__file__) or not all(r["pass_check"] for r in test["records"]):
        raise RuntimeError("Matching passing synthetic controls required")
    rows=[]
    for eps,seed in ((0.,20260917),(2e-4,20260917),(2e-4,20260918)):
        path=ROOT/"direct"/("MEL_eps%.0e_seed%d.npz"%(eps,seed))
        with np.load(path,allow_pickle=False) as data:
            if not bool(data["done"]):
                raise RuntimeError("Full trajectory required")
            spatial=np.fft.fft(data["my_xt"].astype(float),axis=1)/data["my_xt"].shape[1]
        for a,b in ((5,7),(6,6)):
            result=measure_pair(spatial[:,a],spatial[:,b])
            result.update(eps=eps,seed=seed,modes=[a,b],source_sha256=sha(path),
                          degenerate=(a==b))
            rows.append(result)
    report=dict(script_sha256=sha(__file__),fit_window_ns=[10,35],rows=rows,
                status="descriptive_only_no_pairing_verdict",
                note="No temporal slice at f_K; positive temporal branch only. A held-out idler and spectator experiment remains necessary for a causal pair-assignment test.")
    (ROOT/"direct/phase_analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("stage",choices=("controls","analyze"))
    args=p.parse_args()
    {"controls":controls,"analyze":analyze}[args.stage]()
