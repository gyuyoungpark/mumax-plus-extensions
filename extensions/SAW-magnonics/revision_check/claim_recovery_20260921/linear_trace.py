"""Parameter-free tangent-LLG trajectories from the declared initial seeds.

The only magnetic inputs are the static engine Jacobians and the initial
condition. Pumped GPU trajectories are read only in the compare stage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

import spatial_linear_check as SL
from check_dispersion_reference import checkpoint, resume, sha

ROOT=Path(__file__).resolve().parent
H=SL.H
DT=20e-12
N=2001
MODES=np.array([-30,-18,-6,6,18,30])


def predict():
    report=json.loads((ROOT/"spatial_linear.json").read_text())
    if report["script_sha256"]!=sha(SL.__file__):
        raise RuntimeError("Spatial reference source mismatch")
    pbc=tuple(report["pbc"])
    positives=[]
    for mode in (6,18,30):
        A,info=SL.operator(pbc,mode)
        positives.append(A)
    J=SL.field_matrix(info["gamma"])
    blocks=[a.conj() for a in reversed(positives)]+positives
    diagonal=np.stack([a+(report["Bstar"]-SL.B_REF)*J for a in blocks])
    d=2*H.NY
    period=1/6e9
    cases=((0.,20260917),(2e-4,20260917),(2e-4,20260918))
    for strain,seed in cases:
        path=ROOT/("prediction_eps%.0e_seed%d.npz"%(strain,seed))
        identity=dict(script_sha256=sha(__file__),reference_sha256=sha(ROOT/"spatial_linear.json"),
                      strain=strain,seed=seed,amplitude=1e-4,dt_record=DT,nt=N,
                      modes=MODES.tolist(),note="no pumped trace used; no fitted parameters")
        if bool(resume(path,identity).get("done",False)):
            print("cached",path.name,flush=True)
            continue
        b=-2*H.YIG_LIT["B1"]/H.YIG_LIT["MS"]*strain
        def rhs(s,flat):
            Y=flat.reshape(6,d,6*d)
            DY=diagonal@Y
            DY[1:]+=(-.5j*b*np.exp(-2j*np.pi*s))*(J@Y[:-1])
            DY[:-1]+=(.5j*b*np.exp(2j*np.pi*s))*(J@Y[1:])
            return (period*DY).ravel()
        sol=solve_ivp(rhs,(0,1),np.eye(6*d,dtype=complex).ravel(),method="DOP853",
                      rtol=1e-10,atol=1e-12,t_eval=np.arange(26)/25)
        if not sol.success:
            raise RuntimeError(sol.message)
        propagators=sol.y.T.reshape(26,6*d,6*d)
        vals,vec=np.linalg.eig(propagators[-1])
        initial,_=H.make_seed(seed,H.BOX_PRIMARY["NX"],H.BOX_PRIMARY["dx"],1e-4,
                              2*H.BOX_PRIMARY["q"])
        spatial=np.fft.fft(initial[1:,0],axis=-1)/initial.shape[-1]
        u0=np.concatenate([spatial[:,:,int(mode)%initial.shape[-1]].ravel()
                           *np.exp(-1j*np.pi*mode/initial.shape[-1]) for mode in MODES])
        coefficients=np.linalg.solve(vec,u0)
        predicted=np.empty((N,6),complex)
        for i in range(N):
            cycles,remainder=divmod(3*i,25)
            u=propagators[remainder]@(vec@(vals**cycles*coefficients))
            predicted[i]=u.reshape(6,2,H.NY)[:,0].mean(axis=1)
        checkpoint(path,identity,done=True,my_modes=predicted,t=np.arange(N)*DT,modes=MODES,
                   initial_state=initial,nfev=sol.nfev)
        print("predicted",path.name,flush=True)


def compare():
    rows=[]
    for strain,seed in ((0.,20260917),(2e-4,20260917),(2e-4,20260918)):
        predpath=ROOT/("prediction_eps%.0e_seed%d.npz"%(strain,seed))
        datapath=ROOT/"direct"/("MEL_eps%.0e_seed%d.npz"%(strain,seed))
        with np.load(predpath,allow_pickle=False) as p:
            identity=json.loads(str(p["identity"]))
            if identity["script_sha256"]!=sha(__file__) or not bool(p["done"]):
                raise RuntimeError("Prediction identity mismatch")
            expected=p["my_modes"].copy()
        with np.load(datapath,allow_pickle=False) as data:
            if not bool(data["done"]) or data["my_xt"].shape[0]!=N:
                raise RuntimeError("Complete matching observed trace required")
            observed=np.fft.fft(data["my_xt"].astype(float),axis=1)/data["my_xt"].shape[1]
        j=3
        actual=observed[:,6]*np.exp(-1j*np.pi*6/observed.shape[1])
        prediction=expected[:,j]
        error=float(np.linalg.norm(actual-prediction)/np.linalg.norm(actual))
        rows.append(dict(strain=strain,seed=seed,mode=6,
                         complex_trace_relative_l2_error=error,
                         fitted_parameters=0,prediction_sha256=sha(predpath),data_sha256=sha(datapath)))
    report=dict(rows=rows,source_sha256=sha(__file__),
                scope="one commensurate box, weak deterministic seeds, MEL only; not full pairing certification")
    (ROOT/"direct/linear_trace_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("stage",choices=("predict","compare"))
    args=p.parse_args()
    with threadpool_limits(limits=1):
        {"predict":predict,"compare":compare}[args.stage]()
