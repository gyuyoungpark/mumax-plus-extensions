"""Engine-derived spatial LLG reference for a collinear MEL-only experiment.

No fit to pumped trajectories. All transverse y rows are retained; Fourier
harmonics are increased to test truncation. Floquet eigenvalues are predictions,
not a replacement for the nonlinear engine/control runs.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from check_dispersion_reference import checkpoint, resume, sha

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runs"))
import _harness as H

B_REF = .05
F_PUMP = 6e9
STEP = 1e-4
MATERIAL = H.YIG_LIT


def construct(pbc, B0=B_REF, eps=0):
    H.PBC = tuple(pbc)
    bx = H.BOX_PRIMARY
    initial = np.zeros((3, H.NZ, H.NY, bx["NX"]))
    initial[0] = 1
    decl = dict(box=bx, material=MATERIAL, B0=B0, eps0=eps, f_saw=F_PUMP,
                temperature=0, rng_seed=0, a_seed=STEP, k_cut=0)
    cs = H.conditions(decl, source="engine spatial linearization", nt=2,
                      dt_rec=1e-12, block_records=2)
    cs.require_execution("spatial reference", nx=bx["NX"])
    world, magnet, _ = H.build(bx, MATERIAL, B0, eps, F_PUMP, initial, conditions=cs)
    return world, magnet, initial, cs


def operator(pbc, mode, step=STEP):
    world, magnet, initial, cs = construct(pbc)
    identity = dict(script=sha(__file__), stamp=cs.stamp(), mode=mode, step=step)
    path = HERE / ("linear_pbc%s_mode%d_h%.0e.npz" % ("x".join(map(str,pbc)), mode, step))
    saved = resume(path, identity)
    if bool(saved.get("done", False)):
        return saved["matrix"], json.loads(str(saved["info"]))
    ny, nx = H.NY, H.BOX_PRIMARY["NX"]
    phase = 2*np.pi*mode*(np.arange(nx)+.5)/nx
    plane = np.exp(1j*phase)
    matrix = saved.get("matrix", np.full((2*ny,2*ny), np.nan+0j))
    leakage = saved.get("leakage", np.full(2*ny, np.nan))
    for col in range(2*ny):
        if np.isfinite(matrix[:,col]).all():
            continue
        action = np.zeros((2,ny,nx), dtype=complex)
        for carrier, multiplier in ((plane.real, 1), (plane.imag, 1j)):
            if not np.any(carrier):
                continue
            responses = []
            for sign in (1,-1):
                state = initial.copy()
                state[1+col//ny,0,col%ny] = sign*step*carrier
                state /= np.linalg.norm(state,axis=0)
                magnet.magnetization = state
                responses.append(magnet.llg_torque.eval()[1:,0].astype(float))
            action += multiplier*(responses[0]-responses[1])/(2*step)
        projected = np.mean(action*plane.conj(),axis=-1)
        matrix[:,col] = projected.ravel()
        leakage[col] = np.linalg.norm(action-projected[...,None]*plane)/np.linalg.norm(action)
        checkpoint(path, identity, matrix=matrix, leakage=leakage, done=False)
    gamma = float(magnet.gamma.eval().mean())
    info = dict(mode=mode, pbc=list(pbc), B_ref=B_REF, gamma=gamma,
                max_projection_leakage=float(leakage.max()),
                fundamental_Hz=fundamental(matrix))
    checkpoint(path, identity, matrix=matrix, leakage=leakage,
               info=json.dumps(info), done=True)
    print("operator", json.dumps(info), flush=True)
    return matrix, info


def fundamental(matrix):
    values,vectors = np.linalg.eig(matrix)
    candidates = np.flatnonzero(values.imag > 0)
    weights = [sum(abs(vectors[c*H.NY:(c+1)*H.NY,j].sum())**2 for c in (0,1))
               for j in candidates]
    return float(values[candidates[np.argmax(weights)]].imag/(2*np.pi))


def field_matrix(gamma):
    eye = np.eye(H.NY)
    alpha = MATERIAL["ALPHA"]
    return gamma/(1+alpha**2)*np.block([[-alpha*eye,-eye],[eye,-alpha*eye]])


def check_pump_linearization(pbc, J):
    # Independently compare the analytic modulation of the tangent operator
    # against the engine on a transverse field not used to construct J.
    off = construct(pbc, eps=0)
    on = construct(pbc, eps=1e-4)
    nx = H.BOX_PRIMARY["NX"]
    x = (np.arange(nx)+.5)*H.BOX_PRIMARY["dx"]
    rng = np.random.default_rng(20260921)
    v = rng.normal(size=(2,H.NY,1))*np.cos(2*np.pi*7*x/H.BOX_PRIMARY["L"])
    differences=[]
    for state in (off,on):
        world,magnet,initial,_ = state
        responses=[]
        for sign in (1,-1):
            m=initial.copy()
            m[1:,0]=sign*STEP*v
            m/=np.linalg.norm(m,axis=0)
            magnet.magnetization=m
            responses.append(magnet.llg_torque.eval()[1:,0].astype(float))
        differences.append((responses[0]-responses[1])/(2*STEP))
    observed=(differences[1]-differences[0]).reshape(2*H.NY,nx)
    b=-2*MATERIAL["B1"]/MATERIAL["MS"]*1e-4
    predicted=(J@v.reshape(2*H.NY,nx))*b*np.sin(H.BOX_PRIMARY["q"]*x)
    error=float(np.linalg.norm(observed-predicted)/np.linalg.norm(predicted))
    if error > 5e-4:
        raise AssertionError(f"Pump tangent check failed: {error}")
    return error


def floquet(blocks, J, Bstar, strain, rtol):
    n=len(blocks)
    d=J.shape[0]
    diagonal=np.stack([a+(Bstar-B_REF)*J for a in blocks])
    b=-2*MATERIAL["B1"]/MATERIAL["MS"]*strain
    period=1/F_PUMP
    def rhs(s, flat):
        Y=flat.reshape(n,d,n*d)
        DY=diagonal@Y
        DY[1:] += (-.5j*b*np.exp(-2j*np.pi*s))*(J@Y[:-1])
        DY[:-1] += (.5j*b*np.exp(2j*np.pi*s))*(J@Y[1:])
        return (period*DY).ravel()
    sol=solve_ivp(rhs,(0,1),np.eye(n*d,dtype=complex).ravel(),method="DOP853",
                  rtol=rtol,atol=rtol*.01,t_eval=[1])
    if not sol.success:
        raise RuntimeError(sol.message)
    values,vectors=np.linalg.eig(sol.y[:,-1].reshape(n*d,n*d))
    j=int(np.argmax(abs(values)))
    weights=np.sum(abs(vectors[:,j].reshape(n,d))**2,axis=1)
    return dict(strain=strain,rtol=rtol,harmonics=n,
                amplitude_growth_per_s=float(np.log(abs(values[j]))/period),
                quasifrequency_Hz=float(np.angle(values[j])/(2*np.pi*period)),
                harmonic_weights=(weights/weights.sum()).tolist(),nfev=sol.nfev)


def main():
    with threadpool_limits(limits=1):
        half=H.BOX_PRIMARY["bin_qhalf"]
        rows=[]
        for pbc in ((2,2,0),(2,32,0),(4,64,0),(8,128,0)):
            A,info=operator(pbc,half)
            rows.append(info)
        pbc=(4,64,0)
        base,info=operator(pbc,half)
        fine,_=operator(pbc,half,STEP/2)
        J=field_matrix(info["gamma"])
        Bstar=brentq(lambda B:fundamental(base+(B-B_REF)*J)-F_PUMP/2,.02,.08,xtol=1e-13)
        conv=abs(rows[-1]["fundamental_Hz"]-rows[-2]["fundamental_Hz"])
        fd=abs(fundamental(base)-fundamental(fine))
        if conv > 2e6 or fd > 1e5:
            raise AssertionError(f"Linear reference not converged: PBC {conv} Hz; FD {fd} Hz")
        pump_error=check_pump_linearization(pbc,J)
        positive=[operator(pbc,j*half)[0] for j in (1,3,5)]
        blocks=[a.conj() for a in reversed(positive)]+positive
        report=dict(script_sha256=sha(__file__),material=MATERIAL,pbc=list(pbc),
                    Bstar=Bstar,box=H.BOX_PRIMARY,convergence_Hz=conv,
                    perturbation_convergence_Hz=fd,pump_tangent_rel_error=pump_error,
                    boundary_scan=rows,floquet=[],
                    status="linear_prediction_only_not_pairing_certification")
        destination=HERE/"spatial_linear.json"
        old=json.loads(destination.read_text()) if destination.exists() else None
        if old and any(old.get(k)!=v for k,v in report.items() if k!="floquet"):
            raise RuntimeError("Floquet summary identity mismatch")
        if old:
            report["floquet"]=old["floquet"]
        cases=[(0.,6,1e-8),(3e-5,6,1e-8),(7e-5,6,1e-8),(1e-4,6,1e-8),
               (2e-4,6,1e-8),(1e-4,6,1e-10),(1e-4,2,1e-8)]
        for strain,n,tol in cases:
            if any(r["strain"]==strain and r["harmonics"]==n and r["rtol"]==tol for r in report["floquet"]):
                continue
            chosen=blocks if n==6 else blocks[2:4]
            result=floquet(chosen,J,Bstar,strain,tol)
            report["floquet"].append(result)
            tmp=destination.with_suffix(".tmp")
            tmp.write_text(json.dumps(report,indent=2),encoding="utf-8")
            os.replace(tmp,destination)
            print("Floquet",json.dumps(result),flush=True)
        print("Bstar",Bstar,"PBC convergence Hz",conv,"pump tangent error",pump_error,flush=True)


if __name__=="__main__":
    main()
