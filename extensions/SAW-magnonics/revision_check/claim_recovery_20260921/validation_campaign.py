"""Prospective, checkpointed follow-up to the completed matched controls.

No manuscript or old data are modified. Run preflight, then run; analyze can
be resumed without integration. Failure stops dependent stages, never grants
a physical PASS. Thermal streams are independently seeded per saved block.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

import direct_check as DC
import spatial_linear_check as SL
from check_dispersion_reference import checkpoint, resume, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
H = SL.H
sys.path.insert(0, str(ROOT.parent))
import saw_analysis as SA
import growth_interval as GI

DT = 20e-12
EPS = 2e-4
F = 6e9
SEEDS = list(range(20261001, 20261013))
THERMAL_SEEDS = list(range(20262001, 20262013))
GEOMETRIES = {
    "base": dict(periods=12, dx=5e-9, dy=10e-9, dz=20e-9, ny=8, nz=1, dt=4e-13),
    "long": dict(periods=18, dx=5e-9, dy=10e-9, dz=20e-9, ny=8, nz=1, dt=4e-13),
    "thickness": dict(periods=12, dx=5e-9, dy=10e-9, dz=10e-9, ny=8, nz=2, dt=2e-13),
    "fine": dict(periods=12, dx=2.5e-9, dy=5e-9, dz=10e-9, ny=16, nz=2, dt=1e-13),
}


def safe(v):
    if isinstance(v, dict):
        return {k: safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def write(name, v):
    OUT.mkdir(exist_ok=True)
    dest = OUT / name
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(v), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(dest)


def configure(name):
    g = GEOMETRIES[name]
    H.NY, H.NZ, H.CY, H.CZ = g["ny"], g["nz"], g["dy"], g["dz"]
    H.DT_STEP = g["dt"]
    H.PBC = (4, 64, 0)
    H.SAW_PHASE = 0.
    H.ENABLE_BARNETT = False
    H.SEED_KIND = "explicit_validation_initial_state"
    return H.box(g["periods"], dx=g["dx"])


def setup():
    configure("base")
    H.SEED_KIND = "bandlimited_yflat_randphase"
    report, _ = DC.setup()
    configure("base")
    p = dict(schema="saw_validation/1", driver=sha(__file__),
             reference=sha(ROOT/"spatial_linear.json"),
             harness=sha(H.__file__), spectrum=sha(SA.__file__), growth=sha(GI.__file__),
             B0=report["Bstar"], material=H.YIG_LIT, strain=EPS,
             geometries=GEOMETRIES, seeds=SEEDS, thermal_seeds=THERMAL_SEEDS,
             record_interval_s=DT, deterministic_duration_ns=40,
             ensemble_window_ns=[30, 40], ensemble_primary_mode="+q/2 at +3 GHz",
             ensemble_channels=["MEL", "OFF", "MR"],
             coherence="abs(sum(a_j**2))/sum(abs(a_j)**2), one independent seed per term",
             null="independent uniform pair phases conditional on each measured |a_j|^2",
             null_samples=100000, primary_p_limit=.01, bootstrap_samples=10000,
             convergence=dict(frequency_difference_Hz=3e6, relative_growth_difference=.05,
                              all_growth_must_be_readable=True, refit_B0=False),
             threshold_strains=[0., 2e-5, 5e-5],
             threshold_scope="q/2 Floquet sector only; NOT a proven global minimum over all modes",
             threshold_duration_rule="ceil(2/abs(predicted_rate)/20ns)*20ns; max 600ns",
             thermal=dict(T_K=300., burn_ns=10., burn_alpha=.05, observation_ns=40., dt_step=2e-13,
                          block_ns=2., noise="engine Brown noise, independent deterministic seed for every block",
                          background="matched pump-off ensemble", optical_calibration="absent"),
             visibility="conditional count requirements only; no absolute BLS SNR",
             limits="No independence of overlapping time windows; no detector noise inferred from FFT residuals.")
    path = OUT/"plan.json"
    if path.exists() and json.loads(path.read_text()) != p:
        raise RuntimeError("Frozen validation plan changed")
    write("plan.json", p)
    return p


def decl(p, geom, eps, seed, duration, initial_kind, *, mr=0., temp=0., alpha=None):
    bx = configure(geom)
    mat = dict(H.YIG_LIT, KMR=mr)
    if alpha is not None:
        mat["ALPHA"] = alpha
    return dict(box=bx, material=mat, B0=p["B0"], eps0=eps, f_saw=F,
                temperature=temp, rng_seed=seed, a_seed=1e-4, k_cut=2*bx["q"],
                seed_kind=H.SEED_KIND, initialization=initial_kind,
                T_run=duration, DT_REC=DT, DT_STEP=H.DT_STEP,
                source_sha256=p["driver"], plan_sha256=sha(OUT/"plan.json"),
                enable_mel=(mr == 0.), saw_direction=1)


def field_check(world, magnet, d):
    from mumaxplus import _cpp
    from mumaxplus.fieldquantity import FieldQuantity
    if not np.isclose(world.timesolver.timestep, d["DT_STEP"], rtol=2e-6):
        raise RuntimeError("Wrong applied step")
    if world.timesolver.adaptive_timestep:
        raise RuntimeError("Adaptive stepping is undeclared")
    if not d["eps0"]:
        if magnet.enable_saw:
            raise RuntimeError("Pump-off world has SAW enabled")
        return 0.
    actual = FieldQuantity(_cpp.chiral_saw_field(magnet._impl)).eval().astype(float)
    state = magnet.magnetization.eval().astype(float)
    bx, mat = d["box"], d["material"]
    x = (np.arange(bx["NX"])+.5)*bx["dx"]
    phase = bx["q"]*x - 2*np.pi*F*float(world.timesolver.time)
    expected = np.zeros_like(actual)
    if d["enable_mel"]:
        expected[0] = -2*mat["B1"]/mat["MS"]*d["eps0"]*np.sin(phase)*state[0]
    if mat["KMR"]:
        rotation = H.XI*d["eps0"]/2*np.cos(phase)
        expected[0] += mat["KMR"]/mat["MS"]*rotation*state[2]
        expected[2] += mat["KMR"]/mat["MS"]*rotation*state[0]
    error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    if error > 5e-4:
        raise RuntimeError("Applied pump failed SI check")
    return error


def make_world(d, initial, cs):
    world, magnet, meta = H.build(d["box"], d["material"], d["B0"], d["eps0"], F,
                                   initial, temperature=d["temperature"],
                                   enable_mel=d["enable_mel"], conditions=cs)
    field_check(world, magnet, d)
    return world, magnet, meta


def deterministic(p, label, geom, eps, seed, initial, duration=40e-9, mr=0., kind="explicit plane wave"):
    d = decl(p, geom, eps, seed, duration, kind, mr=mr)
    nt = round(duration/DT)+1
    path = OUT/(label+".npz")
    br = H.BlockRun(str(path), nt, d["box"]["NX"], DT, block_records=100, manifest=d)
    if not br.done:
        w, m, meta = make_world(d, initial, br.conditions)
        br.integrate(w, m, dict(saw_meta=json.dumps(meta), **H.prov_kw(__file__, d)), progress_every=500)
    if not np.isfinite(br.my_xt).all() or abs(br.my_xt).max() > .05:
        raise RuntimeError("Deterministic small-angle bound exceeded")
    return br.my_xt.astype(float)


def plane(p, geom):
    bx = configure(geom)
    x = (np.arange(bx["NX"])+.5)*bx["dx"]
    phase = bx["q"]/2*x + np.pi/8
    ex = 2*H.YIG_LIT["AEX"]*(bx["q"]/2)**2/H.YIG_LIT["MS"]
    ratio = np.sqrt((p["B0"]+ex)/(p["B0"]+ex+H.MU0*H.YIG_LIT["MS"]))
    m = np.zeros((3,H.NZ,H.NY,bx["NX"]))
    m[1] = 1e-4*np.cos(phase)
    m[2] = -1e-4*ratio*np.sin(phase)
    m[0] = np.sqrt(1-m[1]**2-m[2]**2)
    return m


def matrix(p, geom, mode):
    d = decl(p, geom, 0., 0, 0., "central-difference tangent operator")
    bx = d["box"]
    cs = H.conditions(d, source="validation static operator", nt=2, dt_rec=DT, block_records=2)
    ident = dict(driver=p["driver"], conditions=cs.stamp(), mode=int(mode), h=1e-4)
    path = OUT/("matrix_%s_k%d.npz" % (geom, mode))
    old = resume(path, ident)
    if bool(old.get("done", False)):
        return old["A"], float(old["gamma"])
    initial = np.zeros((3,H.NZ,H.NY,bx["NX"]))
    initial[0] = 1
    w,m,_ = make_world(d, initial, cs)
    transverse = H.NY*H.NZ
    A = old.get("A", np.full((2*transverse,2*transverse), np.nan+0j))
    phase = np.exp(2j*np.pi*mode*(np.arange(bx["NX"])+.5)/bx["NX"])
    for col in range(2*transverse):
        if np.isfinite(A[:,col]).all():
            continue
        response = np.zeros((2,H.NZ,H.NY,bx["NX"]), complex)
        component, at = divmod(col, transverse)
        z,y = divmod(at,H.NY)
        for carrier, weight in ((phase.real,1), (phase.imag,1j)):
            vals = []
            for sign in (1,-1):
                s = initial.copy()
                s[1+component,z,y] = sign*1e-4*carrier
                s /= np.linalg.norm(s,axis=0)
                m.magnetization = s
                vals.append(m.llg_torque.eval()[1:].astype(float))
            response += weight*(vals[0]-vals[1])/2e-4
        projected = np.mean(response*phase.conj(),axis=-1)
        leakage = np.linalg.norm(response-projected[...,None]*phase)/np.linalg.norm(response)
        if leakage > 1e-3:
            raise RuntimeError("Fourier-sector projection leakage")
        A[:,col] = projected.ravel()
        checkpoint(path, ident, A=A, done=False)
    gamma = float(m.gamma.eval().mean())
    checkpoint(path, ident, A=A, gamma=gamma, done=True)
    return A,gamma


def fit(series, duration):
    times, amps = [], []
    n = round(2e-9/DT)
    for start in range(0,len(series)-n+1,10):
        f,s = SA.time_spectrum(series[start:start+n],DT,window="hann")
        amps.append(abs(s[np.argmin(abs(f-3e9))])/np.hanning(n).sum())
        times.append((start+(n-1)/2)*DT)
    t,a = np.asarray(times),np.asarray(amps)
    select = (t >= 10e-9)&(t <= duration)
    result = GI.fit_growth_interval(t[select], a[select], floor_abs=1e-8, min_efold=1., label="fixed initial linear window")
    return {k:result[k] for k in ("status","gamma","gamma_se","r2","efolds","reason","branch")}


def geometry_stage(p):
    rows = []
    for name in GEOMETRIES:
        status("geometry",name)
        bx = configure(name)
        A,_ = matrix(p,name,bx["bin_qhalf"])
        vals,vectors = np.linalg.eig(A)
        trans = H.NY*H.NZ
        candidates = np.flatnonzero(vals.imag < 0)
        j = max(candidates, key=lambda j:sum(abs(vectors[c*trans:(c+1)*trans,j].sum())**2 for c in (0,1)))
        frequency = float(-vals[j].imag/(2*np.pi))
        my = deterministic(p,"geometry_"+name,name,EPS,0,plane(p,name))
        series = np.fft.fft(my,axis=1)[:,bx["bin_qhalf"]]/bx["NX"]
        rows.append(dict(geometry=name, frequency_Hz=frequency, growth=fit(series,40e-9),
                         data_sha256=sha(OUT/("geometry_"+name+".npz"))))
        write("geometry_partial.json",dict(rows=rows,status="incomplete"))
    base = rows[0]
    for row in rows:
        row["frequency_difference_Hz"] = abs(row["frequency_Hz"]-base["frequency_Hz"])
        row["relative_growth_difference"] = (abs(row["growth"]["gamma"]/base["growth"]["gamma"]-1)
                                               if row["growth"]["status"] == base["growth"]["status"] == "ok" else None)
    passed = all(r["growth"]["status"] == "ok" and r["frequency_difference_Hz"] <= 3e6
                 and r["relative_growth_difference"] <= .05 for r in rows)
    write("geometry.json",dict(status="PASS" if passed else "HOLD",rows=rows,
                                scope="q/2 frequency and fixed-window growth, fixed B0, deterministic weak seed"))
    if not passed:
        raise RuntimeError("Geometry convergence criterion not met; dependent production stages stopped")


def coefficient(my, mode):
    # The spatial projection uses cell centres, not index-zero coordinates.
    s = np.fft.fft(my,axis=1)[:,mode]/my.shape[1]*np.exp(-1j*np.pi*mode/my.shape[1])
    t = np.arange(len(s))*DT
    mask = (t >= 30e-9)&(t < 40e-9)
    w = np.hanning(mask.sum())
    return np.sum(s[mask]*np.exp(2j*np.pi*3e9*t[mask])*w)/w.sum()


def coherence(a, rng, nnull=100000):
    weights = abs(a)**2
    if len(a) < 2 or weights.sum() <= 0:
        raise ValueError("A nonempty independent ensemble is required")
    observed = float(abs(np.sum(a*a))/weights.sum())
    count = 0
    for _ in range(nnull//1000):
        null = abs(np.exp(2j*np.pi*rng.random((1000,len(a))))@weights)/weights.sum()
        count += int(np.count_nonzero(null >= observed))
    return dict(C=observed, conditional_exceedance=(1+count)/(nnull+1),
                null_rms=float(np.linalg.norm(weights)/weights.sum()),
                effective_n=float(weights.sum()**2/np.sum(weights**2)), n=len(a),
                phase_rad=float(np.angle(1j*np.sum(a*a))),
                null_note="conditional on measured amplitudes and independent uniform pair phases")


def ensemble_stage(p):
    bx = configure("base")
    ensembles = {key:[] for key in ("MEL","OFF","MR")}
    for seed in SEEDS:
        initial,_ = H.make_seed(seed,bx["NX"],bx["dx"],1e-4,2*bx["q"],n_y=8,n_z=1)
        for channel in ensembles:
            label = "ensemble_%s_%d" % (channel,seed)
            status("ensemble",label)
            mr = H.MU0*H.YIG_LIT["MS"]**2/2 if channel == "MR" else 0.
            my = deterministic(p,label,"base",0. if channel == "OFF" else EPS,seed,initial,mr=mr,
                               kind="new independent random phase seed, amplitude 1e-4 per mode")
            ensembles[channel].append(coefficient(my,6))
    rng = np.random.default_rng(281901)
    rows = {key:coherence(np.asarray(v),rng) for key,v in ensembles.items()}
    boot = []
    a,b = np.asarray(ensembles["MEL"]),np.asarray(ensembles["OFF"])
    for _ in range(10000):
        j = rng.integers(0,len(a),len(a))
        boot.append(abs(np.sum(a[j]**2))/np.sum(abs(a[j])**2)-abs(np.sum(b[j]**2))/np.sum(abs(b[j])**2))
    ci = np.quantile(boot,[.025,.975]).tolist()
    write("ensemble.json",dict(rows=rows,delta_C_MEL_minus_OFF_bootstrap_95=ci,
                                state="complete", source_sha256=p["driver"],
                                scope="conditional deterministic initial-phase ensemble, not thermal or detector statistics",
                                supported=rows["MEL"]["conditional_exceedance"] < .01 and ci[0] > 0,
                                coefficients={k:[[z.real,z.imag] for z in v] for k,v in ensembles.items()}))


def floquet(p, strain):
    configure("base")
    modes = np.array([-30,-18,-6,6,18,30])
    ident = dict(driver=p["driver"], reference=p["reference"], strain=float(strain))
    path = OUT/("floquet_%.12e.npz" % strain)
    old = resume(path,ident)
    if bool(old.get("done",False)):
        return float(old["growth"]),old["initial"]
    blocks=[]
    H.DT_STEP=2e-13
    H.SEED_KIND="bandlimited_yflat_randphase"
    for mode in modes:
        A,info=SL.operator((4,64,0),abs(int(mode)))
        if mode < 0:
            A=A.conj()
        blocks.append(A+(p["B0"]-SL.B_REF)*SL.field_matrix(info["gamma"]))
    configure("base")
    J=SL.field_matrix(info["gamma"])
    diagonal=np.stack(blocks)
    n,d=len(modes),len(J)
    b=-2*H.YIG_LIT["B1"]/H.YIG_LIT["MS"]*strain
    def rhs(s,flat):
        Y=flat.reshape(n,d,n*d)
        out=diagonal@Y
        out[1:]+=(-.5j*b*np.exp(-2j*np.pi*s))*(J@Y[:-1])
        out[:-1]+=(.5j*b*np.exp(2j*np.pi*s))*(J@Y[1:])
        return (out/F).ravel()
    sol=solve_ivp(rhs,(0,1),np.eye(n*d,dtype=complex).ravel(),method="DOP853",rtol=1e-10,atol=1e-12,t_eval=[1])
    if not sol.success:
        raise RuntimeError(sol.message)
    vals,vec=np.linalg.eig(sol.y[:,-1].reshape(n*d,n*d))
    j=int(np.argmax(abs(vals)))
    v=vec[:,j].reshape(n,d)
    real_v=(v+v[::-1].conj())/2
    if np.linalg.norm(real_v) < .1*np.linalg.norm(v):
        real_v=(1j*v+(1j*v)[::-1].conj())/2
    bx=configure("base")
    phases=np.exp(2j*np.pi*modes[:,None]*(np.arange(bx["NX"])+.5)/bx["NX"])
    transverse=(real_v.T@phases).real.reshape(2,1,8,bx["NX"])
    transverse *= 1e-4/np.max(abs(transverse))
    initial=np.concatenate((np.sqrt(1-np.sum(transverse**2,axis=0))[None],transverse))
    growth=float(np.log(abs(vals[j]))*F)
    checkpoint(path,ident,done=True,growth=growth,initial=initial,multiplier=vals[j])
    return growth,initial


def threshold_stage(p):
    rows=[]
    for strain in p["threshold_strains"]:
        status("threshold",str(strain))
        predicted,initial=floquet(p,strain)
        duration=np.ceil(2/abs(predicted)/20e-9)*20e-9
        if duration > 600e-9:
            raise RuntimeError("Threshold duration cap exceeded; do not silently shorten")
        label="threshold_eps%.0e" % strain
        my=deterministic(p,label,"base",strain,0,initial,duration=duration,kind="predicted q/2-sector Floquet eigenvector, physical real projection")
        series=np.fft.fft(my,axis=1)[:,6]/my.shape[1]
        observed=fit(series,duration)
        rows.append(dict(strain=strain,duration_ns=duration*1e9,predicted_amplitude_rate=predicted,measured=observed))
        write("threshold_partial.json",dict(rows=rows,status="incomplete"))
    qualified=all(r["measured"]["status"]=="ok" and np.sign(r["predicted_amplitude_rate"])==np.sign(r["measured"]["gamma"])
                  and abs(r["measured"]["gamma"]/r["predicted_amplitude_rate"]-1)<.05 for r in rows)
    root=brentq(lambda eps:floquet(p,eps)[0],2e-5,5e-5,xtol=1e-10) if qualified else None
    write("threshold.json",dict(status="qhalf_bracket_supported" if qualified else "HOLD",rows=rows,
                                 predicted_qhalf_threshold=root,direct_bracket=[2e-5,5e-5] if qualified else None,
                                 scope=p["threshold_scope"]))
    if not qualified:
        raise RuntimeError("Growth-rate prediction/sign criterion failed")


def thermal_blocks(p,label,initial,temperature,duration,seed,*,strain=0.,alpha=None,block_records=100,stop_after=None):
    d=decl(p,"base",strain,seed,duration,"checkpointed thermal block stream",temp=temperature,alpha=alpha)
    H.DT_STEP=p["thermal"]["dt_step"]
    d["DT_STEP"]=H.DT_STEP
    nt=round(duration/DT)+1
    cs=H.conditions(d,source="thermal blocks",nt=nt,dt_rec=DT,block_records=block_records)
    cs.require_execution("thermal blocks",nx=d["box"]["NX"])
    d["initial_sha256"]=__import__("hashlib").sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()
    ident=dict(declaration=d,condition_stamp=cs.stamp(),block_records=block_records)
    path=OUT/(label+".npz")
    old=resume(path,ident)
    if bool(old.get("done",False)):
        if (old["my_xt"].shape!=(nt,d["box"]["NX"]) or abs(float(old["time"])-duration)>1e-18
                or int(old["next_step"]) != nt-1 or old["mag_state"].shape != initial.shape
                or not np.isfinite(old["my_xt"]).all() or not np.isfinite(old["mag_state"]).all()):
            raise RuntimeError("Invalid completed thermal state")
        return old
    w,m,meta=make_world(d,initial,cs)
    if not hasattr(m,"thermal_seed"):
        raise RuntimeError("A reproducible thermal RNG is required")
    my=np.zeros((nt,d["box"]["NX"]),np.float32)
    energy=np.zeros(nt,float)
    start=int(old.get("next_step",0))
    if old:
        if start%block_records or abs(float(old["time"])-start*DT)>1e-18:
            raise RuntimeError("Thermal resume must begin at a saved block boundary")
        pending_state=old["mag_state"]
        w.timesolver.time=float(old["time"])
        my[:start+1]=old["my_xt"]
        energy[:start+1]=old["mean_transverse_power"]
    else:
        pending_state=m.magnetization.eval()
        my[0]=H.sample_my(m)
        energy[0]=float((m.magnetization.eval()[1:]**2).sum(axis=0).mean())
    provenance=H.prov_kw(__file__,d)
    for block_start in range(start,nt-1,block_records):
        # Reassignment normalizes the field in the engine. Perform the same
        # operation once per block in uninterrupted AND resumed execution.
        m.magnetization=pending_state
        block=block_start//block_records
        rng_seed=int(np.random.SeedSequence([int(seed),int(block)]).generate_state(1,dtype=np.uint64)[0])
        m.thermal_seed=rng_seed
        end=min(block_start+block_records,nt-1)
        for i in range(block_start,end):
            w.timesolver.run(DT)
            state=m.magnetization.eval()
            my[i+1]=state[1].mean(axis=(0,1))
            energy[i+1]=float((state[1:]**2).sum(axis=0).mean())
        state=m.magnetization.eval()
        pending_state=state
        if not np.isfinite(state).all() or abs(float(w.timesolver.time)-end*DT)>1e-18:
            raise RuntimeError("Invalid thermal integration state")
        checkpoint(path,ident,done=end==nt-1,next_step=end,time=float(w.timesolver.time),
                   my_xt=my[:end+1],mean_transverse_power=energy[:end+1],mag_state=state,
                   block_seed=rng_seed,provenance=json.dumps(provenance),saw_meta=json.dumps(meta))
        print(label,"t=%.1f ns" % (end*DT*1e9),flush=True)
        if stop_after is not None and end>=stop_after and end<nt-1:
            return resume(path,ident)
    return resume(path,ident)


def thermal_stage(p):
    outputs=[]
    for seed in THERMAL_SEEDS:
        status("thermal",str(seed))
        bx=configure("base")
        initial=np.zeros((3,1,8,bx["NX"])); initial[0]=1
        warm=thermal_blocks(p,"warm_%d"%seed,initial,300.,10e-9,seed,alpha=.05)
        # Separate noise streams after the common equilibrated initial state.
        for channel,offset,strain in (("OFF",100000,0.),("MEL",200000,EPS)):
            thermal_blocks(p,"thermal_%s_%d"%(channel,seed),warm["mag_state"],300.,40e-9,
                           seed+offset,strain=strain)
        warm_power=warm["mean_transverse_power"]
        ratio=float(warm_power[-100:].mean()/warm_power[250:350].mean())
        outputs.append(dict(seed=seed,burn_late_to_middle_transverse_ratio=ratio))
    write("thermal_completion.json",dict(state="complete",initialization=outputs,
                                         note="Burn uses alpha=.05; observation uses original .0005. Stationarity is checked, not assumed."))
    thermal_analysis(p)


def thermal_analysis(p):
    bx=configure("base")
    values={c:[] for c in ("MEL","OFF")}
    for seed in THERMAL_SEEDS:
        for channel in values:
            path=OUT/("thermal_%s_%d.npz"%(channel,seed))
            with np.load(path,allow_pickle=False) as d:
                if not bool(d["done"]):
                    raise RuntimeError("Thermal ensemble incomplete")
                my=d["my_xt"].astype(float)
            k,f,S=SA.spectrum(my[1000:],bx["dx"],DT,window="hann",window_x="rect",detrend=False,shift=False)
            kmask=abs(k-bx["q"]/2)<=1.1*bx["dk"]
            fmask=(f>=2.5e9)&(f<=3.5e9)
            values[channel].append(float(np.sum(abs(S[np.ix_(fmask,kmask)])**2)))
    on,off=np.asarray(values["MEL"]),np.asarray(values["OFF"])
    rng=np.random.default_rng(34518)
    boot=[]
    for _ in range(10000):
        j=rng.integers(0,len(on),len(on))
        boot.append(on[j].mean()/off[j].mean())
    r=float(on.mean()/off.mean())
    # The simulation shares an initial thermal state for causal comparison.
    # Independent experimental exposures do not share that microscopic state.
    variance=float((np.var(on,ddof=1)+np.var(off,ddof=1))/off.mean()**2)
    paired_variance=float(np.var(on-off,ddof=1)/off.mean()**2)
    count_rows=[]
    for background_ratio in (0.,1.,10.,100.):
        required=(25*(r+1+2*background_ratio)/(r-1)**2 if r>1 else None)
        count_rows.append(dict(background_to_off_magnon_count=background_ratio,
                               required_off_magnon_counts_per_on_off_exposure_for_shot_noise_SNR5=required))
    write("visibility.json",dict(state="conditional_only",n=len(on),
                                  observable="my power; k bins 5,6,7; 2.5-3.5 GHz; 20-40ns Hann; not calibrated optical counts",
                                  gain_ratio=r,gain_ratio_paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
                                  independent_exposure_variance_in_off_mean_squared=variance,
                                  matched_simulation_difference_variance_in_off_mean_squared=paired_variance,
                                  count_requirements=count_rows,
                                  formulas=dict(shot_noise="SNR=(r-1)*sqrt(M/(r+1+2*b)) for equal on/off exposures",
                                                repeated_pulses="SNR=sqrt(N)*(r-1)*M/sqrt((r+1+2*b)*M+v*M^2)",
                                                time="t=M/R_off_magnon, with R requiring experimental calibration"),
                                  assumptions=["linear optical transfer selecting the stated my band", "independent Poisson counts conditional on intensity", "identically prepared independent pulses", "fixed optical background", "thermal sampling uncertainty is separate from photon shot noise"],
                                  absolute_SNR=None,experimental_observability_certified=False,raw_band_powers=values,
                                  input_sha256={"thermal_%s_%d"%(c,s):sha(OUT/("thermal_%s_%d.npz"%(c,s)))
                                                for s in THERMAL_SEEDS for c in values}))


def cpu_controls():
    rng=np.random.default_rng(818)
    n=12
    samples=3000
    angles=rng.uniform(0,2*np.pi,(samples,n))
    a=np.exp(1j*angles)
    cutoff=np.quantile(abs(np.mean(a*a,axis=1)),.99)
    b=np.exp(1.5)*a.real+1j*np.exp(-1.5)*a.imag
    power=float(np.mean(abs(np.sum(b*b,axis=1))/np.sum(abs(b)**2,axis=1)>cutoff))
    if power<.9:
        raise RuntimeError("Prospective ensemble control lacks power")
    nx=1400
    t=np.arange(2001)*DT
    x=(np.arange(nx)+.5)/nx
    # An unequal-amplitude reverse wave must not determine the forward coefficient.
    signal=np.cos(2*np.pi*(6*x[None,:]-3e9*t[:,None])+.4)+.01*np.cos(2*np.pi*(-6*x[None,:]-2.5e9*t[:,None]))
    c=coefficient(signal,6)
    if abs(c-.5*np.exp(.4j))>1e-5:
        raise RuntimeError("Positive temporal branch regression")
    return dict(ensemble_n=n,synthetic_null_99_cutoff=float(cutoff),squeezed_control_power=power,
                forward_coefficient_error=float(abs(c-.5*np.exp(.4j))),
                note="Power control assumes the stated independent initial-phase model, not experimental noise")


def preflight(p):
    controls=cpu_controls()
    bx=configure("base")
    initial=plane(p,"base")
    d=decl(p,"base",0.,41518,0.,"thermal-noise amplitude control",temp=300.)
    cs=H.conditions(d,source="thermal noise control",nt=2,dt_rec=DT,block_records=2)
    w,m,_=make_world(d,initial,cs)
    m.thermal_seed=41518
    draws=np.concatenate([m.thermal_noise.eval().astype(float).ravel() for _ in range(8)])
    gamma=float(m.gamma.eval().mean())
    expected=2*H.KB*300.*gamma*H.YIG_LIT["ALPHA"]/((1+H.YIG_LIT["ALPHA"]**2)*H.YIG_LIT["MS"]*bx["dx"]*H.CY*H.CZ)
    variance_ratio=float(np.var(draws)/expected)
    if abs(variance_ratio-1)>.03:
        raise RuntimeError("Engine thermal noise variance failed the declared FDT check")
    full=thermal_blocks(p,"thermal_smoke_full",initial,300.,.4e-9,41519,block_records=5)
    thermal_blocks(p,"thermal_smoke_split",initial,300.,.4e-9,41519,block_records=5,stop_after=5)
    split=thermal_blocks(p,"thermal_smoke_split",initial,300.,.4e-9,41519,block_records=5)
    error=float(abs(full["my_xt"]-split["my_xt"]).max())
    if error>1e-7:
        raise RuntimeError("Thermal block checkpoint replay failed")
    refusal=False
    try:
        thermal_blocks(p,"thermal_smoke_split",initial,300.,.4e-9,41520,block_records=5)
    except RuntimeError as exc:
        refusal="identity mismatch" in str(exc)
    if not refusal:
        raise RuntimeError("Changed thermal seed was not refused")
    write("preflight.json",dict(status="PASS",driver=p["driver"],plan=sha(OUT/"plan.json"),
                                 cpu=controls,thermal_noise_variance_ratio=variance_ratio,
                                 thermal_resume_max_abs=error,wrong_seed_refused=refusal))


def status(stage,case):
    write("status.json",dict(state="running",stage=stage,case=case,pid=os.getpid(),
                               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    print(stage,case,flush=True)


def run(p):
    pre=json.loads((OUT/"preflight.json").read_text())
    if pre["status"]!="PASS" or pre["driver"]!=p["driver"] or pre["plan"]!=sha(OUT/"plan.json"):
        raise RuntimeError("Matching preflight required")
    lock=OUT/"worker.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        geometry_stage(p)
        ensemble_stage(p)
        threshold_stage(p)
        thermal_stage(p)
        write("status.json",dict(state="complete",scope="bounded validation stages; consult each result and its limits"))
    except BaseException as exc:
        write("status.json",dict(state="stopped",error=repr(exc),pid=os.getpid()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("preflight","run","analyze"))
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        p=setup()
        if args.stage=="preflight":
            preflight(p)
        elif args.stage=="run":
            run(p)
        else:
            thermal_analysis(p)
