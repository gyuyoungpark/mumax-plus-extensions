"""Predeclared thermal-ensemble phase endpoint, using the frozen estimator.

Call after all thermal trajectories finish. This adds no new integration,
frequency/window search, or altered selection rule.
"""

import json
import numpy as np
import validation_campaign as V
from check_dispersion_reference import checkpoint, resume, sha


def main():
    p=json.loads((V.OUT/"plan.json").read_text())
    if p["driver"]!=sha(V.__file__):
        raise RuntimeError("Frozen estimator changed")
    values={"MEL":[],"OFF":[]}
    inputs={}
    for seed in p["thermal_seeds"]:
        for channel in values:
            path=V.OUT/("thermal_%s_%d.npz"%(channel,seed))
            identity=dict(source=sha(__file__),estimator=p["driver"],input_sha256=sha(path))
            inputs[path.name]=identity["input_sha256"]
            cache=V.OUT/("phase_%s_%d.npz"%(channel,seed))
            old=resume(cache,identity)
            if bool(old.get("done",False)):
                value=complex(old["coefficient"])
            else:
                with np.load(path,allow_pickle=False) as d:
                    record=json.loads(str(d["identity"]))["declaration"]
                    if (not bool(d["done"]) or d["my_xt"].shape!=(2001,1400)
                            or abs(float(d["time"])-40e-9)>1e-18
                            or record["source_sha256"]!=p["driver"]
                            or record["temperature"]!=300.
                            or record["eps0"]!=(V.EPS if channel=="MEL" else 0.)
                            or not np.isfinite(d["my_xt"]).all()):
                        raise RuntimeError("Incomplete or foreign thermal ensemble member")
                    value=V.coefficient(d["my_xt"].astype(float),6)
                checkpoint(cache,identity,done=True,coefficient=value)
            values[channel].append(value)
    rng=np.random.default_rng(388155)
    rows={k:V.coherence(np.asarray(a),rng) for k,a in values.items()}
    a,b=map(np.asarray,(values["MEL"],values["OFF"]))
    delta=[]
    for _ in range(10000):
        j=rng.integers(0,len(a),len(a))
        delta.append(abs(np.sum(a[j]**2))/np.sum(abs(a[j])**2)-abs(np.sum(b[j]**2))/np.sum(abs(b[j])**2))
    ci=np.quantile(delta,[.025,.975]).tolist()
    V.write("thermal_phase.json",dict(state="complete",rows=rows,
                                       paired_bootstrap_delta_C_95=ci,input_sha256=inputs,
                                       source_sha256=sha(__file__),estimator_sha256=p["driver"],
                                       supported=rows["MEL"]["conditional_exceedance"]<.01 and ci[0]>0,
                                       scope="300 K micromagnetic ensemble, fixed +q/2 and 30-40ns window; not an optical phase calibration",
                                       qualifications=["phase-null model is conditional on measured amplitudes", "shared initial state preserved in bootstrap pairs", "thermal preparation stationarity requires separate review", "not evidence for two distinct spatial modes"]))
    print(json.dumps(V.safe(rows),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
