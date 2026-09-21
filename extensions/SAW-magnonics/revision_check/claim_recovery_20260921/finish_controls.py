"""Record completion without changing the frozen producer or its verdicts."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "controls"
sys.path.insert(0, str(ROOT.parent))
import saw_analysis as SA
from check_dispersion_reference import checkpoint, resume, sha


def strict(value):
    if isinstance(value, dict):
        return {k: strict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    analysis = json.loads((OUT / "analysis.json").read_text())
    plan = json.loads((OUT / "plan.json").read_text())
    source = sha(ROOT / "control_suite.py")
    if (analysis["source_sha256"] != source or plan["source_sha256"] != source
            or analysis["plan_sha256"] != sha(OUT / "plan.json")):
        raise RuntimeError("Frozen source/plan mismatch")
    names = {c["name"] for c in plan["cases"]}
    if {r["case"] for r in analysis["rows"]} != names or len(analysis["rows"]) != len(names):
        raise RuntimeError("Missing or duplicated control")
    checks = []
    for row in analysis["rows"]:
        path = OUT / (row["case"] + ".npz")
        if sha(path) != row["data_sha256"]:
            raise RuntimeError("Analyzed data changed")
        with np.load(path, allow_pickle=False) as d:
            declaration = json.loads(str(d["run_manifest"]))
            if (not bool(d["done"]) or int(d["next_index"]) != 2001
                    or int(d["nt_total"]) != 2001
                    or abs(float(d["t_state"]) - plan["duration_s"]) > 1e-18
                    or not np.isfinite(d["my_xt"]).all()
                    or not np.isfinite(d["mag_state"]).all()
                    or declaration["source_sha256"] != source):
                raise RuntimeError("Incomplete or inconsistent final state")
            checks.append(dict(case=row["case"], data_sha256=sha(path),
                               final_time_s=float(d["t_state"]), records=2001,
                               run_identity=str(d["run_identity"])))

    baseline = ROOT / "direct/MEL_eps2e-04_seed20260917.npz"
    if sha(baseline) != analysis["baseline_sha256"]:
        raise RuntimeError("Baseline changed")
    identity = dict(source_sha256=sha(__file__), baseline_sha256=sha(baseline),
                    spectrum_source_sha256=sha(SA.__file__),
                    control_analysis_sha256=sha(OUT / "analysis.json"))
    cache = OUT / "completion_baseline.npz"
    saved = resume(cache, identity)
    if bool(saved.get("done", False)):
        shares = json.loads(str(saved["shares_json"]))
    else:
        with np.load(baseline, allow_pickle=False) as d:
            if not bool(d["done"]):
                raise RuntimeError("Baseline is unfinished")
            my = d["my_xt"].astype(float)
        k, f, spectrum = SA.spectrum(my[500:], plan["box"]["dx"], plan["dt_record"],
                                     window="hann", window_x="rect", detrend=False, shift=False)
        power = (abs(spectrum[(f >= 2.5e9) & (f <= 3.5e9)])**2).sum(axis=0)
        shares = dict(positive=float(power[k > 0].sum()/power.sum()),
                      negative=float(power[k < 0].sum()/power.sum()),
                      zero=float(power[k == 0].sum()/power.sum()))
        checkpoint(cache, identity, done=True, shares_json=json.dumps(shares))
    result = dict(state="complete", scope="four matched deterministic controls, not full pairing certification",
                  source_sha256=sha(__file__), frozen_producer_sha256=source,
                  evidence_sha256=sha(OUT / "analysis.json"), raw_completion=checks,
                  forward_MEL_baseline_directional_power=shares, rows=strict(analysis["rows"]),
                  null_note="Null growth values mean HELD, not zero growth or a certified negative point.",
                  limitations=analysis["limitations"])
    destination = OUT / "completion.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(dict(state=result["state"], controls=len(checks),
                          forward_MEL_directional_power=shares), indent=2))


if __name__ == "__main__":
    main()
