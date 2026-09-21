"""Apply the declared finite-window estimator to prediction and observation.

The prediction files predate completion of the direct runs and are not changed.
This comparison is not an onset certificate or a new fitting-window search.
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import growth_interval as GI
import saw_analysis as SA
from check_dispersion_reference import checkpoint, resume, sha


def estimate(series, plan):
    dt = plan["dt_rec"]
    length = round(plan["growth_segment_ns"] * 1e-9 / dt)
    stride = round(0.2e-9 / dt)
    times, amplitudes = [], []
    for start in range(0, len(series) - length + 1, stride):
        frequency, spectrum = SA.time_spectrum(
            series[start:start + length], dt, window="hann")
        amplitudes.append(abs(spectrum[np.argmin(abs(frequency - 3e9))])
                          / np.hanning(length).sum())
        times.append((start + (length - 1) / 2) * dt)
    times, amplitudes = np.asarray(times), np.asarray(amplitudes)
    lo, hi = np.asarray(plan["growth_analysis_window_ns"]) * 1e-9
    selected = (times >= lo) & (times <= hi)
    fit = GI.fit_growth_interval(times[selected], amplitudes[selected],
                                 floor_abs=1e-8, min_efold=1.0,
                                 label="declared 10-40 ns window")
    result = {key: fit[key] for key in
              ("status", "gamma", "gamma_se", "r2", "efolds", "reason", "branch")}
    result["descriptive_slope_per_s"] = float(np.polyfit(
        times[selected], np.log(amplitudes[selected]), 1)[0])
    return result


def main():
    out = ROOT / "direct"
    plan = json.loads((out / "plan.json").read_text())
    if plan["driver_sha256"] != sha(ROOT / "direct_check.py"):
        raise RuntimeError("Original driver changed; re-establish estimator equivalence")
    if plan["growth_analysis_window_ns"] != [10, 40]:
        raise RuntimeError("Do not change the declared analysis interval")
    rows = []
    for strain, seed in plan["jobs"]:
        prediction = ROOT / ("prediction_eps%.0e_seed%d.npz" % (strain, seed))
        observation = out / ("MEL_eps%.0e_seed%d.npz" % (strain, seed))
        with np.load(prediction, allow_pickle=False) as p, np.load(observation, allow_pickle=False) as d:
            pi = json.loads(str(p["identity"]))
            declaration = json.loads(str(d["run_manifest"]))
            if (not bool(p["done"]) or not bool(d["done"])
                    or pi["script_sha256"] != sha(ROOT / "linear_trace.py")
                    or pi["reference_sha256"] != sha(ROOT / "spatial_linear.json")
                    or pi["strain"] != strain or pi["seed"] != seed
                    or pi["dt_record"] != plan["dt_rec"]
                    or pi["amplitude"] != plan["seed_amplitude"]):
                raise RuntimeError("Complete, matching prediction and data are required")
            expected = dict(eps0=strain, rng_seed=seed, DT_REC=plan["dt_rec"],
                            DT_STEP=plan["dt_step"], B0=plan["B0"],
                            a_seed=plan["seed_amplitude"], material=plan["material"],
                            driver_sha256=plan["driver_sha256"],
                            reference_sha256=pi["reference_sha256"])
            if any(declaration.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Observed run declaration differs from the analysis plan")
            if len(p["t"]) != len(d["my_xt"]):
                raise RuntimeError("Prediction and observation durations differ")
            reference = p["my_modes"][:, 3].copy()
            my = d["my_xt"].astype(float)
            if np.max(abs(my)) > plan["max_linear_my"]:
                raise RuntimeError("Declared small-angle bound exceeded")
            measured = np.fft.fft(my, axis=1)[:, 6] / my.shape[1]
        identity = dict(source_sha256=sha(__file__), plan_sha256=sha(out / "plan.json"),
                        growth_source_sha256=sha(GI.__file__),
                        spectrum_source_sha256=sha(SA.__file__),
                        prediction_sha256=sha(prediction), observation_sha256=sha(observation))
        cache = out / ("growth_compare_eps%.0e_seed%d.npz" % (strain, seed))
        previous = resume(cache, identity)
        if bool(previous.get("done", False)):
            row = json.loads(str(previous["result_json"]))
        else:
            row = dict(strain=strain, seed=seed,
                       predicted_finite_window=estimate(reference, plan),
                       observed_finite_window=estimate(measured, plan),
                       identity=identity)
            checkpoint(cache, identity, done=True, result_json=json.dumps(row))
        rows.append(row)
    report = dict(rows=rows, scope="same estimator, fixed window, no prediction adjustment",
                  note="Regression standard errors are not independent statistical or physical error budgets. HELD rates cannot enter a threshold bracket.")
    dest = out / "growth_prediction_comparison.json"
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(dest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
