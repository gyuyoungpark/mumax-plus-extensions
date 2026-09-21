"""Emulate the SINGLE build's fixed-step clock, then test stored k=0 traces.

The synthetic oscillator is explicitly a clock diagnostic, not new magnetic
simulation data. Engine time is checked with the actual empty-world solver.
No dispersion certificate or manuscript result is overwritten.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from check_dispersion_reference import checkpoint, resume, sha

HERE = Path(__file__).resolve().parent
RC = HERE.parent
sys.path.insert(0, str(RC))
import saw_analysis as SA

NT, DT_REC, DT_STEP = 20001, 5e-12, 2e-13


def clock_trace():
    from mumaxplus import World, _cpp
    identity = dict(script=sha(__file__), engine=sha(_cpp.__file__),
                    nt=NT, dt_rec=DT_REC, dt_step=DT_STEP,
                    solver=sha(RC.parents[2] / "src/core/timesolver.cpp"),
                    stepper=sha(RC.parents[2] / "src/core/rungekutta.cpp"))
    path = HERE / "single_clock.npz"
    saved = resume(path, identity)
    clocks = saved.get("clocks", np.zeros(NT))
    integrated = saved.get("integrated", np.zeros(NT))
    actual = saved.get("actual", np.zeros(NT))
    index = int(saved.get("next_index", 1))
    world = World((5e-9, 1e-8, 2e-8))
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = DT_STEP
    world.timesolver.time = clocks[index-1]
    step, duration = np.float32(DT_STEP), np.float32(DT_REC)
    t, tau = np.float32(clocks[index-1]), integrated[index-1]
    for i in range(index, NT):
        stop = np.float32(t + duration)
        while t < np.float32(stop - step):
            tau += float(step)
            t = np.float32(t + step)
        last_step = np.float32(stop - t)
        tau += float(last_step)
        t = np.float32(t + last_step)
        clocks[i], integrated[i] = float(t), tau
        world.timesolver.run(DT_REC)
        actual[i] = world.timesolver.time
        if i % 500 == 0 or i == NT - 1:
            checkpoint(path, identity, clocks=clocks, integrated=integrated,
                       actual=actual, next_index=i+1, done=i == NT-1)
    if not np.array_equal(actual, clocks):
        raise AssertionError("Clock emulation differs from the actual engine")
    return clocks, integrated, actual


def peak(signal):
    freq, spectrum = SA.time_spectrum(signal, DT_REC, window="hann")
    good = np.flatnonzero((freq > 2.5e9) & (freq < 3.5e9))
    j = good[np.argmax(abs(spectrum[good]))]
    y0, y1, y2 = np.log(abs(spectrum[j-1:j+2]))
    delta = 0.5*(y0-y2)/(y0-2*y1+y2)
    if abs(delta) > 0.5:
        raise AssertionError("peak interpolation leaves its bracket")
    return float(freq[j] + delta*(freq[1]-freq[0]))


def frequency_fit(signal, times, guess, decay):
    def profile(f_GHz, return_details=False):
        phase = 2*np.pi*f_GHz*1e9*times
        envelope = np.exp(decay*times)
        basis = np.column_stack((envelope*np.cos(phase), envelope*np.sin(phase),
                                 np.ones(times.size)))
        coefficients = np.linalg.lstsq(basis, signal, rcond=None)[0]
        resid = signal - basis @ coefficients
        mse = float(np.mean(resid**2)/np.var(signal))
        return (mse, coefficients.tolist()) if return_details else mse
    # The objective has aliases about 1/T apart; search a dense local grid first.
    grid = np.linspace(guess/1e9 - .03, guess/1e9 + .03, 301)
    errors = [profile(f) for f in grid]
    j = int(np.argmin(errors))
    if j in (0, len(grid)-1):
        raise AssertionError("frequency search hit boundary")
    fit = minimize_scalar(profile, bounds=(grid[j-1], grid[j+1]),
                          method="bounded", options={"xatol": 1e-12})
    return dict(frequency_Hz=float(fit.x*1e9), relative_mse=float(fit.fun),
                purpose="descriptive fit, not held-out model selection")


def main():
    clocks, tau, actual = clock_trace()
    static_path = HERE / "dispersion_reference.json"
    static = json.loads(static_path.read_text(encoding="utf-8"))
    rows = []
    for it, record in enumerate(static["recorded_dispersion"]):
        ref = next(row for row in static["results"]
                   if row["pbc"] == [2,2,0] and row["B0"] == record["B0"]
                   and row["perturbation"] == 1e-4) if it == 1 else static["results"][0]
        file = RC / "runs/out/G4" / f"G4_disp_it{it}_B{record['B0']*1e3:.4f}mT.npz"
        with np.load(file, allow_pickle=False) as data:
            signal = data["my_xt"].astype(float).mean(axis=1)
            if float(data["t_state"]) != clocks[-1]:
                raise AssertionError("saved final clock does not match reconstruction")
        f0, decay = ref["projected_llg_frequency_Hz"], ref["projected_llg_decay_per_s"]
        synthetic = np.exp(decay*tau)*np.cos(2*np.pi*f0*tau)
        rows.append(dict(input_path=str(file), input_sha256=sha(file),
                         B0=record["B0"], static_llg_frequency_Hz=f0,
                         observed_fft_Hz=peak(signal),
                         synthetic_clock_diagnostic_fft_Hz=peak(synthetic),
                         fit_nominal_time=frequency_fit(signal, np.arange(NT)*DT_REC, f0, decay),
                         fit_integrated_steps=frequency_fit(signal, tau, f0, decay)))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(script_sha256=sha(__file__), static_result_sha256=sha(static_path),
                  actual_engine_clock_exact_match=bool(np.array_equal(actual, clocks)),
                  nominal_final_time_s=(NT-1)*DT_REC, engine_final_time_s=float(clocks[-1]),
                  sum_applied_timesteps_s=float(tau[-1]), rows=rows,
                  caveat="pump-off autonomous diagnostic only; cannot correct a driven run by relabelling its time axis")
    tmp = HERE / "solver_clock.tmp"
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, HERE / "solver_clock.json")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
