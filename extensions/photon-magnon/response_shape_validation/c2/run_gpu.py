"""Generate fresh nonlinear native-Gilbert GPU free-response records.

Only the installed mumaxplus import is used. Build this checkout in DOUBLE
precision first; there are no binary search paths or bundled simulation data.
Generated public observations and validator-only model inputs are separated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

U = 2 * np.pi * 1e9
GAMMA_G = 1.7595e11
G0, O0 = 1.561154518256, 261.310962155227
MS, LAT = 624000.0, 3.13e-10
CELL = (40e-9, 40e-9, 10e-9)
CASE_R = {"A": 0.084287616, "B": 0.126, "C": 0.99}
SOURCE = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def waveform(t, meta):
    x = (t - meta["start_s"]) / meta["duration_s"]
    if not 0 < x < 1:
        return 0.0
    return (meta["amplitude_T"] * math.sin(math.pi * x) ** 4
            * math.cos(2 * math.pi * meta["carrier_Hz"] * (t - meta["start_s"])))


def physical_case(r):
    contrast = (1 - r * r) / (1 + r * r)
    b = np.sqrt(O0 * O0 + G0 * G0 * contrast * contrast)
    alpha = 2 * r * G0 / ((1 + r * r) * b)
    eta = 1 / (1 + alpha * alpha)
    h = b / eta * U / GAMMA_G
    return {"r": r, "BE_T": h * (1 / r - r) / 2, "BA_T": h * r,
            "alpha": alpha, "gamma_rad_s_T": GAMMA_G,
            "mu_s_J_T": MS * np.prod(CELL), "Gamma_bar": G0, "Omega_bar": O0}


def record_plan(group, cid):
    """(variant, timestep, amplitude factor, record kinds, sample count mode)."""
    all_kinds = ("seedA", "seedB", "pulse")
    if group == "final":
        plan = [("halfstep", 25e-15, 1.0, all_kinds, True),
                ("quarterstep", 12.5e-15, 1.0, all_kinds, True)]
        if cid in ("A", "B"):
            plan += [("quarter_halfamp", 12.5e-15, 0.5, ("pulse",), True),
                     ("eighthstep", 6.25e-15, 1.0, ("seedA", "seedB"), True),
                     ("eighthseedhalf", 6.25e-15, 0.5, ("seedA", "seedB"), True)]
        return plan
    if group == "base":
        return [("base", 50e-15, 1.0, all_kinds, False)]
    if group == "refine":
        return [] if cid == "C" else [
            ("halfstep", 25e-15, 1.0, all_kinds, False),
            ("halfamp", 50e-15, 0.5, ("pulse",), False)]
    if group == "precision25":
        return [("halfstep", 25e-15, 1.0, all_kinds, True)]
    if group == "precision12":
        return [("quarterstep", 12.5e-15, 1.0, all_kinds, True)] + (
            [("quarter_halfamp", 12.5e-15, 0.5, ("pulse",), True)] if cid != "C" else [])
    if cid == "C":
        return []
    if group == "precision6":
        return [("eighthstep", 6.25e-15, 1.0, all_kinds, True),
                ("eighth_halfamp", 6.25e-15, 0.5, ("pulse",), True)]
    return [("eighthseedhalf", 6.25e-15, 0.5, ("seedA", "seedB"), True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Common run root; creates response_shape_validation/c2 below it.")
    parser.add_argument("--group", default="final", choices=("final", "base", "refine", "precision25", "precision12", "precision6", "precision_seedamp"))
    parser.add_argument("--case", choices=tuple(CASE_R))
    parser.add_argument("--only", choices=("all", "free", "pulse"), default="all")
    parser.add_argument("--check-runtime", action="store_true", help="Check DOUBLE import and required GPU bindings without generating records.")
    parser.add_argument("--expected-module-sha256", help="Optional caller-supplied digest of the newly built native binary.")
    args = parser.parse_args()

    # Respect an explicit precision setting, then reject non-DOUBLE imports.
    os.environ.setdefault("MUMAXPLUS_FP_PRECISION", "DOUBLE")
    import mumaxplus as mp
    if mp.FP_PRECISION != "DOUBLE":
        raise RuntimeError("C2 requires a DOUBLE build. Set MUMAXPLUS_FP_PRECISION=DOUBLE before running.")
    cpp = mp._cpp
    if args.expected_module_sha256 and sha(cpp.__file__) != args.expected_module_sha256.lower():
        raise RuntimeError("Loaded native binary does not match --expected-module-sha256.")
    world = mp.World(cellsize=CELL)
    afm = mp.Antiferromagnet(world, mp.Grid((1, 1, 1)))
    required = ("enable_cavity_afm", "enable_aux_mode", "cavity_energy_field")
    if not all(hasattr(afm._impl, key) for key in required) or not hasattr(world._impl, "reset_timesolver_equations"):
        raise RuntimeError("Required extension bindings are absent. Build and install this checkout first.")
    del afm, world

    here = args.output.resolve() / "response_shape_validation" / "c2"
    public, private = here / "public", here / "private"
    for folder in (public, private, here / "source_versions", here / "runtime"):
        folder.mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    (here / "source_versions" / f"run_gpu_{source_hash}.py").write_bytes(SOURCE.read_bytes())
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "precision": mp.FP_PRECISION, "module_path": str(Path(cpp.__file__).resolve()),
                  "module_sha256": sha(cpp.__file__), "wrapper_path": str(Path(mp.__file__).resolve()),
                  "wrapper_sha256": sha(mp.__file__), "source_sha256": source_hash,
                  "python": sys.version, "numpy": np.__version__, "required_bindings": list(required),
                  "runtime_paths_are_local_outputs_only": True}
    runtime_id = hashlib.sha256(json.dumps({k: provenance[k] for k in (
        "precision", "module_sha256", "wrapper_sha256", "source_sha256")}, sort_keys=True).encode()).hexdigest()
    write_json(here / "runtime" / f"{runtime_id}.json", provenance)
    if args.check_runtime:
        print(json.dumps({"runtime_checked": True, "precision": mp.FP_PRECISION,
                          "module_sha256": provenance["module_sha256"]}), flush=True)
        return
    protocol = {"scope": "Fresh DOUBLE native-Gilbert nonlinear free-response calibration; no coupled data.",
                "sample_interval_s": 200e-15, "final_free_duration_s": 40e-12,
                "final_pulse_duration_s": 80e-12, "pilot_duration_s": 200e-12,
                "final_pole_windows_s": [[0, 40e-12], [5e-12, 32e-12], [10e-12, 40e-12]],
                "final_pulse_windows_s": [[0, 80e-12], [2e-12, 64e-12], [10e-12, 80e-12]],
                "public_schema": "t_s,m6,Mx,B_T and waveform metadata; no r,BE,BA,alpha or analytic coefficients.",
                "numerical_target": "Each observable error or sampled numerical envelope <0.1 of its shape effect; not an experimental criterion.",
                "source_sha256": source_hash}
    write_json(here / "protocol.json", protocol)
    truth_file = private / "truth_by_case.json"
    truth = json.loads(truth_file.read_text(encoding="utf-8")) if truth_file.exists() else {
        "scope": "Generated model inputs for post-freeze validation only; never a fit/predict input.", "cases": {}}
    for cid, r in CASE_R.items():
        if args.case and cid != args.case:
            continue
        tr = physical_case(r)
        if cid in truth["cases"] and truth["cases"][cid] != tr:
            raise RuntimeError(f"Case {cid} already exists with a different model setup; use a fresh output root.")
        truth["cases"][cid] = tr
        write_json(truth_file, truth)
        for variant, step, factor, kinds, precision in record_plan(args.group, cid):
            for kind in kinds:
                if (args.only == "free" and kind == "pulse") or (args.only == "pulse" and kind != "pulse"):
                    continue
                name = f"{cid}_{variant}_{kind}"
                meta_path, raw_path = public / f"{name}.json", public / f"{name}.npz"
                private_path = private / f"{name}.json"
                if private_path.exists():
                    saved = json.loads(private_path.read_text(encoding="utf-8"))
                    if saved["runtime_id"] != runtime_id or not meta_path.exists() or not raw_path.exists() or sha(raw_path) != saved["raw_sha256"]:
                        raise RuntimeError(f"Existing {name} has incompatible provenance or missing/changed data; use a fresh output root.")
                    print("Already completed " + name, flush=True)
                    continue
                world = mp.World(cellsize=CELL)
                afm = mp.Antiferromagnet(world, mp.Grid((1, 1, 1)))
                for sub in (afm.sub1, afm.sub2):
                    sub.msat = MS
                    sub.gamma = GAMMA_G
                    sub.ku1 = tr["BA_T"] * MS / 2
                    sub.anisU = (0, 0, 1)
                    sub.alpha = tr["alpha"]
                    sub.aex = 0.0
                    sub.enable_demag = False
                    sub.enable_openbc = True
                afm.afmex_cell = -tr["BE_T"] * MS * LAT ** 2 / 4
                afm.afmex_nn, afm.latcon = 0.0, LAT
                world.bias_magnetic_field = (0, 0, 0)
                afm._impl.enable_cavity_afm = False
                afm._impl.enable_aux_mode = False
                initial = np.zeros(4)
                if kind == "seedA":
                    initial[0] = 1e-6 * factor
                if kind == "seedB":
                    initial[2] = 1e-6 * factor
                afm.sub1.magnetization = (*initial[:2], np.sqrt(1 - initial[:2] @ initial[:2]))
                afm.sub2.magnetization = (*initial[2:], -np.sqrt(1 - initial[2:] @ initial[2:]))
                wave = {"type": "finite_sin4_cos", "start_s": 2e-12, "duration_s": 40e-12,
                        "carrier_Hz": 261e9, "amplitude_T": 2e-6 * factor}
                if kind == "pulse":
                    # The identical physical field enters each sublattice's
                    # native effective field, hence precession AND damping.
                    for sub in (afm.sub1, afm.sub2):
                        sub.bias_magnetic_field.add_time_term(lambda t, meta=wave: (waveform(t, meta), 0.0, 0.0))
                world._impl.reset_timesolver_equations()
                world.timesolver.adaptive_timestep = False
                world.timesolver.timestep = step
                sample = 200e-15
                count = int(round(sample / step))
                sample_count = (401 if kind == "pulse" else 201) if precision else 1001
                t = np.arange(sample_count) * sample
                m = np.empty((len(t), 6))
                start = time.perf_counter()
                for i in range(len(t)):
                    if i:
                        world.timesolver.steps(count)
                    m[i] = [*afm.sub1.magnetization.average(), *afm.sub2.magnetization.average()]
                    if i % 200 == 0:
                        write_json(here / "progress.json", {"run": name, "sample": i, "samples": len(t), "wall_s": time.perf_counter() - start})
                field = np.array([waveform(ti, wave) if kind == "pulse" else 0.0 for ti in t])
                np.savez_compressed(raw_path, t_s=t, m6=m, Mx=m[:, 0] + m[:, 3], B_T=field)
                raw_hash = sha(raw_path)
                write_json(meta_path, {"case_id": cid, "variant": variant, "record_type": kind,
                    "time_units": "s", "field_units": "T", "magnetization": "mA,mB unit vectors; Mx=mAx+mBx",
                    "raw_file": raw_path.name, "raw_sha256": raw_hash, "waveform": wave if kind == "pulse" else None,
                    "integration_step_s": step, "sample_interval_s": sample,
                    "initial_condition": "physical sublattice seed" if kind != "pulse" else "collinear equilibrium, zero B and response before pulse"})
                wall_s = time.perf_counter() - start
                write_json(private_path, {**tr, "case_id": cid, "runtime_id": runtime_id,
                    "precision": mp.FP_PRECISION, "source_sha256": source_hash, "raw_sha256": raw_hash,
                    "native_Gilbert": True, "cavity_enabled": False, "external_field_in_total_Gilbert_field": True,
                    "seed_amplitude": float(np.linalg.norm(initial)), "grid": [1, 1, 1], "cell_m": list(CELL),
                    "wall_s": wall_s, "max_transverse_sublattice_radius": float(max(
                        np.hypot(m[:, 0], m[:, 1]).max(), np.hypot(m[:, 3], m[:, 4]).max()))})
                print(json.dumps({"completed": name, "wall_s": wall_s}), flush=True)


if __name__ == "__main__":
    main()
