"""Freeze a public-only C2 fit before reading generated model truth.

The frozen bytes cannot be overwritten. --resume-validation verifies their
digest and the fitter digest before repeating post-freeze validation.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Common run root containing response_shape_validation/c2.")
    parser.add_argument("--resume-validation", action="store_true")
    args = parser.parse_args()
    here = args.output.resolve() / "response_shape_validation" / "c2"
    fitter = Path(__file__).with_name("fit_public.py")
    public_path = here / "fit" / "calibration_public.json"
    data = json.loads(public_path.read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in data["cases"]}
    require(data.get("final_records_required") is True, "Partial development fits cannot be frozen.")
    require(set(cases) == set("ABC"), "The final calibration requires all three cases.")
    require(data["analyzer_sha256"] == sha(fitter), "Fitter source changed since calibration; refit first.")
    for cid in ("A", "B"):
        case = cases[cid]
        require(case["chosen_pole_variant"] == "eighthseedhalf", f"Missing final half-seed pole fit for {cid}.")
        expected_half = {"quarterstep": "quarter_halfamp", "eighthstep": "eighth_halfamp"}.get(case["chosen_variant"])
        require(expected_half and expected_half in case["envelope_variants"], f"Missing final half-drive comparison for {cid}.")
    require(cases["C"]["chosen_pole_variant"] == cases["C"]["chosen_variant"] == "quarterstep", "Missing final fine pair for C.")
    require(data.get("local_shape_resolution", {}).get("pass") is True, "Public-data rho variation target failed.")
    public_dir = here / "public"
    # Recheck every observation actually used. No private directory is opened
    # until the frozen result and its digest have been written below.
    for record in data["public_read_manifest"]:
        path = (public_dir / record["path"]).resolve()
        require(path.is_relative_to(public_dir.resolve()), "Public read manifest escapes its directory.")
        require(sha(path) == record["sha256"], "Public observation changed after fitting.")
    frozen_path = here / "calibration_frozen.json"
    provenance_path = here / "calibration_frozen_provenance.json"
    if args.resume_validation:
        freeze = json.loads(provenance_path.read_text(encoding="utf-8"))
        frozen_hash = sha(frozen_path)
        require(frozen_hash == freeze["sha256"] == sha(public_path), "Frozen calibration no longer matches its source or provenance.")
        require(freeze["fitter_source_sha256"] == sha(fitter), "Fitter source changed after freezing.")
    else:
        require(not frozen_path.exists(), "Refuse to overwrite frozen calibration; use --resume-validation.")
        with frozen_path.open("xb") as handle:
            handle.write(public_path.read_bytes())
        frozen_hash = sha(frozen_path)
        freeze = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "file": frozen_path.name,
                  "sha256": frozen_hash, "fitter_source_sha256": sha(fitter),
                  "scope": "Only free GPU observations and calibrated physical Bx waveform enter the frozen fit. No coupled data or material truth enter fitting."}
        provenance_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    # This is the first point at which validator-only model inputs are read.
    truth_path = here / "private" / "truth_by_case.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["cases"]
    U = data["unit_rate_rad_s"]
    rows = []
    for cid, fit in cases.items():
        tr = truth[cid]
        r, g = tr["r"], tr["gamma_rad_s_T"]
        eta = 1 / (1 + tr["alpha"] ** 2)
        wm = g * np.sqrt(tr["BA_T"] * (2 * tr["BE_T"] + tr["BA_T"])) / U
        a, b = eta * tr["alpha"] / r, eta * wm
        A, B = 2 * g * r * a / U, 2 * g * r * b / U
        expected = {"Gamma_bar": tr["Gamma_bar"], "Omega_bar": tr["Omega_bar"],
                    "A_tilde_per_T": A, "B_tilde_per_T": B, "rho_s": A / B / U}
        errors = {key: abs(fit[key] - expected[key]) for key in expected}
        rows.append({"case_id": cid, "truth_r_disclosed": r, "expected_after_freeze": expected,
                     "fit": {key: fit[key] for key in expected}, "absolute_errors": errors,
                     "Gamma_error_Hz": errors["Gamma_bar"] * 1e9, "Omega_error_Hz": errors["Omega_bar"] * 1e9,
                     "rho_relative_error": errors["rho_s"] / abs(expected["rho_s"]),
                     "numerical_variation_envelope": fit["numerical_variation_envelope"],
                     "truth_inside_numerical_variation_envelope": {
                         key: bool(errors[key] <= fit["numerical_variation_envelope"][key]) for key in expected},
                     "rho_error_target_pass": bool(errors["rho_s"] < .1 * data["local_shape_resolution"]["delta_rho_s"])})
    records = []
    metadata_names = sorted({r["path"] for r in data["public_read_manifest"] if r["path"].endswith(".json")})
    for name in metadata_names:
        meta_path = public_dir / name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_path = public_dir / meta["raw_file"]
        private_path = here / "private" / name
        private_meta = json.loads(private_path.read_text(encoding="utf-8"))
        require(sha(raw_path) == meta["raw_sha256"] == private_meta["raw_sha256"], "Record digest mismatch.")
        source = here / "source_versions" / f"run_gpu_{private_meta['source_sha256']}.py"
        runtime_path = here / "runtime" / f"{private_meta['runtime_id']}.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        require(sha(source) == private_meta["source_sha256"] == runtime["source_sha256"], "Executed source snapshot mismatch.")
        require(runtime["precision"] == private_meta["precision"] == "DOUBLE", "Non-DOUBLE record found.")
        records.append({"public_metadata": name, "raw_sha256": sha(raw_path),
                        "private_metadata_sha256": sha(private_path), "runtime_sha256": sha(runtime_path),
                        "executed_source_sha256": sha(source), "module_sha256": runtime["module_sha256"],
                        "wall_s": private_meta["wall_s"]})
    result = {"frozen_calibration_sha256": frozen_hash, "frozen_utc": freeze["frozen_utc"],
              "truth_validation_utc": datetime.now(timezone.utc).isoformat(), "private_truth_sha256": sha(truth_path),
              "rows": rows, "local_shape_resolution": data["local_shape_resolution"],
              "GPU_records": len(records), "records": records, "total_GPU_wall_s": sum(r["wall_s"] for r in records),
              "rho_error_target_pass": all(r["rho_error_target_pass"] for r in rows),
              "scope": "Numerical calibration of a specified theoretical model using actual GPU records, not experimental inference.",
              "limitations": ["Input field and physical Mx readout normalization and phase are known.",
                              "No instrumental noise, unknown gain or material uncertainty is inferred.",
                              "Finite numerical variation envelopes are not rigorous interval bounds or confidence intervals.",
                              "C3 EP and response errors require separate comparison with each shape effect."]}
    require(sha(frozen_path) == frozen_hash, "Frozen bytes changed during validation.")
    (here / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_sha256": frozen_hash, "GPU_records": len(records),
                      "rho_error_target_pass": result["rho_error_target_pass"]}), flush=True)
    if not result["rho_error_target_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
