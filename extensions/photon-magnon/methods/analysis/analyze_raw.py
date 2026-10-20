"""Explicit stored-raw analysis CLI. Help performs no NumPy import or analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE / "src"
SOURCE_PATHS = {
    "analysis/analyze_raw.py": Path(__file__).resolve(),
    "src/cavity_method/provenance.py": SOURCE_ROOT / "cavity_method/provenance.py",
    "src/cavity_method/spectrum.py": SOURCE_ROOT / "cavity_method/spectrum.py",
    "src/cavity_method/__init__.py": SOURCE_ROOT / "cavity_method/__init__.py",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Inspect retained raw complex traces, or perform separately authorized full Fourier analysis. No simulation or fitting commands exist.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("source-identities", help="Print current analysis source hashes only; no data/numeric operation")
    inspect = commands.add_parser("inspect-raw", help="Validate declared raw arrays/timebase only; no FFT or peak analysis")
    transform = commands.add_parser("full-spectrum", help="Explicit approved analysis of one named raw record; never runs cases automatically")
    for command in (inspect, transform):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--raw-manifest", type=Path, required=True)
        command.add_argument("--raw-label", required=True, help="Exact key in the raw manifest raw_files group")
        command.add_argument("--time-key", required=True)
        group = command.add_mutually_exclusive_group(required=True)
        group.add_argument("--complex-key")
        group.add_argument("--real-key", help="Requires --imaginary-key as well")
        command.add_argument("--imaginary-key")
        command.add_argument("--imaginary-sign", type=int, choices=(-1, 1),
                             help="Required for split raw quadratures: +1 gives q+i*p; -1 gives mx-i*my")
        command.add_argument("--clock-rtol", type=float, required=True)
        command.add_argument("--clock-atol-s", type=float, required=True)
    inspect.add_argument("--output-json", type=Path, help="Optional NEW diagnostic JSON inside analysis_outputs; otherwise stdout")
    transform.add_argument("--authorization", type=Path)
    transform.add_argument("--print-review-configuration", action="store_true",
                           help="Print exact canonical settings/source identities without importing NumPy or executing FFT; no approval is generated")
    transform.add_argument("--execute-approved-analysis", action="store_true",
                           help="Explicit execution acknowledgement; cannot authorize new simulations")
    transform.add_argument("--output", type=Path, required=True,
                           help="NEW .npz under this workspace's analysis_outputs directory")
    transform.add_argument("--window", choices=("rectangular", "hann"), required=True)
    transform.add_argument("--fft-points", type=int)
    transform.add_argument("--peak-band-hz", type=float, nargs=2, metavar=("MIN", "MAX"))
    transform.add_argument("--minimum-separation-hz", type=float)
    transform.add_argument("--minimum-relative-height", type=float)
    transform.add_argument("--minimum-absolute-power", type=float, default=0.0)
    transform.add_argument("--permit-peaks-below-registered-window-criterion", action="store_true")
    transform.add_argument("--expected-branches-json", type=Path,
                           help="Externally supplied named frequencies in Hz; this tool computes no theory")
    transform.add_argument("--assignment-tolerance-hz", type=float)
    return result


def _output_path(path: Path) -> Path:
    target = path.resolve()
    allowed = (WORKSPACE / "analysis_outputs").resolve()
    if not target.is_relative_to(allowed) or target == allowed:
        raise ValueError("Outputs must be new files under methods/analysis_outputs")
    if not target.parent.is_dir():
        raise ValueError("Output parent must already exist; create/review it explicitly")
    if target.exists():
        raise ValueError("Existing output files are never overwritten")
    return target


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    # parse_args exits on --help before importing any package/numeric dependency.
    SOURCE_PATHS.update({path.relative_to(WORKSPACE).as_posix(): path
                         for path in (SOURCE_ROOT / "cavity_method").rglob("*.py")})
    sys.path.insert(0, str(SOURCE_ROOT))
    from cavity_method import provenance as prov

    if arguments.command == "source-identities":
        print(json.dumps({label: item["sha256"] for label, item in prov.identities(SOURCE_PATHS).items()}, indent=2))
        return 0
    try:
        manifest, manifest_identity = prov.read_json_with_identity(arguments.raw_manifest)
        prov.verify_raw_manifest(manifest, raw_path=arguments.input, raw_label=arguments.raw_label)
        if arguments.real_key and not arguments.imaginary_key:
            raise ValueError("--real-key requires --imaginary-key; no missing quadrature is inferred")
        if arguments.complex_key and arguments.imaginary_key:
            raise ValueError("--imaginary-key cannot accompany --complex-key")
        if arguments.real_key and arguments.imaginary_sign is None:
            raise ValueError("Split quadratures require --imaginary-sign +1 or -1")
        if arguments.complex_key and arguments.imaginary_sign is not None:
            raise ValueError("--imaginary-sign is only for split quadratures")
        operations = {"full_complex_fft"}
        extra_inputs: dict[str, Path] = {}
        expected = None
        configuration = None
        authorization_identity = None
        output = None
        if arguments.command == "full-spectrum":
            if not arguments.execute_approved_analysis and not arguments.print_review_configuration:
                raise ValueError("No FFT executed: --execute-approved-analysis plus a matching review record is required")
            if arguments.peak_band_hz is not None:
                if arguments.minimum_separation_hz is None or arguments.minimum_relative_height is None:
                    raise ValueError("Peak selection requires explicit separation in Hz and relative height")
                operations.add("peak_selection")
            elif (arguments.minimum_separation_hz is not None
                  or arguments.minimum_relative_height is not None
                  or arguments.permit_peaks_below_registered_window_criterion):
                raise ValueError("Peak settings require --peak-band-hz")
            if arguments.expected_branches_json:
                if "peak_selection" not in operations or arguments.assignment_tolerance_hz is None:
                    raise ValueError("Branch coverage requires explicit peak selection and assignment tolerance")
                operations.add("branch_coverage")
                extra_inputs["expected_branches.json"] = arguments.expected_branches_json
            elif arguments.assignment_tolerance_hz is not None:
                raise ValueError("Assignment tolerance requires externally supplied expected branches")
            configuration = {
                "raw_label": arguments.raw_label, "time_key": arguments.time_key,
                "complex_key": arguments.complex_key, "real_key": arguments.real_key,
                "imaginary_key": arguments.imaginary_key, "imaginary_sign": arguments.imaginary_sign,
                "carrier_convention": prov.TIME_CONVENTION,
                "window": arguments.window, "fft_points": arguments.fft_points,
                "clock_relative_tolerance": arguments.clock_rtol,
                "clock_absolute_tolerance_s": arguments.clock_atol_s,
                "peak_band_hz": arguments.peak_band_hz,
                "minimum_separation_hz": arguments.minimum_separation_hz,
                "minimum_relative_height": arguments.minimum_relative_height,
                "minimum_absolute_power": arguments.minimum_absolute_power,
                "permit_peaks_below_registered_window_criterion": arguments.permit_peaks_below_registered_window_criterion,
                "expected_branches_input": "expected_branches.json" if arguments.expected_branches_json else None,
                "assignment_tolerance_hz": arguments.assignment_tolerance_hz,
                "operations": sorted(operations),
            }
            if arguments.print_review_configuration:
                print(json.dumps({"analysis_configuration": configuration,
                    "analysis_configuration_sha256": prov.configuration_sha256(configuration),
                    "analysis_sources_sha256": {label: item["sha256"] for label, item in prov.identities(SOURCE_PATHS).items()},
                    "raw_input_sha256": prov.sha256_file(arguments.input),
                    "raw_manifest_sha256": manifest_identity["sha256"],
                    "additional_inputs_sha256": {label: item["sha256"] for label, item in prov.identities(extra_inputs).items()} if extra_inputs else {},
                    "approval_created": False, "fft_executed": False, "simulation_executed": False}, indent=2))
                return 0
            if arguments.authorization is None:
                raise ValueError("Actual Fourier execution requires an explicit --authorization review record")
            authorization, authorization_identity = prov.read_json_with_identity(arguments.authorization)
            if authorization.get("raw_manifest_sha256") != manifest_identity["sha256"]:
                raise ValueError("The raw manifest bytes read do not match the approval record")
            prov.verify_analysis_authorization(
                authorization, raw_path=arguments.input,
                raw_manifest_path=arguments.raw_manifest, sources=SOURCE_PATHS,
                operations=operations, configuration=configuration, additional_inputs=extra_inputs,
            )
            if arguments.expected_branches_json:
                expected = prov.read_verified_json(arguments.expected_branches_json,
                    expected_sha256=authorization["additional_inputs_sha256"]["expected_branches.json"])
                if expected.get("frequency_unit") != "Hz" or not isinstance(expected.get("branches"), dict):
                    raise ValueError("Expected branch JSON needs frequency_unit='Hz' and a branches object")
            output = _output_path(arguments.output)
            if output.suffix.lower() != ".npz":
                raise ValueError("Full spectrum output must have .npz suffix")
            for related in (Path(str(output) + ".analysis.json"), Path(str(output) + ".manifest.json")):
                _output_path(related)
        diagnostic_output = (_output_path(arguments.output_json)
                             if arguments.command == "inspect-raw" and arguments.output_json else None)

        from cavity_method import spectrum as spec
        raw = spec.load_raw_trace_npz(
            arguments.input, raw_manifest=manifest, raw_label=arguments.raw_label,
            time_key=arguments.time_key, complex_key=arguments.complex_key,
            real_key=arguments.real_key, imaginary_key=arguments.imaginary_key,
            imaginary_sign=arguments.imaginary_sign,
        )
        clock_policy = spec.ClockPolicy(arguments.clock_rtol, arguments.clock_atol_s)
        diagnostics = spec.validate_complex_trace(raw.time_s, raw.signal_complex, clock_policy=clock_policy)
        if arguments.command == "inspect-raw":
            receipt = {"operation": "raw_preprocessing_diagnostics_only", "fft_executed": False,
                       "fit_executed": False, "simulation_executed": False,
                       "raw_identity": manifest["raw_files"][arguments.raw_label],
                       "raw_manifest_sha256": manifest_identity["sha256"],
                       "source_identities": prov.identities(SOURCE_PATHS),
                       "diagnostics": diagnostics,
                       "published_data_repaired": False}
            if diagnostic_output:
                prov.atomic_write_json(diagnostic_output, receipt)
            else:
                print(json.dumps(receipt, indent=2, allow_nan=False))
            return 0

        transformed = spec.full_complex_spectrum(
            raw, clock_policy=clock_policy, window=arguments.window,
            carrier_convention=prov.TIME_CONVENTION, fft_points=arguments.fft_points,
        )
        analysis: dict[str, Any] = {"spectrum_metadata": dict(transformed.metadata),
                                    "simulation_executed": False, "fit_executed": False,
                                    "published_data_repaired": False}
        if "peak_selection" in operations:
            peak_policy = spec.PeakPolicy(
                *arguments.peak_band_hz, arguments.minimum_separation_hz,
                arguments.minimum_relative_height, arguments.minimum_absolute_power,
                not arguments.permit_peaks_below_registered_window_criterion,
            )
            selected = spec.select_band_peaks(transformed, policy=peak_policy)
            analysis["peak_selection"] = selected
            if "branch_coverage" in operations:
                analysis["branch_coverage"] = spec.branch_coverage_metrics(
                    expected_branches_hz=expected["branches"],
                    observed_peak_frequencies_hz=[peak["frequency_hz"] for peak in selected["peaks"]],
                    assignment_tolerance_hz=arguments.assignment_tolerance_hz,
                    nominal_independent_bin_width_hz=transformed.metadata["nominal_independent_bin_width_hz"],
                    registered_resolution_criterion_hz=transformed.metadata["registered_resolution_criterion_hz"],
                    registered_frequency_uncertainty_floor_hz=transformed.metadata["registered_frequency_uncertainty_floor_hz"],
                )
        # Each artifact is atomic. The final manifest is published LAST. A partial
        # multi-file failure leaves no complete receipt and must not be resumed
        # merely because an orphan NPZ exists.
        spec.save_full_spectrum_npz(output, transformed)
        analysis_path = Path(str(output) + ".analysis.json")
        prov.atomic_write_json(analysis_path, analysis)
        output_manifest = prov.build_manifest(
            record_kind="stored_raw_analysis", sources=SOURCE_PATHS, binaries={},
            units=dict(transformed.metadata["units"]), precision="mixed",
            sampling=manifest["sampling"],
            signal_declarations={raw.source_keys["signal_declaration_label"]:
                                 manifest["signal_declarations"][raw.source_keys["signal_declaration_label"]]},
            parameters={"window": arguments.window, "fft_points": arguments.fft_points,
                        "clock_policy": diagnostics["clock_policy"],
                        "operations": sorted(operations), "source_simulation_parameters": manifest["parameters"]},
            raw_files={arguments.raw_label: arguments.input}, native_binary_use="not_executed",
        )
        if output_manifest["raw_files"][arguments.raw_label] != manifest["raw_files"][arguments.raw_label]:
            raise ValueError("Raw input changed during analysis; no complete receipt will be published")
        final_source_hashes = {label: item["sha256"] for label, item in output_manifest["sources"].items()}
        if final_source_hashes != authorization["analysis_sources_sha256"]:
            raise ValueError("Analysis source changed during execution; no complete receipt will be published")
        output_manifest.update({
            "status": "complete", "fft_executed": True, "fit_executed": False,
            "simulation_executed": False, "published_data_repaired": False,
            "raw_manifest_sha256": manifest_identity["sha256"],
            "authorization_sha256": authorization_identity["sha256"],
            "analysis_configuration": configuration,
            "analysis_configuration_sha256": prov.configuration_sha256(configuration),
            "source_precision": manifest["precision"],
            "analysis_fft_precision": transformed.metadata["fft_output_dtype"],
            "additional_inputs_sha256": authorization.get("additional_inputs_sha256", {}),
            "outputs": {"full_spectrum.npz": prov.file_identity(output),
                        "analysis.json": prov.file_identity(analysis_path)},
        })
        prov.atomic_write_json(Path(str(output) + ".manifest.json"), output_manifest)
        print(json.dumps({"status": "complete", "output": str(output),
                          "fft_executed": True, "simulation_executed": False,
                          "published_data_repaired": False}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Analysis refused or incomplete: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
