"""Raw complex-trace diagnostics and explicit, full Fourier analysis.

The physical carrier is exp(-i 2*pi*f*t). numpy.fft.fft uses a negative-sign
kernel, so physical frequency is -numpy.fft.fftfreq. Both frequency halves and
the complex coefficients are retained. This is analysis code, not a simulator.
NumPy is imported only inside an explicitly called numeric operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import TIME_CONVENTION, atomic_write_bytes, verify_raw_manifest


class TraceValidationError(ValueError):
    """Raw complex/time information is missing or does not meet declared policy."""


def _numpy() -> Any:
    import numpy
    return numpy


@dataclass(frozen=True)
class ClockPolicy:
    """Operator-declared tolerance, in seconds and relative to mean sample dt.

    No interpolation or timestamp replacement occurs. The DFT assumes a uniform
    clock; accepting small deviations is an explicit approximation recorded in
    the diagnostics. Specify both tolerances; there is no hidden default.
    """

    relative_tolerance: float
    absolute_tolerance_s: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise TraceValidationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RawTrace:
    time_s: Any
    signal_complex: Any
    signal_unit: str
    source_keys: Mapping[str, Any]
    source_sampling: Mapping[str, Any]


def validate_timebase(time_s: Any, *, policy: ClockPolicy) -> dict[str, Any]:
    """Check finite, strictly increasing, explicitly tolerated uniform times."""
    np = _numpy()
    original = np.asarray(time_s)
    if original.ndim != 1 or len(original) < 2 or original.dtype.kind not in "iuf":
        raise TraceValidationError("Time must be a one-dimensional real array with >=2 samples")
    time = original.astype(np.float64, copy=True)
    if not np.isfinite(time).all():
        raise TraceValidationError("Time contains non-finite values")
    intervals = np.diff(time)
    if not (intervals > 0).all():
        raise TraceValidationError("Time must be strictly increasing, with no duplicate samples")
    dt = float((time[-1] - time[0]) / (len(time) - 1))
    deviation = float(np.max(np.abs(intervals - dt)))
    tolerance = policy.absolute_tolerance_s + policy.relative_tolerance * dt
    if deviation > tolerance:
        raise TraceValidationError(
            f"Nonuniform clock: max dt deviation {deviation:.17g} s exceeds declared "
            f"tolerance {tolerance:.17g} s; interpolation is not permitted"
        )
    record_duration = len(time) * dt
    return {
        "samples": len(time), "time_origin_s": float(time[0]),
        "raw_time_dtype": str(original.dtype),
        "time_validation_working_dtype": "float64",
        "last_sample_s": float(time[-1]), "observed_span_s": float(time[-1] - time[0]),
        "mean_dt_s": dt, "minimum_dt_s": float(intervals.min()),
        "maximum_dt_s": float(intervals.max()), "maximum_dt_deviation_s": deviation,
        "maximum_relative_dt_deviation": deviation / dt,
        "clock_policy": asdict(policy), "allowed_dt_deviation_s": tolerance,
        "clock_accepted": True, "timestamps_interpolated": False,
        "dft_uniform_clock_approximation": bool(deviation > 0),
        "record_duration_N_dt_s": record_duration,
        "nominal_independent_bin_width_hz": 1.0 / record_duration,
        "resolution_note": "1/(N*dt) is the nominal finite-record bin scale, not a universal two-line resolution theorem",
    }


def validate_complex_trace(time_s: Any, signal_complex: Any, *,
                           clock_policy: ClockPolicy) -> dict[str, Any]:
    np = _numpy()
    signal = np.asarray(signal_complex)
    if signal.ndim != 1 or signal.dtype.kind != "c":
        raise TraceValidationError(
            "A one-dimensional raw complex signal is required; power-only spectra "
            "or a real-valued proxy cannot reconstruct its missing phase/frequency half"
        )
    diagnostics = validate_timebase(time_s, policy=clock_policy)
    if len(signal) != diagnostics["samples"]:
        raise TraceValidationError("Time and raw complex signal lengths differ")
    if not np.isfinite(signal).all():
        raise TraceValidationError("Raw complex signal contains non-finite values")
    diagnostics["signal_dtype"] = str(signal.dtype)
    diagnostics["complex_phase_retained"] = True
    diagnostics["fft_executed"] = False
    return diagnostics


def load_raw_trace_npz(path: str | Path, *, raw_manifest: Mapping[str, Any],
                       raw_label: str, time_key: str,
                       complex_key: str | None = None,
                       real_key: str | None = None,
                       imaginary_key: str | None = None,
                       imaginary_sign: int | None = None) -> RawTrace:
    """Load explicitly named raw arrays, never infer them from power spectra.

    An authenticated raw manifest is required. Separate real/imaginary raw
    quadratures may be joined only when both keys are explicitly supplied.
    Object/pickle arrays are never deserialized. No FFT or clock repair occurs.
    """
    verify_raw_manifest(raw_manifest, raw_path=path, raw_label=raw_label)
    split = real_key is not None or imaginary_key is not None
    if bool(complex_key) == bool(split) or (split and not (real_key and imaginary_key)):
        raise TraceValidationError("Choose one complex key OR both real and imaginary raw keys")
    if split and (type(imaginary_sign) is not int or imaginary_sign not in {-1, 1}):
        raise TraceValidationError("Split quadratures require explicit imaginary_sign=+1 or -1")
    if complex_key and imaginary_sign is not None:
        raise TraceValidationError("imaginary_sign is only for split raw quadratures")
    requested_signal = {"raw_label": raw_label, "time_key": time_key,
                        "carrier_convention": TIME_CONVENTION,
                        "signal_unit": raw_manifest["units"]["signal"]}
    if complex_key:
        requested_signal["complex_key"] = complex_key
    else:
        requested_signal.update(real_key=real_key, imaginary_key=imaginary_key,
                                imaginary_sign=imaginary_sign)
    declared_labels = [label for label, declaration in raw_manifest["signal_declarations"].items()
                       if all(declaration.get(key) == value for key, value in requested_signal.items())]
    if len(declared_labels) != 1:
        raise TraceValidationError("The selected raw channel/assembly/sign must match exactly one original signal declaration")
    selected = [time_key, complex_key] if complex_key else [time_key, real_key, imaginary_key]
    forbidden = ("power", "spectrum", "spectra", "fft", "frequency", "f_axis")
    if any(not isinstance(key, str) or not key for key in selected):
        raise TraceValidationError("Explicit non-empty raw array keys are required")
    if any(any(word in key.lower() for word in forbidden) for key in selected[1:]):
        raise TraceValidationError("A spectral/power/FFT key is not a raw time trajectory")
    np = _numpy()
    try:
        payload = Path(path).read_bytes()
        identity = raw_manifest["raw_files"][raw_label]
        if hashlib.sha256(payload).hexdigest() != identity["sha256"] or len(payload) != identity["bytes"]:
            raise TraceValidationError("Raw input changed after manifest verification; analysis refused")
        with np.load(io.BytesIO(payload), allow_pickle=False) as record:
            missing = [key for key in selected if key not in record.files]
            if missing:
                raise TraceValidationError(f"Required raw array keys are absent: {missing}")
            time = record[time_key].copy()
            if complex_key:
                signal = record[complex_key].copy()
                keys = {"time": time_key, "complex_signal": complex_key}
            else:
                real, imaginary = record[real_key], record[imaginary_key]
                if (real.dtype.kind not in "iuf" or imaginary.dtype.kind not in "iuf"
                        or real.shape != imaginary.shape):
                    raise TraceValidationError("Raw real/imaginary quadratures must have equal shapes and real numeric dtypes")
                complex_dtype = np.result_type(real.dtype, imaginary.dtype, np.complex64)
                signal = np.empty(real.shape, dtype=complex_dtype)
                signal.real = real
                signal.imag = imaginary_sign * imaginary
                keys = {"time": time_key, "real_signal": real_key, "imaginary_signal": imaginary_key,
                        "imaginary_sign": imaginary_sign,
                        "assembly": "real + imaginary_sign*i*imaginary",
                        "raw_real_dtype": str(real.dtype), "raw_imaginary_dtype": str(imaginary.dtype)}
    except (OSError, ValueError) as exc:
        if isinstance(exc, TraceValidationError):
            raise
        raise TraceValidationError("Cannot read required non-pickled raw trajectory arrays") from exc
    if signal.dtype.kind != "c":
        raise TraceValidationError("The declared complex signal key contains no complex raw data")
    sampling = raw_manifest["sampling"]
    if str(time.dtype) != sampling["time_dtype"] or str(signal.dtype) != sampling["signal_dtype"]:
        raise TraceValidationError("Actual raw storage dtypes do not match the manifest sampling declaration")
    keys["signal_declaration_label"] = declared_labels[0]
    return RawTrace(time, signal, raw_manifest["units"]["signal"], keys, dict(sampling))


@dataclass(frozen=True)
class FullComplexSpectrum:
    raw_time_s: Any
    raw_signal_complex: Any
    window_weights: Any
    windowed_signal_complex: Any
    fft_complex_unshifted: Any
    physical_frequency_hz: Any
    fft_complex_sorted: Any
    amplitude_normalized_power: Any
    metadata: Mapping[str, Any]


def full_complex_spectrum(raw: RawTrace, *, clock_policy: ClockPolicy,
                          window: str, carrier_convention: str,
                          fft_points: int | None = None) -> FullComplexSpectrum:
    """Explicit future analysis operation. No simulation, fit or half-FFT truncation.

    Zero padding, if requested, interpolates the finite-window Fourier grid; it
    does not improve the independent resolution 1/(N*dt). Raw complex/time arrays
    and the complete original-order FFT are returned alongside sorted output.
    """
    if carrier_convention != TIME_CONVENTION:
        raise TraceValidationError("Only the declared exp(-i 2*pi*f*t) carrier convention is supported")
    if not isinstance(raw.signal_unit, str) or not raw.signal_unit:
        raise TraceValidationError("An explicit raw signal unit is required")
    diagnostics = validate_complex_trace(raw.time_s, raw.signal_complex, clock_policy=clock_policy)
    np = _numpy()
    time = np.array(raw.time_s, copy=True)
    signal = np.array(raw.signal_complex, copy=True)
    count = len(signal)
    points = count if fft_points is None else fft_points
    if type(points) is not int or points < count:
        raise TraceValidationError("fft_points must be an integer >= raw sample count; truncation is forbidden")
    if window == "rectangular":
        weights = np.ones(count, dtype=np.float64)
    elif window == "hann":
        if count < 3:
            raise TraceValidationError("A Hann window needs >=3 samples with nonzero coherent gain")
        weights = np.hanning(count)
    else:
        raise TraceValidationError("Explicit supported windows: rectangular, hann")
    coherent_gain = float(weights.mean())
    if not coherent_gain > 0:
        raise TraceValidationError("Window coherent gain must be positive")
    windowed = signal * weights
    complete_fft = np.fft.fft(windowed, n=points)
    if not np.isfinite(complete_fft).all():
        raise TraceValidationError("The complete complex FFT contains non-finite coefficients")
    physical = -np.fft.fftfreq(points, d=diagnostics["mean_dt_s"])
    ordering = np.argsort(physical)
    sorted_fft = complete_fft[ordering].copy()
    power = np.abs(sorted_fft / (count * coherent_gain)) ** 2
    if not np.isfinite(power).all():
        raise TraceValidationError("Normalized power contains non-finite values")
    independent = diagnostics["nominal_independent_bin_width_hz"]
    enbw_bins = float(count * np.sum(weights ** 2) / np.sum(weights) ** 2)
    metadata = {
        "time_convention": TIME_CONVENTION,
        "fft_kernel": "numpy.fft.fft: exp(-i 2*pi*k*n/Nfft)",
        "physical_frequency_axis": "-numpy.fft.fftfreq(Nfft, d=mean_dt_s), sorted ascending",
        "both_frequency_halves_retained": True,
        "negative_sign_carrier_maps_to_positive_physical_frequency": True,
        "fft_complex_normalization": "unnormalized NumPy FFT",
        "power_normalization": "abs(FFT/(Nraw*coherent_gain))**2; no one-sided doubling",
        "units": {"time": "s", "frequency": "Hz", "signal": raw.signal_unit,
                  "power": f"({raw.signal_unit})^2"},
        "raw_source_keys": dict(raw.source_keys),
        "raw_sampling_provenance": dict(raw.source_sampling),
        "raw_time_dtype": str(time.dtype), "raw_signal_dtype": str(signal.dtype),
        "fft_output_dtype": str(complete_fft.dtype),
        "raw_samples": count, "fft_points": points,
        "fft_grid_spacing_hz": 1.0 / (points * diagnostics["mean_dt_s"]),
        "nominal_independent_bin_width_hz": independent,
        "zero_padding_interpolation": bool(points > count),
        "zero_padding_improves_physical_resolution": False,
        "window": {"name": window, "coherent_gain": coherent_gain,
                   "equivalent_noise_bandwidth_bins": enbw_bins,
                   "equivalent_noise_bandwidth_hz": enbw_bins * independent},
        "registered_resolution_criterion_hz": max(independent, enbw_bins * independent),
        "registered_frequency_uncertainty_floor_hz": max(independent, enbw_bins * independent) / 2.0,
        "resolution_policy": "Conservative registered criterion=max(1/T, window ENBW); uncertainty floor=criterion/2. This is an analysis policy, not a universal resolution theorem or fitted uncertainty.",
        "phase_reference_time_s": diagnostics["time_origin_s"],
        "phase_reference_note": "FFT phase is referenced to the first sample; no origin phase correction was applied",
        "clock": diagnostics, "fft_executed": True, "fit_executed": False,
    }
    return FullComplexSpectrum(time, signal, weights, windowed, complete_fft.copy(),
                               physical[ordering].copy(), sorted_fft, power, metadata)


def save_full_spectrum_npz(path: str | Path, spectrum: FullComplexSpectrum) -> None:
    """Write a NEW complete record atomically; refuse overwriting any existing file."""
    np = _numpy()
    arrays = {
        "raw_time_s": spectrum.raw_time_s,
        "raw_signal_complex": spectrum.raw_signal_complex,
        "window_weights": spectrum.window_weights,
        "windowed_signal_complex": spectrum.windowed_signal_complex,
        "fft_complex_unshifted": spectrum.fft_complex_unshifted,
        "physical_frequency_hz": spectrum.physical_frequency_hz,
        "fft_complex_sorted": spectrum.fft_complex_sorted,
        "amplitude_normalized_power": spectrum.amplitude_normalized_power,
    }
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise TraceValidationError("Object arrays are not permitted in the complete spectrum record")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue(), overwrite=False)


@dataclass(frozen=True)
class PeakPolicy:
    band_min_hz: float
    band_max_hz: float
    minimum_separation_hz: float
    minimum_relative_height: float
    minimum_absolute_power: float = 0.0
    require_registered_resolution: bool = True

    def __post_init__(self) -> None:
        values = [self.band_min_hz, self.band_max_hz, self.minimum_separation_hz,
                  self.minimum_relative_height, self.minimum_absolute_power]
        if not all(math.isfinite(v) for v in values):
            raise TraceValidationError("All peak policy numbers must be finite")
        if (self.band_min_hz >= self.band_max_hz or self.minimum_separation_hz < 0
                or not 0 <= self.minimum_relative_height <= 1
                or self.minimum_absolute_power < 0):
            raise TraceValidationError("Invalid physical-frequency peak policy")


@dataclass(frozen=True)
class SpectralPeak:
    frequency_hz: float
    amplitude_normalized_power: float
    full_sorted_index: int


def select_band_peaks(spectrum: FullComplexSpectrum, *, policy: PeakPolicy) -> dict[str, Any]:
    """Select full-spectrum local maxima with a declared separation in Hz.

    This is not a linewidth fit or proof of resolved hybridization. Plateau
    maxima are represented by the middle bin. Band endpoints never manufacture
    a peak: neighbors are evaluated on the full sorted spectrum. No 'distance=5'
    or other fixed bin count is used. Padding interpolation is reported.
    """
    np = _numpy()
    frequency = np.asarray(spectrum.physical_frequency_hz)
    power = np.asarray(spectrum.amplitude_normalized_power)
    if (frequency.ndim != 1 or power.shape != frequency.shape or len(power) < 3
            or not np.isfinite(frequency).all() or not np.isfinite(power).all()
            or not (np.diff(frequency) > 0).all() or (power < 0).any()):
        raise TraceValidationError("Invalid sorted full complex spectrum")
    in_band = (frequency >= policy.band_min_hz) & (frequency <= policy.band_max_hz)
    if not in_band.any():
        raise TraceValidationError("The requested physical frequency band contains no bins")
    threshold = max(policy.minimum_absolute_power,
                    policy.minimum_relative_height * float(power[in_band].max()))
    candidates: list[SpectralPeak] = []
    i = 1
    while i < len(power) - 1:
        end = i
        while end + 1 < len(power) and power[end + 1] == power[i]:
            end += 1
        if end < len(power) - 1 and power[i] > power[i - 1] and power[end] > power[end + 1]:
            center = (i + end) // 2
            if in_band[center] and power[center] >= threshold:
                candidates.append(SpectralPeak(float(frequency[center]), float(power[center]), center))
        i = end + 1
    nominal = float(spectrum.metadata["nominal_independent_bin_width_hz"])
    resolution = float(spectrum.metadata["registered_resolution_criterion_hz"])
    if not (math.isfinite(nominal) and math.isfinite(resolution) and 0 < nominal <= resolution):
        raise TraceValidationError("Invalid nominal/window resolution metadata")
    minimum = max(policy.minimum_separation_hz, resolution) if policy.require_registered_resolution else policy.minimum_separation_hz
    selected: list[SpectralPeak] = []
    for peak in sorted(candidates, key=lambda p: (-p.amplitude_normalized_power, p.frequency_hz)):
        if all(abs(peak.frequency_hz - other.frequency_hz) >= minimum for other in selected):
            selected.append(peak)
    selected.sort(key=lambda p: p.frequency_hz)
    separations = [selected[i + 1].frequency_hz - selected[i].frequency_hz for i in range(len(selected) - 1)]
    return {
        "peaks": [asdict(peak) for peak in selected], "candidate_count": len(candidates),
        "policy": asdict(policy), "effective_minimum_separation_hz": minimum,
        "nominal_independent_bin_width_hz": nominal,
        "registered_resolution_criterion_hz": resolution,
        "registered_frequency_uncertainty_floor_hz": float(spectrum.metadata["registered_frequency_uncertainty_floor_hz"]),
        "adjacent_peak_separations_hz": separations,
        "all_selected_pairs_at_least_1_over_T": all(value >= nominal for value in separations),
        "all_selected_pairs_meet_registered_window_criterion": all(value >= resolution for value in separations),
        "zero_padding_interpolation": bool(spectrum.metadata["zero_padding_interpolation"]),
        "fft_grid_spacing_hz": float(spectrum.metadata["fft_grid_spacing_hz"]),
        "linewidth_fit_executed": False,
        "resolution_claim": "Compare nominal 1/T and registered window ENBW criterion separately; peak existence is not proof of separable physical modes",
    }


def _minimum_cost_assignment(cost: Sequence[Sequence[float]]) -> list[int]:
    """Rectangular Hungarian assignment, rows <= columns, with no reused column."""
    rows = len(cost)
    columns = len(cost[0]) if rows else 0
    if rows > columns or any(len(row) != columns for row in cost):
        raise TraceValidationError("Invalid assignment cost dimensions")
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    matched_row, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        column = 0
        best, used = [math.inf] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, columns + 1):
                if not used[candidate]:
                    reduced = cost[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                    if reduced < best[candidate]:
                        best[candidate], previous[candidate] = reduced, column
                    if best[candidate] < delta:
                        delta, next_column = best[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    best[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            predecessor = previous[column]
            matched_row[column] = matched_row[predecessor]
            column = predecessor
            if column == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


def branch_coverage_metrics(*, expected_branches_hz: Mapping[str, float],
                             observed_peak_frequencies_hz: Sequence[float],
                             assignment_tolerance_hz: float,
                             nominal_independent_bin_width_hz: float,
                             registered_resolution_criterion_hz: float,
                             registered_frequency_uncertainty_floor_hz: float) -> dict[str, Any]:
    """Match externally supplied expected branches one-to-one, with missingness.

    No theory/eigenvalues are generated here. A matched-only RMSE is explicitly
    conditional; incomplete branch coverage can never pass validation. Dummy
    assignments prioritize maximum in-tolerance coverage before minimum error.
    """
    if not expected_branches_hz or any(not isinstance(name, str) or not name for name in expected_branches_hz):
        raise TraceValidationError("Explicit named expected branches are required")
    expected = list(expected_branches_hz.items())
    observed = list(observed_peak_frequencies_hz)
    numbers = [value for _, value in expected] + observed
    if (not all(math.isfinite(value) for value in numbers)
            or not math.isfinite(assignment_tolerance_hz) or assignment_tolerance_hz <= 0
            or not math.isfinite(nominal_independent_bin_width_hz) or nominal_independent_bin_width_hz <= 0
            or not math.isfinite(registered_resolution_criterion_hz) or registered_resolution_criterion_hz < nominal_independent_bin_width_hz
            or not math.isfinite(registered_frequency_uncertainty_floor_hz) or registered_frequency_uncertainty_floor_hz < registered_resolution_criterion_hz / 2):
        raise TraceValidationError("Frequencies and positive tolerance/resolution scale must be finite")
    if len(set(observed)) != len(observed):
        raise TraceValidationError("Duplicate observed frequencies cannot act as independent peaks")
    missing_cost = len(expected) + 1.0
    invalid_cost = missing_cost * (len(expected) + 1.0)
    cost = []
    for _, value in expected:
        costs = [abs(value - peak) / assignment_tolerance_hz
                 if abs(value - peak) <= assignment_tolerance_hz else invalid_cost
                 for peak in observed]
        cost.append(costs + [missing_cost] * len(expected))
    assignment = _minimum_cost_assignment(cost)
    matches, missing, used, errors = [], [], set(), []
    for (name, value), index in zip(expected, assignment):
        if index < len(observed) and abs(value - observed[index]) <= assignment_tolerance_hz:
            error = observed[index] - value
            matches.append({"branch": name, "expected_frequency_hz": value,
                            "observed_peak_index": index, "observed_frequency_hz": observed[index],
                            "signed_error_hz": error})
            used.add(index)
            errors.append(error)
        else:
            missing.append(name)
    expected_separations = [abs(a[1] - b[1]) for i, a in enumerate(expected) for b in expected[i + 1:]]
    matched_observed = [entry["observed_frequency_hz"] for entry in matches]
    matched_observed_separations = [abs(a - b) for i, a in enumerate(matched_observed) for b in matched_observed[i + 1:]]
    all_observed_separations = [abs(a - b) for i, a in enumerate(observed) for b in observed[i + 1:]]
    nominally_separable = all(value >= nominal_independent_bin_width_hz for value in expected_separations)
    window_separable = all(value >= registered_resolution_criterion_hz for value in expected_separations)
    matched_observed_separable = all(value >= registered_resolution_criterion_hz for value in matched_observed_separations)
    all_observed_separable = all(value >= registered_resolution_criterion_hz for value in all_observed_separations)
    tolerance_respects_floor = assignment_tolerance_hz >= registered_frequency_uncertainty_floor_hz
    complete = not missing
    return {
        "expected_branch_count": len(expected), "observed_peak_count": len(observed),
        "matched_branch_count": len(matches), "coverage_fraction": len(matches) / len(expected),
        "coverage_complete": complete, "missing_branches": missing,
        "unassigned_observed_peak_indices": [i for i in range(len(observed)) if i not in used],
        "matches": matches,
        "matched_only_rmse_hz": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
        "rmse_scope": "matched branches only; never substitutes for complete coverage",
        "one_peak_reused_for_multiple_branches": False,
        "expected_pair_separations_hz": expected_separations,
        "nominal_independent_bin_width_hz": nominal_independent_bin_width_hz,
        "expected_pairs_at_least_1_over_T": nominally_separable,
        "registered_resolution_criterion_hz": registered_resolution_criterion_hz,
        "registered_frequency_uncertainty_floor_hz": registered_frequency_uncertainty_floor_hz,
        "expected_pairs_meet_registered_window_criterion": window_separable,
        "matched_observed_pair_separations_hz": matched_observed_separations,
        "matched_observed_pairs_meet_registered_window_criterion": matched_observed_separable,
        "all_observed_pair_separations_hz": all_observed_separations,
        "all_observed_pairs_meet_registered_window_criterion": all_observed_separable,
        "assignment_tolerance_respects_registered_uncertainty_floor": tolerance_respects_floor,
        "validation_pass": complete and window_separable and matched_observed_separable and all_observed_separable and tolerance_respects_floor,
        "status": "complete_and_meets_registered_criterion" if complete and window_separable and matched_observed_separable and all_observed_separable and tolerance_respects_floor else "incomplete_or_unresolved_or_below_uncertainty_floor",
        "expected_values_origin": "externally supplied; no scientific prediction computed by this function",
    }
