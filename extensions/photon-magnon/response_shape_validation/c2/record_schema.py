"""Pure standard-library admission of C2 saved metadata, clocks and state shape.

No magnetic model, truth file, scientific runtime, fit or prediction is imported.
Clocks are observed at the stored state; a separate planned clock is diagnostic.
Malformed input is rejected without interpolation, normalization or repair.
"""
from hashlib import sha256
import math
from pathlib import Path

CLOCK_SCHEMA = "c2-observed-clock-v1"
# Local uniformity uses the existing fitter window-selector timing slack.
CLOCK_TOL_S = 1e-24
# Prospective bound on accumulated solver time versus the declared schedule.
# This is a sampling-admission tolerance, not an integration/fit accuracy claim.
SCHEDULE_TOL_S = 1e-21
SCHEMA_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def _finite_real(value, label):
    if isinstance(value, (bool, str, bytes, complex)):
        raise ValueError(label + " must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + " must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _real_vector(values, label, count=None):
    kind = getattr(getattr(values, "dtype", None), "kind", None)
    if kind is not None and kind not in "fiu":
        raise ValueError(label + " must have a real numeric dtype")
    try:
        size = len(values)
    except TypeError as error:
        raise ValueError(label + " must be a one-dimensional sequence") from error
    if count is not None and size != count:
        raise ValueError(label + " length differs from the saved clock")
    return [_finite_real(value, label) for value in values]


def validate_clock_sample(actual, planned, previous=None, interval=None):
    """Check one observed time without modifying it; used during generation."""
    actual = _finite_real(actual, "actual solver time")
    planned = _finite_real(planned, "planned solver time")
    if abs(actual - planned) > SCHEDULE_TOL_S:
        raise ValueError("Observed solver clock differs from its declared schedule")
    if previous is not None:
        previous = _finite_real(previous, "previous solver time")
        interval = _finite_real(interval, "sample interval")
        if interval <= 0 or actual <= previous:
            raise ValueError("Observed solver clock must be strictly increasing")
        if abs((actual - previous) - interval) > CLOCK_TOL_S:
            raise ValueError("Observed solver clock is nonuniform")


def validate_record_schema(meta, raw, *, name=None):
    """Admit a complete saved C2 schema and clock; no physical calculations."""
    if not isinstance(meta, dict) or not isinstance(raw, dict):
        raise ValueError("C2 metadata and arrays must be mappings")
    if set(raw) != {"t_s", "planned_t_s", "m6", "Mx", "B_T"}:
        raise ValueError("C2 arrays must be exactly t_s,planned_t_s,m6,Mx,B_T; use a new observed-clock run")
    if (meta.get("clock_schema") != CLOCK_SCHEMA
            or meta.get("time_sampling") != "observed_solver_clock"
            or meta.get("time_interpolated") is not False):
        raise ValueError("C2 requires observed noninterpolated solver-clock metadata")
    if meta.get("time_units") != "s" or meta.get("field_units") != "T":
        raise ValueError("C2 time/field units must be explicitly s/T")
    if meta.get("magnetization") != "mA,mB unit vectors; Mx=mAx+mBx":
        raise ValueError("C2 must declare the physical sublattice state/readout convention")
    if meta.get("case_id") not in ("A", "B", "C"):
        raise ValueError("C2 case_id must be A, B or C")
    if meta.get("record_type") not in ("seedA", "seedB", "pulse"):
        raise ValueError("C2 record_type must be seedA, seedB or pulse")
    if not isinstance(meta.get("variant"), str) or not meta["variant"]:
        raise ValueError("C2 variant is missing")
    if name is not None and name != "{}_{}_{}".format(meta["case_id"], meta["variant"], meta["record_type"]):
        raise ValueError("C2 metadata identity differs from its record name")
    interval = _finite_real(meta.get("sample_interval_s"), "sample_interval_s")
    step = _finite_real(meta.get("integration_step_s"), "integration_step_s")
    start = _finite_real(meta.get("planned_start_s"), "planned_start_s")
    if interval <= 0 or step <= 0:
        raise ValueError("C2 sampling and integration steps must be positive")
    times = _real_vector(raw["t_s"], "t_s")
    if len(times) < 2:
        raise ValueError("C2 clock needs at least two samples")
    planned = _real_vector(raw["planned_t_s"], "planned_t_s", len(times))
    if type(meta.get("planned_sample_count")) is not int or meta["planned_sample_count"] != len(times):
        raise ValueError("C2 planned sample count differs from the retained arrays")
    dtype = getattr(getattr(raw["t_s"], "dtype", None), "str", None)
    if not isinstance(meta.get("time_dtype"), str) or not meta["time_dtype"]:
        raise ValueError("Stored time dtype must be explicitly recorded")
    if dtype is not None and meta["time_dtype"] != dtype:
        raise ValueError("Recorded time dtype differs from the actual saved array")
    if meta.get("clock_absolute_tolerance_s") != CLOCK_TOL_S or meta.get("schedule_absolute_tolerance_s") != SCHEDULE_TOL_S:
        raise ValueError("Clock tolerance declarations differ from the source-bound schema")
    for index, (actual, scheduled) in enumerate(zip(times, planned)):
        if abs(scheduled - (start + index * interval)) > CLOCK_TOL_S:
            raise ValueError("Planned clock differs from its declared schedule")
        validate_clock_sample(actual, scheduled, times[index - 1] if index else None, interval)
    states = raw["m6"]
    kind = getattr(getattr(states, "dtype", None), "kind", None)
    if kind is not None and kind not in "fiu":
        raise ValueError("m6 must have a real numeric dtype")
    try:
        if len(states) != len(times):
            raise ValueError("m6 length differs from the saved clock")
        for row in states:
            _real_vector(row, "m6 row", 6)
    except TypeError as error:
        raise ValueError("m6 must have shape (samples,6)") from error
    mx = _real_vector(raw["Mx"], "Mx", len(times))
    for index, row in enumerate(states):
        if mx[index] != float(row[0]) + float(row[3]):
            raise ValueError("Mx must equal the stored mAx+mBx exactly")
    _real_vector(raw["B_T"], "B_T", len(times))
    wave = meta.get("waveform")
    if meta["record_type"] == "pulse":
        if not isinstance(wave, dict) or wave.get("type") != "finite_sin4_cos":
            raise ValueError("Pulse waveform metadata must declare finite_sin4_cos")
        for key in ("start_s", "duration_s", "carrier_Hz", "amplitude_T"):
            value = _finite_real(wave.get(key), "waveform " + key)
            if key in ("duration_s", "carrier_Hz") and value <= 0:
                raise ValueError("Pulse duration and carrier must be positive")
    elif wave is not None:
        raise ValueError("Free-seed records must not declare a pulse waveform")
    return {"sample_count": len(times), "sample_interval_s": interval,
            "clock_absolute_tolerance_s": CLOCK_TOL_S,
            "schedule_absolute_tolerance_s": SCHEDULE_TOL_S,
            "finite_strict_uniform_clock_and_state_schema": True}
