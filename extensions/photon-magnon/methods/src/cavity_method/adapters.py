"""Uniform co-simulation adapter; never imports a magnetic engine.

The original exact-hash implementation had one finite lossless CPU FM pilot.
This edited source-only port has not been physically rerun. A caller must bind
its model, protocol, engine/binary and complete checkpoint support. In particular,
q,p alone cannot restore a coupled run. No native/profile equivalence is claimed;
AFM hook declarations remain an unvalidated caller contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .energy import (EnergyCavityModel, OverlapSnapshot, Quadratures,
                     QuadratureRate, UnsupportedCoupling, finite_real,
                     reject_legacy_coupling)


def _json_copy(value: Any) -> Any:
    """Reject nonfinite/nonportable payloads; never pickle native objects."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _vector(value: Any, label: str) -> tuple[float, float, float]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{label} must have exactly three components")
    return tuple(finite_real(x, label) for x in values)


def _assert_no_native(objects: tuple[Any, ...]) -> None:
    """Inspect all declared sources, hosts, sublattices and their impl objects."""
    pending, seen = list(objects), set()
    flags = ("enable_cavity_mode", "enable_multimode_cavity", "enable_cavity_afm",
             "enable_aux_mode")
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        impl = getattr(obj, "_impl", None)
        if impl is not None:
            pending.append(impl)
        # Only the public property is traversed. Unknown native methods are not
        # invoked speculatively; the host itself must be provided by the hooks.
        sublattices = getattr(obj, "sublattices", ())
        if not callable(sublattices):
            pending.extend(tuple(sublattices))
        for name in flags:
            value = getattr(obj, name, False)
            if callable(value):
                raise RuntimeError(f"Cannot inspect native flag {name}; provide explicit guard objects")
            if bool(value):
                raise RuntimeError(f"Native cavity flag {name} is enabled; double counting is forbidden")


@dataclass(frozen=True)
class UniformSourceHooks:
    """Callbacks for ONE complete FM/sublattice magnetic state.

    The field setter owns the source-local bias handle. base_bias_t must include
    every pre-existing local bias contribution; world bias stays engine-owned.
    Time-dependent local biases need a separate validated adapter and are not
    supported here. required_partner_sources must list other AFM sublattices.
    guard_objects must include the source AND its AFM host when applicable.
    No user/native module is imported to construct these callbacks.
    """

    name: str
    read_average: Callable[[], Any]
    read_full_magnetization: Callable[[], Any]
    restore_full_magnetization: Callable[[Any], None]
    set_local_bias_t: Callable[[tuple[float, float, float]], None]
    base_bias_t: tuple[float, float, float]
    guard_objects: tuple[Any, ...]
    source_kind: str
    magnetization_shape: tuple[int, ...]
    host_guard_object: Any | None = None
    required_partner_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.guard_objects:
            raise ValueError("Every source needs a name and explicit native guard objects")
        object.__setattr__(self, "base_bias_t", _vector(self.base_bias_t, "base_bias_t"))
        object.__setattr__(self, "guard_objects", tuple(self.guard_objects))
        object.__setattr__(self, "required_partner_sources", tuple(self.required_partner_sources))
        shape = tuple(self.magnetization_shape)
        if (len(shape) < 2 or shape[0] != 3 or
                any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)):
            raise ValueError("Full magnetization shape must be (3, ...cells), not an average vector")
        object.__setattr__(self, "magnetization_shape", shape)
        if self.source_kind not in ("fm", "afm_sublattice"):
            raise ValueError("source_kind must be fm or afm_sublattice")
        if self.source_kind == "afm_sublattice":
            if self.host_guard_object is None or not self.required_partner_sources:
                raise ValueError("AFM bindings require their host and every partner source name")
            if any(not isinstance(name, str) or not name.strip()
                   for name in self.required_partner_sources):
                raise ValueError("AFM partner names must be nonempty strings")
            if (self.name in self.required_partner_sources or
                    len(set(self.required_partner_sources)) != len(self.required_partner_sources)):
                raise ValueError("AFM partners must be distinct sources, without self or duplicate names")
            object.__setattr__(self, "guard_objects", self.guard_objects+(self.host_guard_object,))
        # Public mumax+ exposes this property on each sublattice. Reject a
        # mistakenly declared FM binding rather than omitting the host guard.
        for obj in self.guard_objects:
            impl = getattr(obj, "_impl", obj)
            flag = getattr(impl, "is_sublattice", False)
            if not callable(flag) and bool(flag) and self.source_kind != "afm_sublattice":
                raise ValueError("A native sublattice must be explicitly declared afm_sublattice")
        for callback in (self.read_average, self.read_full_magnetization,
                         self.restore_full_magnetization, self.set_local_bias_t):
            if not callable(callback):
                raise TypeError("Source hooks must be callable")


@dataclass(frozen=True)
class EngineHooks:
    """Engine callbacks supplied by a future preregistered runner.

    inspect_identity must return engine_name, engine_version, source_revision,
    binary_sha256, precision, version_verified=True. capture_full_state must
    return complete=True, time_s, nonempty parameters, solver_state and
    stochastic_state (explicitly 'disabled' is permitted for a deterministic
    engine). These include all geometry/material/bias/solver settings and any
    RNG/internal integration state required for an actual restart.

    If the engine cannot capture AND restore those fields, no resume-capable
    adapter can be constructed. A native q,p-only restart is never substituted.
    """

    time_s: Callable[[], float]
    advance: Callable[[float], None]
    inspect_identity: Callable[[], Mapping[str, Any]]
    capture_full_state: Callable[[], Mapping[str, Any]]
    restore_full_state: Callable[[Mapping[str, Any]], None]
    complete_state_supported: bool

    def __post_init__(self) -> None:
        if self.complete_state_supported is not True:
            raise UnsupportedCoupling("Full engine-state capture/restore is required; q,p-only resume is unsafe")
        for callback in (self.time_s, self.advance, self.inspect_identity,
                         self.capture_full_state, self.restore_full_state):
            if not callable(callback):
                raise TypeError("Engine hooks must be callable")


@dataclass(frozen=True)
class SynchronousSample:
    time_s: float
    cavities: Mapping[str, Quadratures]
    magnetization_average: Mapping[str, tuple[float, float, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cavities", MappingProxyType(dict(self.cavities)))
        object.__setattr__(self, "magnetization_average",
                           MappingProxyType(dict(self.magnetization_average)))


class UniformCosimulationAdapter:
    """Symmetric split with synchronous pre/end-step outputs.

    Cavity half RK4 at fixed initial spins -> magnetic full dt at fixed midpoint
    cavity field -> cavity half RK4 at fixed final spins. Original finite CPU FM
    evidence is confined to its exact source hashes and one setting; no general
    numerical order, accuracy or validation of this edited port is asserted.
    Only initial-condition excitation is admitted by this adapter. A general
    time-dependent drive needs its waveform identity and complete restart state
    bound to the protocol before it can be supported. No implicit run loop,
    time cap, fit, FFT, output file or execution occurs.
    """

    def __init__(self, model: EnergyCavityModel, engine: EngineHooks,
                 sources: tuple[UniformSourceHooks, ...], *,
                 initial_state: Mapping[str, Quadratures], macrostep_s: float,
                 preregistration_id: str, source_freeze_sha256: str,
                 clock_absolute_tolerance_s: float = 1e-24,
                 clock_relative_tolerance: float = 1e-12,
                 legacy_g_diss: float = 0.0, legacy_phase_rad: float = 0.0,
                 backend: str = "python_uniform") -> None:
        reject_legacy_coupling(g_diss=legacy_g_diss, phase_rad=legacy_phase_rad)
        if backend != "python_uniform":
            raise UnsupportedCoupling("This adapter cannot certify native/CUDA/profile equivalence")
        if any(m.profile_id != "uniform" for m in model.modes):
            raise UnsupportedCoupling("Spatial profiles require a separately validated spatial adapter")
        if not preregistration_id or not isinstance(preregistration_id, str):
            raise ValueError("A recorded future-execution preregistration_id is required")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(source_freeze_sha256)):
            raise ValueError("The reviewed implementation/source-freeze SHA-256 is required")
        model.validate_state(initial_state)
        self.model, self.engine = model, engine
        self.sources = tuple(sources)
        if len({s.name for s in self.sources}) != len(self.sources):
            raise ValueError("Duplicate source bindings")
        names = {s.name for s in self.sources}
        if names != {s.name for s in model.sources}:
            raise ValueError("Every declared magnetic source must have exactly one full-state binding")
        by_name = {s.name: s for s in self.sources}
        for source in self.sources:
            if not set(source.required_partner_sources).issubset(names):
                raise ValueError("All coupled AFM sublattices must be bound separately")
            if source.source_kind == "afm_sublattice":
                for name in source.required_partner_sources:
                    partner = by_name[name]
                    if (partner.source_kind != "afm_sublattice" or
                            partner.host_guard_object is not source.host_guard_object):
                        raise ValueError("AFM partners must be bound AFM sources on the same host")
                peers = {s.name for s in self.sources
                         if s is not source and s.source_kind == "afm_sublattice"
                         and s.host_guard_object is source.host_guard_object}
                if not peers or set(source.required_partner_sources) != peers:
                    raise ValueError("AFM partner declarations must include every distinct bound same-host peer")
                # This checks hook metadata, not physical callback ownership or
                # host completeness. Those remain the caller's explicit contract.
        self._state = dict(initial_state)
        self.dt = finite_real(macrostep_s, "macrostep_s")
        self.abs_tol = finite_real(clock_absolute_tolerance_s, "clock_absolute_tolerance_s")
        self.rel_tol = finite_real(clock_relative_tolerance, "clock_relative_tolerance")
        if self.dt <= 0 or self.abs_tol < 0 or self.rel_tol < 0:
            raise ValueError("Macrostep must be positive and clock tolerances nonnegative")
        self.protocol_id = preregistration_id
        self.source_freeze_sha256 = source_freeze_sha256.lower()
        self.accepted_steps = 0
        self._prepared = False
        self._broken = False
        self._identity = None
        self._time = None

    @property
    def state(self) -> Mapping[str, Quadratures]:
        return MappingProxyType(dict(self._state))

    def _clock_close(self, actual: float, expected: float) -> bool:
        tol = self.abs_tol + self.rel_tol*max(abs(actual), abs(expected))
        if tol > self.dt*1e-3:
            raise RuntimeError("Clock tolerance exceeds 0.1% of macrostep; refine the explicit tolerance")
        return abs(actual-expected) <= tol

    def _read_identity(self) -> dict:
        identity = _json_copy(self.engine.inspect_identity())
        if identity.get("version_verified") is not True:
            raise RuntimeError("Loaded engine and binary version must be independently verified")
        for key in ("engine_name", "engine_version", "source_revision", "precision"):
            if not isinstance(identity.get(key), str) or not identity[key].strip():
                raise RuntimeError(f"Missing engine identity: {key}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(identity.get("binary_sha256", ""))):
            raise RuntimeError("A verified loaded-binary SHA-256 is required")
        return identity

    def _guard(self) -> None:
        _assert_no_native(tuple(obj for source in self.sources for obj in source.guard_objects))
        if self._identity is not None and self._read_identity() != self._identity:
            raise RuntimeError("Engine identity changed during the coupled run")

    def _engine_snapshot(self) -> dict:
        record = _json_copy(self.engine.capture_full_state())
        if record.get("complete") is not True:
            raise RuntimeError("Engine checkpoint is not a complete restart state")
        if not isinstance(record.get("parameters"), dict) or not record["parameters"]:
            raise RuntimeError("Full engine parameters must be recorded for resume")
        for key in ("time_s", "solver_state", "stochastic_state"):
            if key not in record or record[key] is None:
                raise RuntimeError(f"Missing engine restart field: {key}")
        time = finite_real(record["time_s"], "checkpoint engine time")
        if not self._clock_close(time, finite_real(self.engine.time_s(), "engine time")):
            raise RuntimeError("Engine snapshot and clock are not synchronous")
        return record

    def _write_feedback(self, state: Mapping[str, Quadratures]) -> None:
        bx = self.model.uniform_feedback_x_t(state)
        for source in self.sources:
            base = source.base_bias_t
            source.set_local_bias_t((base[0]+bx, base[1], base[2]))

    def _full_magnetization(self, source: UniformSourceHooks, value: Any) -> Any:
        value = _json_copy(value)

        def validate(node, shape):
            if not shape:
                finite_real(node, "full magnetization component")
                return
            if not isinstance(node, list) or len(node) != shape[0]:
                raise ValueError(f"Full magnetization shape mismatch for {source.name}")
            for item in node:
                validate(item, shape[1:])

        validate(value, source.magnetization_shape)
        return value

    def _sample(self) -> SynchronousSample:
        return SynchronousSample(
            finite_real(self.engine.time_s(), "engine time"), dict(self._state),
            {s.name: _vector(s.read_average(), "magnetization average") for s in self.sources},
        )

    def _overlaps(self, sample: SynchronousSample) -> OverlapSnapshot:
        return OverlapSnapshot(
            {m.name: {s.name: sample.magnetization_average[s.name][0]
                      for s in self.sources} for m in self.model.modes},
            {m.name: "uniform" for m in self.model.modes},
        )

    def prepare(self) -> SynchronousSample:
        """Explicit future-execution entry; does not advance physical time."""
        if self._prepared or self._broken:
            raise RuntimeError("Adapter is already prepared or requires full checkpoint restoration")
        self._identity = self._read_identity()
        self._guard()
        self._engine_snapshot()  # Refuse missing restart support before field writes.
        for source in self.sources:
            self._full_magnetization(source, source.read_full_magnetization())
        self._broken = True  # Field writes may partly succeed before a callback fails.
        try:
            self._write_feedback(self._state)
            sample = self._sample()
            self._time, self._prepared, self._broken = sample.time_s, True, False
            return sample
        except Exception:
            self._prepared = False
            raise

    def _half_step(self, state: Mapping[str, Quadratures], overlaps: OverlapSnapshot,
                   t: float, duration: float,
                   drive: Callable[[float], Mapping[str, QuadratureRate]] | None
                   ) -> dict[str, Quadratures]:
        def rhs(x, time):
            return self.model.rhs(x, overlaps, drive=None if drive is None else drive(time))

        def shift(x, rate, fraction):
            return {name: Quadratures(x[name].q+fraction*duration*r.dq_dt,
                                      x[name].p+fraction*duration*r.dp_dt)
                    for name, r in rate.items()}

        k1 = rhs(state, t)
        k2 = rhs(shift(state, k1, 0.5), t+duration/2)
        k3 = rhs(shift(state, k2, 0.5), t+duration/2)
        k4 = rhs(shift(state, k3, 1.0), t+duration)
        return {name: Quadratures(
            x.q+duration*(k1[name].dq_dt+2*k2[name].dq_dt+2*k3[name].dq_dt+k4[name].dq_dt)/6,
            x.p+duration*(k1[name].dp_dt+2*k2[name].dp_dt+2*k3[name].dp_dt+k4[name].dp_dt)/6,
        ) for name, x in state.items()}

    def advance_one_macrostep(self, *,
                             drive: Callable[[float], Mapping[str, QuadratureRate]] | None = None
                             ) -> tuple[SynchronousSample, SynchronousSample]:
        """For a future authorized runner; return aligned samples before/after.

        A failed clock/engine operation permanently invalidates stepping until a
        full checkpoint is restored. Partial engine advancement is not retried.
        """
        if drive is not None:
            raise UnsupportedCoupling(
                "Time-dependent drive callbacks need a waveform/source identity "
                "and complete restart state. This adapter currently supports "
                "initial-condition excitation only."
            )
        if not self._prepared or self._broken:
            raise RuntimeError("Prepare the adapter, or restore a complete checkpoint, first")
        self._guard()
        before = self._sample()
        if not self._clock_close(before.time_s, self._time):
            self._broken = True
            raise RuntimeError("Engine was advanced outside the coupled adapter")
        expected_end = before.time_s+self.dt
        self._clock_close(expected_end, expected_end)  # Check tolerance before any advance.
        try:
            middle = self._half_step(self._state, self._overlaps(before),
                                     before.time_s, self.dt/2, drive)
            self._write_feedback(middle)
            self.engine.advance(self.dt)
            actual_end = finite_real(self.engine.time_s(), "engine time")
            if not self._clock_close(actual_end, expected_end):
                raise RuntimeError("Magnetic engine did not advance by the requested macrostep")
            # Only overlaps are consumed from this private temporary sample;
            # its cavity coordinates are not exposed as a synchronous output.
            # Use END spins for the second cavity half step.
            final_spins = self._sample()
            final = self._half_step(middle, self._overlaps(final_spins),
                                    before.time_s+self.dt/2, self.dt/2, drive)
            self._write_feedback(final)
            self._state = final
            self._time = actual_end
            self.accepted_steps += 1
            self._guard()
            return before, self._sample()
        except Exception:
            self._broken = True
            raise

    def _configuration(self) -> dict:
        return {"model": self.model.manifest(), "macrostep_s": self.dt,
                "clock_absolute_tolerance_s": self.abs_tol,
                "clock_relative_tolerance": self.rel_tol,
                "preregistration_id": self.protocol_id,
                "source_freeze_sha256": self.source_freeze_sha256,
                "drive_contract": "none_initial_condition_excitation_only",
                "base_bias_t": {s.name: list(s.base_bias_t) for s in self.sources},
                "source_layout": {s.name: {"kind": s.source_kind,
                                           "magnetization_shape": list(s.magnetization_shape),
                                           "partner_sources": list(s.required_partner_sources)}
                                  for s in self.sources},
                "splitting": "cavity_RK4_half__spin_full__cavity_RK4_half"}

    def checkpoint(self) -> dict:
        """Return a portable complete restart payload, without writing a file."""
        if not self._prepared or self._broken:
            raise RuntimeError("Cannot checkpoint an unprepared or partially failed coupled state")
        self._guard()
        if not self._clock_close(finite_real(self.engine.time_s(), "engine time"), self._time):
            raise RuntimeError("Cavity and engine clocks differ")
        magnetic_states = {s.name: self._full_magnetization(s, s.read_full_magnetization())
                           for s in self.sources}
        payload = {"schema": "complete_coupled_checkpoint_v1",
                   "configuration": self._configuration(), "identity": self._identity,
                   "engine": self._engine_snapshot(), "time_s": self._time,
                   "magnetization_full": magnetic_states,
                   "cavities": {name: {"q": x.q, "p": x.p} for name, x in self._state.items()},
                   "accepted_steps": self.accepted_steps}
        payload["payload_sha256"] = _digest(payload)
        return _json_copy(payload)

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> SynchronousSample:
        """Restore only after exact model/configuration/binary checks.

        The engine callback must restore all parameters, solver/RNG state AND
        time. Source callbacks restore full arrays. Readback verifies all fields.
        If any restoration/readback fails, the adapter remains invalidated.
        """
        record = _json_copy(checkpoint)
        digest = record.pop("payload_sha256", None)
        if digest != _digest(record) or record.get("schema") != "complete_coupled_checkpoint_v1":
            raise ValueError("Invalid or altered complete checkpoint")
        if record.get("configuration") != self._configuration():
            raise ValueError("Checkpoint model, parameters, protocol or time-step configuration differs")
        identity = self._read_identity()
        if record.get("identity") != identity:
            raise ValueError("Checkpoint engine/binary identity differs")
        names = {s.name for s in self.sources}
        if set(record.get("magnetization_full", {})) != names:
            raise ValueError("Checkpoint must contain every complete magnetic state")
        for source in self.sources:
            self._full_magnetization(source, record["magnetization_full"][source.name])
        engine_record = record.get("engine", {})
        if (engine_record.get("complete") is not True or not engine_record.get("parameters")
                or any(k not in engine_record or engine_record[k] is None
                       for k in ("time_s", "solver_state", "stochastic_state"))):
            raise ValueError("Checkpoint lacks complete engine restart state")
        state = {name: Quadratures(**x) for name, x in record.get("cavities", {}).items()}
        self.model.validate_state(state)
        time = finite_real(record.get("time_s"), "checkpoint time")
        if not self._clock_close(finite_real(engine_record["time_s"], "engine checkpoint time"), time):
            raise ValueError("Checkpoint clocks differ")
        steps = record.get("accepted_steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Checkpoint step count must be a nonnegative integer")
        self._broken = True
        try:
            self.engine.restore_full_state(engine_record)
            for source in self.sources:
                source.restore_full_magnetization(record["magnetization_full"][source.name])
            self._identity = identity
            self._guard()
            if not self._clock_close(finite_real(self.engine.time_s(), "engine time"), time):
                raise RuntimeError("Restored engine clock differs")
            for source in self.sources:
                if (self._full_magnetization(source, source.read_full_magnetization())
                        != record["magnetization_full"][source.name]):
                    raise RuntimeError("Full magnetic-state readback differs; exact resume is not established")
            if self._engine_snapshot() != engine_record:
                raise RuntimeError("Engine parameters/solver/RNG readback differs")
            self._state, self._time = state, time
            self.accepted_steps = steps
            self._write_feedback(state)
            self._prepared, self._broken = True, False
            return self._sample()
        except Exception:
            self._prepared = False
            self._broken = True
            raise
