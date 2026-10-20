"""Explicit file provenance and atomic I/O. This module uses only the stdlib.

File hashing never imports, loads or executes a native binary. An authorization
file is a review record supplied by the operator, not a cryptographic proof of
human identity. Analysis authorization never authorizes a new simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = 1
TIME_CONVENTION = "exp(-i 2*pi*f*t)"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ANALYSIS_OPERATIONS = frozenset(
    {"full_complex_fft", "peak_selection", "branch_coverage"}
)


class ProvenanceError(ValueError):
    """A required identity, declaration or authorization is missing/invalid."""


def _json_value(value: Any, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProvenanceError(f"Non-finite JSON number at {location}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _json_value(item, f"{location}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(f"Non-string JSON key at {location}")
            _json_value(item, f"{location}.{key}")
        return
    raise ProvenanceError(f"Unsupported JSON type at {location}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    _json_value(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProvenanceError(f"{label} must be a lowercase SHA256 digest")
    return value


def sha256_file(path: str | Path) -> str:
    """Hash an explicitly named ordinary file, without loading/executing it."""
    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"Required file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {"sha256": sha256_file(source), "bytes": source.stat().st_size}


def _label(label: str) -> str:
    """Keep labels portable; do not automatically disclose machine paths."""
    if (not isinstance(label, str) or not label or "\\" in label or ":" in label
            or PurePosixPath(label).is_absolute()
            or any(part in {".", ".."} for part in PurePosixPath(label).parts)):
        raise ProvenanceError("File labels must be portable relative names")
    return label


def identities(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    if not paths:
        raise ProvenanceError("At least one explicit file identity is required")
    return {_label(label): file_identity(path) for label, path in sorted(paths.items())}


def parse_json_bytes(payload: bytes) -> dict[str, Any]:
    """Parse held bytes; reject duplicate keys, non-finite values and non-objects."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProvenanceError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ProvenanceError(f"Non-finite JSON constant: {value}")

    try:
        result = json.loads(payload.decode("utf-8"),
                            object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("Cannot decode JSON document") from exc
    if not isinstance(result, dict):
        raise ProvenanceError("JSON document must be an object")
    _json_value(result)
    return result


def read_json_with_identity(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read once, parse and identify exactly the same retained bytes."""
    payload = Path(path).read_bytes()
    return parse_json_bytes(payload), {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def read_json(path: str | Path) -> dict[str, Any]:
    return read_json_with_identity(path)[0]


def read_verified_json(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    require_sha256(expected_sha256, "expected_sha256")
    document, identity = read_json_with_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ProvenanceError("JSON input changed or differs from its reviewed hash")
    return document


def configuration_sha256(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes, *, overwrite: bool = False) -> None:
    """Publish a fully flushed neighboring temporary file atomically.

    Without overwrite, an atomic hard-link publication refuses an existing
    destination, including in a concurrent-writer race. Unsupported hard links
    fail closed. Only the exact temporary file created here is removed.
    """
    target = Path(path)
    if not target.parent.is_dir():
        raise ProvenanceError("Output parent must already exist")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.",
                                          suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def build_manifest(*, record_kind: str, sources: Mapping[str, str | Path],
                   binaries: Mapping[str, str | Path], units: Mapping[str, str],
                   precision: str, parameters: Mapping[str, Any],
                   raw_files: Mapping[str, str | Path],
                   native_binary_use: str,
                   sampling: Mapping[str, Any],
                   signal_declarations: Mapping[str, Any]) -> dict[str, Any]:
    """Build declared provenance from explicit inputs; discover no binaries/env.

    raw_trajectory requires named native binaries. stored_raw_analysis must
    declare native_binary_use='not_executed' and does not silently load one.
    """
    if record_kind not in {"raw_trajectory", "stored_raw_analysis"}:
        raise ProvenanceError("Unknown record_kind")
    if precision not in {"float32", "float64", "complex64", "complex128", "mixed"}:
        raise ProvenanceError("An explicit numeric precision is required")
    if units.get("time") != "s" or not units.get("signal"):
        raise ProvenanceError("Units must declare time='s' and a signal unit")
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in units.items()):
        raise ProvenanceError("Unit declarations must be non-empty strings")
    if record_kind == "raw_trajectory" and (not binaries or native_binary_use != "declared"):
        raise ProvenanceError("Raw simulation provenance needs explicit binary identities")
    if record_kind == "stored_raw_analysis" and native_binary_use != "not_executed":
        raise ProvenanceError("Analysis provenance must not claim native execution")
    _json_value(parameters)
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": record_kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": identities(sources),
        "binaries": identities(binaries) if binaries else {},
        "native_binary_use": native_binary_use,
        "units": dict(units),
        "precision": precision,
        "sampling": dict(sampling),
        "signal_declarations": dict(signal_declarations),
        "parameters": dict(parameters),
        "time_convention": TIME_CONVENTION,
        "raw_files": identities(raw_files),
        "python_version": sys.version.split()[0],
    }
    validate_manifest(result)
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _json_value(manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError("Unsupported manifest schema")
    if manifest.get("record_kind") not in {"raw_trajectory", "stored_raw_analysis"}:
        raise ProvenanceError("Manifest must identify its record kind")
    if manifest.get("time_convention") != TIME_CONVENTION:
        raise ProvenanceError("Missing/unsupported complex carrier convention")
    units = manifest.get("units")
    if not isinstance(units, Mapping) or units.get("time") != "s" or not units.get("signal"):
        raise ProvenanceError("Manifest must declare time and signal units")
    if not all(isinstance(key, str) and isinstance(value, str) and value
               for key, value in units.items()):
        raise ProvenanceError("Every manifest unit must be a non-empty string")
    if manifest.get("precision") not in {"float32", "float64", "complex64", "complex128", "mixed"}:
        raise ProvenanceError("Manifest must declare precision")
    if not isinstance(manifest.get("parameters"), Mapping):
        raise ProvenanceError("Manifest must declare parameters")
    sampling = manifest.get("sampling")
    if (not isinstance(sampling, Mapping)
            or sampling.get("time_dtype") not in {"float32", "float64"}
            or sampling.get("signal_dtype") not in {"complex64", "complex128"}
            or sampling.get("timestamps_interpolated") is not False):
        raise ProvenanceError("Manifest must preserve explicit raw time/signal dtypes and declare no interpolation")
    for group in ("sources", "raw_files", "binaries"):
        entries = manifest.get(group)
        if not isinstance(entries, Mapping) or (group != "binaries" and not entries):
            raise ProvenanceError(f"Missing manifest identity group: {group}")
        for label, item in entries.items():
            _label(label)
            if not isinstance(item, Mapping):
                raise ProvenanceError(f"Invalid file identity: {label}")
            require_sha256(item.get("sha256"), label)
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                raise ProvenanceError(f"Invalid byte size: {label}")
    if manifest["record_kind"] == "raw_trajectory":
        if not manifest["binaries"] or manifest.get("native_binary_use") != "declared":
            raise ProvenanceError("Raw trajectory manifest lacks native binary provenance")
    elif manifest.get("native_binary_use") != "not_executed":
        raise ProvenanceError("Analysis manifest must declare no native execution")
    declarations = manifest.get("signal_declarations")
    if not isinstance(declarations, Mapping) or not declarations:
        raise ProvenanceError("Explicit raw signal declarations are required")
    for name, signal in declarations.items():
        _label(name)
        if (not isinstance(signal, Mapping) or signal.get("raw_label") not in manifest["raw_files"]
                or not isinstance(signal.get("time_key"), str) or not signal["time_key"]
                or signal.get("carrier_convention") != TIME_CONVENTION
                or signal.get("signal_unit") != units["signal"]):
            raise ProvenanceError("Raw signal declaration lacks keys, units or carrier convention")
        complex_key = signal.get("complex_key")
        split = signal.get("real_key") is not None or signal.get("imaginary_key") is not None
        if bool(complex_key) == bool(split):
            raise ProvenanceError("Signal declaration needs exactly complex storage or split quadratures")
        if complex_key:
            if not isinstance(complex_key, str) or signal.get("imaginary_sign") is not None:
                raise ProvenanceError("Invalid assembled complex signal declaration")
        elif (not isinstance(signal.get("real_key"), str) or not signal["real_key"]
              or not isinstance(signal.get("imaginary_key"), str) or not signal["imaginary_key"]
              or type(signal.get("imaginary_sign")) is not int
              or signal["imaginary_sign"] not in {-1, 1}):
            raise ProvenanceError("Split signal declaration requires both keys and an explicit sign")


def verify_raw_manifest(manifest: Mapping[str, Any], *, raw_path: str | Path,
                        raw_label: str) -> None:
    validate_manifest(manifest)
    if manifest.get("record_kind") != "raw_trajectory":
        raise ProvenanceError("A raw trajectory manifest is required, not a spectrum/analysis receipt")
    expected = manifest["raw_files"].get(_label(raw_label))
    if expected != file_identity(raw_path):
        raise ProvenanceError("Raw file hash/size does not match the declared manifest")


def verify_analysis_authorization(
    authorization: Mapping[str, Any], *, raw_path: str | Path,
    raw_manifest_path: str | Path, sources: Mapping[str, str | Path],
    operations: set[str], configuration: Mapping[str, Any],
    additional_inputs: Mapping[str, str | Path] | None = None,
) -> None:
    """Fail closed unless the explicit review record matches every exact input.

    Required JSON fields are documented in analysis/README.txt. No function here
    grants authorization, reads credentials or executes analysis/simulations.
    """
    _json_value(authorization)
    if (authorization.get("schema_version") != SCHEMA_VERSION
            or authorization.get("authorization_kind") != "stored_raw_analysis"
            or authorization.get("new_simulation_authorized") is not False):
        raise ProvenanceError("Authorization must be restricted to stored raw analysis")
    if not isinstance(authorization.get("approval_reference"), str) or not authorization["approval_reference"].strip():
        raise ProvenanceError("An explicit human approval reference is required")
    declared_ops = authorization.get("operations")
    if (not isinstance(declared_ops, list) or not declared_ops
            or any(not isinstance(x, str) or x not in ANALYSIS_OPERATIONS for x in declared_ops)
            or len(set(declared_ops)) != len(declared_ops)
            or not operations <= set(declared_ops)):
        raise ProvenanceError("The requested analysis operations were not explicitly authorized")
    if not operations or not operations <= ANALYSIS_OPERATIONS:
        raise ProvenanceError("Unknown/empty requested analysis operation")
    if authorization.get("raw_input_sha256") != sha256_file(raw_path):
        raise ProvenanceError("Authorization raw hash mismatch")
    if authorization.get("raw_manifest_sha256") != sha256_file(raw_manifest_path):
        raise ProvenanceError("Authorization raw manifest hash mismatch")
    expected_sources = {label: item["sha256"] for label, item in identities(sources).items()}
    if authorization.get("analysis_sources_sha256") != expected_sources:
        raise ProvenanceError("Authorization source hashes do not match this implementation")
    if authorization.get("analysis_configuration_sha256") != configuration_sha256(configuration):
        raise ProvenanceError("Authorization analysis configuration does not match")
    expected_extra = ({label: item["sha256"] for label, item in identities(additional_inputs).items()}
                      if additional_inputs else {})
    if authorization.get("additional_inputs_sha256", {}) != expected_extra:
        raise ProvenanceError("Authorization additional input hashes do not match")


def write_checkpoint(path: str | Path, *, job_id: str, input_manifest_sha256: str,
                     outputs: Mapping[str, str | Path], allow_update: bool = False) -> None:
    """Atomic, explicit I/O checkpoint. It contains no executable resume action."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ProvenanceError("A non-empty job_id is required")
    require_sha256(input_manifest_sha256, "input_manifest_sha256")
    target = Path(path)
    updating = target.exists()
    if updating:
        if not allow_update:
            raise ProvenanceError("Checkpoint already exists; explicit allow_update is required")
        old = read_json(target)
        if (old.get("schema_version") != SCHEMA_VERSION
                or old.get("kind") != "analysis_io_checkpoint"
                or old.get("job_id") != job_id
                or old.get("input_manifest_sha256") != input_manifest_sha256):
            raise ProvenanceError("Existing checkpoint does not belong to this job/input; replacement refused")
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "kind": "analysis_io_checkpoint",
        "job_id": job_id, "input_manifest_sha256": input_manifest_sha256,
        "outputs": identities(outputs),
        "resume_behavior": "verify identities only; execute nothing",
    }
    atomic_write_json(path, checkpoint, overwrite=updating)


def read_checkpoint(path: str | Path, *, expected_input_manifest_sha256: str,
                    outputs: Mapping[str, str | Path]) -> dict[str, Any]:
    checkpoint = read_json(path)
    require_sha256(expected_input_manifest_sha256, "expected_input_manifest_sha256")
    if (checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("kind") != "analysis_io_checkpoint"
            or checkpoint.get("input_manifest_sha256") != expected_input_manifest_sha256
            or checkpoint.get("outputs") != identities(outputs)):
        raise ProvenanceError("Checkpoint input/output identity mismatch; resume refused")
    return checkpoint
