"""Standard-library metadata/I/O checks. No scientific array or FFT operation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cavity_method import provenance as p


class AnalysisMetadataIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.txt"
        self.binary = self.root / "explicit_binary.bin"
        self.raw = self.root / "raw.bin"
        self.source.write_bytes(b"read-only source identity")
        self.binary.write_bytes(b"not loaded or executed")
        self.raw.write_bytes(b"opaque I/O test payload; not scientific data")

    def manifest(self):
        return p.build_manifest(
            record_kind="raw_trajectory", sources={"source.txt": self.source},
            binaries={"native.bin": self.binary}, units={"time": "s", "signal": "1"},
            precision="mixed", parameters={"declared": True},
            sampling={"time_dtype": "float32", "signal_dtype": "complex128", "timestamps_interpolated": False},
            signal_declarations={"cavity": {"raw_label": "raw.bin", "time_key": "time", "complex_key": "a",
                                            "carrier_convention": p.TIME_CONVENTION, "signal_unit": "1"}},
            raw_files={"raw.bin": self.raw}, native_binary_use="declared")

    def test_hash_is_content_only(self):
        self.assertEqual(p.sha256_file(self.binary), hashlib.sha256(self.binary.read_bytes()).hexdigest())

    def test_json_rejects_nonfinite_values(self):
        with self.assertRaises(p.ProvenanceError):
            p.canonical_json_bytes({"invalid": float("nan")})

    def test_json_rejects_duplicate_keys(self):
        path = self.root / "duplicate.json"
        path.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaises(p.ProvenanceError):
            p.read_json(path)

    def test_atomic_output_refuses_overwrite(self):
        path = self.root / "output.json"
        p.atomic_write_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            p.atomic_write_json(path, {"replacement": True})
        self.assertEqual(p.read_json(path), {"original": True})
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_manifest_has_explicit_raw_sampling_and_native_identity(self):
        manifest = self.manifest()
        p.validate_manifest(manifest)
        p.verify_raw_manifest(manifest, raw_path=self.raw, raw_label="raw.bin")
        self.assertEqual(manifest["sampling"]["time_dtype"], "float32")
        self.assertEqual(manifest["time_convention"], p.TIME_CONVENTION)
        self.assertNotIn(str(self.root), p.canonical_json_bytes(manifest).decode())

    def test_raw_manifest_refuses_missing_binary_or_sampling_identity(self):
        for group in ("binaries", "sampling"):
            manifest = self.manifest()
            manifest[group] = {}
            with self.assertRaises(p.ProvenanceError):
                p.validate_manifest(manifest)

    def test_checkpoint_verifies_exact_input_and_output_hashes(self):
        path = self.root / "checkpoint.json"
        digest = "a" * 64
        p.write_checkpoint(path, job_id="io-only", input_manifest_sha256=digest,
                           outputs={"raw.bin": self.raw})
        p.read_checkpoint(path, expected_input_manifest_sha256=digest, outputs={"raw.bin": self.raw})
        self.raw.write_bytes(b"changed")
        with self.assertRaises(p.ProvenanceError):
            p.read_checkpoint(path, expected_input_manifest_sha256=digest, outputs={"raw.bin": self.raw})

    def authorization(self):
        manifest_path = self.root / "manifest.json"
        p.atomic_write_json(manifest_path, self.manifest())
        auth = {"schema_version": 1, "authorization_kind": "stored_raw_analysis",
                "new_simulation_authorized": False, "approval_reference": "I/O fixture only; executes no science",
                "operations": ["full_complex_fft"],
                "raw_input_sha256": p.sha256_file(self.raw),
                "raw_manifest_sha256": p.sha256_file(manifest_path),
                "analysis_sources_sha256": {"source.txt": p.sha256_file(self.source)},
                "analysis_configuration_sha256": p.configuration_sha256({"fixture": "metadata only"}),
                "additional_inputs_sha256": {}}
        return manifest_path, auth

    def test_authorization_is_exact_and_analysis_only(self):
        manifest_path, auth = self.authorization()
        p.verify_analysis_authorization(auth, raw_path=self.raw, raw_manifest_path=manifest_path,
                                         sources={"source.txt": self.source}, operations={"full_complex_fft"},
                                         configuration={"fixture": "metadata only"})
        auth["new_simulation_authorized"] = True
        with self.assertRaises(p.ProvenanceError):
            p.verify_analysis_authorization(auth, raw_path=self.raw, raw_manifest_path=manifest_path,
                                             sources={"source.txt": self.source}, operations={"full_complex_fft"},
                                             configuration={"fixture": "metadata only"})

    def test_authorization_refuses_source_changes(self):
        manifest_path, auth = self.authorization()
        self.source.write_bytes(b"different source")
        with self.assertRaises(p.ProvenanceError):
            p.verify_analysis_authorization(auth, raw_path=self.raw, raw_manifest_path=manifest_path,
                                             sources={"source.txt": self.source}, operations={"full_complex_fft"},
                                             configuration={"fixture": "metadata only"})

    def test_authorization_refuses_changed_configuration(self):
        manifest_path, auth = self.authorization()
        with self.assertRaises(p.ProvenanceError):
            p.verify_analysis_authorization(auth, raw_path=self.raw, raw_manifest_path=manifest_path,
                sources={"source.txt": self.source}, operations={"full_complex_fft"},
                configuration={"fixture": "different assembly/window/tolerance"})

    def test_checkpoint_update_requires_same_job_and_explicit_opt_in(self):
        path = self.root / "checkpoint.json"
        options = dict(job_id="first", input_manifest_sha256="a" * 64, outputs={"raw.bin": self.raw})
        p.write_checkpoint(path, **options)
        with self.assertRaises(p.ProvenanceError):
            p.write_checkpoint(path, **options)
        p.write_checkpoint(path, allow_update=True, **options)
        with self.assertRaises(p.ProvenanceError):
            p.write_checkpoint(path, job_id="other", input_manifest_sha256="a" * 64,
                               outputs={"raw.bin": self.raw}, allow_update=True)


class AdapterMetadataGuardTests(unittest.TestCase):
    """Inert callback/metadata regressions; no magnetic or cavity RHS executes."""

    def hooks(self, name="fm", *, host=None, partners=(), kind="fm", average=None):
        from cavity_method.adapters import UniformSourceHooks
        return UniformSourceHooks(
            name=name, read_average=average or (lambda: (0.0, 0.0, 1.0)),
            read_full_magnetization=lambda: [[0.0], [0.0], [1.0]],
            restore_full_magnetization=lambda payload: None,
            set_local_bias_t=lambda value: None,
            base_bias_t=(0.0, 0.0, 0.0), guard_objects=(object(),),
            source_kind=kind, magnetization_shape=(3, 1),
            host_guard_object=host, required_partner_sources=partners)

    def adapter(self, sources):
        from types import SimpleNamespace
        from cavity_method.adapters import EngineHooks, UniformCosimulationAdapter
        from cavity_method.energy import Quadratures

        def forbidden_advance(duration):
            raise AssertionError("A metadata regression must never advance an engine")

        # Deliberately not an EnergyCavityModel: these callbacks supply inert
        # metadata, so no physical source, field, energy or numerical RHS runs.
        model = SimpleNamespace(
            modes=(SimpleNamespace(name="fixture", profile_id="uniform"),),
            sources=tuple(SimpleNamespace(name=s.name) for s in sources),
            validate_state=lambda value: None,
            manifest=lambda: {"fixture": "no physical model"},
            uniform_feedback_x_t=lambda state: 0.0)
        engine = EngineHooks(
            time_s=lambda: 0.0, advance=forbidden_advance,
            inspect_identity=lambda: {
                "engine_name": "INERT_METADATA_FIXTURE_ONLY",
                "engine_version": "fixture", "source_revision": "fixture",
                "binary_sha256": "0" * 64, "precision": "metadata only",
                "version_verified": True},
            capture_full_state=lambda: {
                "complete": True, "time_s": 0.0,
                "parameters": {"fixture": "no engine"},
                "solver_state": {"fixture": "never advanced"},
                "stochastic_state": "disabled"},
            restore_full_state=lambda payload: None,
            complete_state_supported=True)
        return UniformCosimulationAdapter(
            model, engine, tuple(sources), initial_state={"fixture": Quadratures(0.0, 0.0)},
            macrostep_s=1e-9, preregistration_id="INERT_METADATA_FIXTURE_ONLY",
            source_freeze_sha256="0" * 64)

    def test_failed_final_restore_sample_requires_full_restore(self):
        fail = [False]

        def average():
            if fail[0]:
                raise ValueError("injected metadata readback failure")
            return (0.0, 0.0, 1.0)

        adapter = self.adapter([self.hooks(average=average)])
        adapter.prepare()
        checkpoint = adapter.checkpoint()
        fail[0] = True
        with self.assertRaisesRegex(ValueError, "injected metadata"):
            adapter.restore_checkpoint(checkpoint)
        self.assertFalse(adapter._prepared)
        self.assertTrue(adapter._broken)
        with self.assertRaisesRegex(RuntimeError, "full checkpoint"):
            adapter.prepare()

    def test_afm_empty_partner_declaration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.hooks("a", host=object(), partners=(), kind="afm_sublattice")

    def test_afm_self_partner_is_rejected(self):
        with self.assertRaises(ValueError):
            self.hooks("a", host=object(), partners=("a",), kind="afm_sublattice")

    def test_afm_duplicate_partners_are_rejected(self):
        with self.assertRaises(ValueError):
            self.hooks("a", host=object(), partners=("b", "b"), kind="afm_sublattice")

    def test_afm_partner_must_be_on_the_same_host(self):
        a = self.hooks("a", host=object(), partners=("b",), kind="afm_sublattice")
        b = self.hooks("b", host=object(), partners=("a",), kind="afm_sublattice")
        with self.assertRaisesRegex(ValueError, "same host"):
            self.adapter([a, b])

    def test_afm_partner_must_be_an_afm_binding(self):
        a = self.hooks("a", host=object(), partners=("b",), kind="afm_sublattice")
        with self.assertRaisesRegex(ValueError, "same host"):
            self.adapter([a, self.hooks("b")])

    def test_afm_declaration_must_cover_all_bound_same_host_peers(self):
        host = object()
        a = self.hooks("a", host=host, partners=("b",), kind="afm_sublattice")
        b = self.hooks("b", host=host, partners=("a", "c"), kind="afm_sublattice")
        c = self.hooks("c", host=host, partners=("a", "b"), kind="afm_sublattice")
        with self.assertRaisesRegex(ValueError, "every distinct"):
            self.adapter([a, b, c])

    def test_distinct_same_host_afm_metadata_group_is_accepted(self):
        host = object()
        a = self.hooks("a", host=host, partners=("b",), kind="afm_sublattice")
        b = self.hooks("b", host=host, partners=("a",), kind="afm_sublattice")
        adapter = self.adapter([a, b])
        self.assertFalse(adapter._prepared)


if __name__ == "__main__":
    unittest.main()
