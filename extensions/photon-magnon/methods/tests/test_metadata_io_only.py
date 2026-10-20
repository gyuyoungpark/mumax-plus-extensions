"""Software-only checks: JSON, hashes, atomic I/O, and refused authorizations.

No magnetic/cavity equations, numerical arrays, FFTs, fits or kernels execute.
Temporary files are explicit metadata fixtures, not scientific records.
"""
from pathlib import Path
import tempfile
import unittest

from cavity_method.provenance import (
    ProvenanceError, atomic_write_bytes, atomic_write_json, canonical_json_bytes,
    file_identity, identities, read_checkpoint, read_json, sha256_file,
    verify_analysis_authorization, write_checkpoint,
)


class MetadataIoOnly(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix='cavity_metadata_only_')
        self.root = Path(self._temporary.name).resolve()

    def tearDown(self):
        self._temporary.cleanup()

    def fixture(self, name='fixture.txt', payload=b'METADATA FIXTURE ONLY\n'):
        path = self.root / name
        path.write_bytes(payload)
        return path

    def test_canonical_json_is_deterministic(self):
        self.assertEqual(canonical_json_bytes({'b': 'two', 'a': 'one'}),
                         canonical_json_bytes({'a': 'one', 'b': 'two'}))

    def test_duplicate_json_keys_are_rejected(self):
        path = self.fixture('duplicate.json', b'{"label":"one","label":"two"}')
        with self.assertRaises(ProvenanceError):
            read_json(path)

    def test_nonfinite_json_is_rejected(self):
        path = self.fixture('nonfinite.json', b'{"invalid":NaN}')
        with self.assertRaises(ProvenanceError):
            read_json(path)

    def test_atomic_publication_cannot_overwrite_a_record(self):
        path = self.root / 'published.txt'
        atomic_write_bytes(path, b'original metadata')
        with self.assertRaises(FileExistsError):
            atomic_write_bytes(path, b'replacement metadata')
        self.assertEqual(path.read_bytes(), b'original metadata')

    def test_json_round_trip(self):
        path = self.root / 'record.json'
        value = {'kind': 'metadata_fixture', 'status': 'not_scientific'}
        atomic_write_json(path, value)
        self.assertEqual(read_json(path), value)

    def test_hash_identity_detects_changed_bytes(self):
        path = self.fixture()
        first = file_identity(path)
        path.write_bytes(b'different metadata fixture')
        self.assertNotEqual(first, file_identity(path))

    def test_file_labels_cannot_be_absolute_or_parent_traversals(self):
        path = self.fixture()
        for label in ('../fixture.txt', 'C:/fixture.txt', '/fixture.txt'):
            with self.assertRaises(ProvenanceError):
                identities({label: path})

    def test_checkpoint_refuses_changed_outputs(self):
        source = self.fixture('manifest_input.txt')
        output = self.fixture('output.txt')
        checkpoint = self.root / 'checkpoint.json'
        digest = sha256_file(source)
        write_checkpoint(checkpoint, job_id='METADATA_TEST_ONLY',
                         input_manifest_sha256=digest, outputs={'output.txt': output})
        read_checkpoint(checkpoint, expected_input_manifest_sha256=digest,
                        outputs={'output.txt': output})
        output.write_bytes(b'changed metadata output')
        with self.assertRaises(ProvenanceError):
            read_checkpoint(checkpoint, expected_input_manifest_sha256=digest,
                            outputs={'output.txt': output})

    def test_analysis_requires_explicit_human_approval_record(self):
        raw = self.fixture('raw_fixture.txt')
        manifest = self.fixture('raw_manifest_fixture.txt')
        source = self.fixture('source_fixture.txt')
        with self.assertRaises(ProvenanceError):
            verify_analysis_authorization(
                {}, raw_path=raw, raw_manifest_path=manifest,
                sources={'source_fixture.txt': source}, operations={'full_complex_fft'},
                configuration={'kind': 'metadata_fixture_only'})


if __name__ == '__main__':
    unittest.main()
