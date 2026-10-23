"""Public setup tests use synthetic files, never dataset rows or model inference."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import public_runtime as runtime
from public_models import install_bytes


class PublicRuntimeTests(unittest.TestCase):
    def test_manifest_is_anchored(self):
        self.assertEqual(runtime.read_manifest()['schema'], 1)

    def test_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runtime/manifest.json'
            path.parent.mkdir()
            path.write_text('{}')
            with self.assertRaises(ValueError):
                runtime.read_manifest(tmp)

    def test_unsafe_paths_rejected(self):
        for name in ('../x', '/tmp/x', 'C:/x', 'a\\b', 'a//b', './x'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.safe_path(runtime.ROOT, name)

    def test_install_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model'
            install_bytes(p, b'valid', hashlib.sha256(b'valid').hexdigest())
            install_bytes(p, b'valid', hashlib.sha256(b'valid').hexdigest())
            with self.assertRaises(FileExistsError):
                install_bytes(p, b'other', hashlib.sha256(b'other').hexdigest())
            self.assertEqual(p.read_bytes(), b'valid')

    def test_invalid_download_is_never_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model'
            with self.assertRaises(ValueError):
                install_bytes(p, b'wrong', '0' * 64)
            self.assertFalse(p.exists())

    def test_runtime_contains_no_evaluation_rows(self):
        for name in ('manifest.json', 'policy.json', 'normalization.json', 'classifier/metadata.json'):
            payload = (runtime.ROOT / 'runtime' / name).read_text()
            for forbidden in ('VizWiz_train_', 'VizWiz_val_', 'C:\\\\Users\\\\', 'reference_answers'):
                self.assertNotIn(forbidden, payload)
        for row in runtime.read_manifest()['assets']:
            self.assertFalse(row['path'].startswith('reports/'))

    def test_frozen_policy_values(self):
        policy = json.loads((runtime.ROOT / 'runtime/policy.json').read_text())
        self.assertEqual((policy['threshold_answerable'], policy['threshold_reliable'], policy['ocr_threshold']), (0., .05, .9))

    def test_reuses_original_decision_methods(self):
        from phase4_pipeline import Phase4Pipeline
        from phase3_pipeline import Phase3Pipeline
        self.assertIs(Phase4Pipeline.run.__globals__['Phase3Pipeline'], Phase3Pipeline)
        source = ast.parse((runtime.ROOT / 'public_runtime.py').read_text())
        self.assertFalse(any(isinstance(n, ast.ClassDef) for n in source.body))

    def test_public_cache_guard_runs_before_followup(self):
        from public_runtime_session import PublicSession
        from PIL import Image
        from image_conversation import prepare
        image = Image.new('RGB', (20, 20), 'white')
        session = PublicSession.__new__(PublicSession)
        with patch('public_runtime.verify') as verifier, patch('public_runtime.guarded_paths', return_value=[]):
            session.for_conversation(prepare({}, image), image)
            session.for_conversation(prepare({}, image), image)
            self.assertEqual(verifier.call_count, 1)
        session._public_guard.check()

    def test_no_split_verification_in_public_adapter(self):
        source = (runtime.ROOT / 'public_runtime.py').read_text()
        self.assertNotIn('checked_splits(', source)
        self.assertNotIn('verify_freeze(', source)


if __name__ == '__main__':
    unittest.main()
