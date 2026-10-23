import hashlib

import pytest

from prepare_easyocr_models import ensure_model, file_md5


def test_reuses_only_verified_existing_easyocr_weight(tmp_path):
    payload = b'official test weight'
    target = tmp_path / 'test.pth'
    target.write_bytes(payload)
    expected = hashlib.md5(payload).hexdigest()
    assert file_md5(target) == expected
    assert ensure_model(tmp_path, 'test.pth', 'https://example.invalid/test.zip', expected) == target
    with pytest.raises(FileExistsError, match='wrong hash'):
        ensure_model(tmp_path, 'test.pth', 'https://example.invalid/test.zip', '0' * 32)
