"""Local preparation checks: synthetic manifests and ZIP members only."""
import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from prepare_phase2_local import download_archive, inspect_archive, validate_source_manifest, check_image


def source_fixture(path):
    row = {'question': 'What does the label say?', 'answers': [{'answer': 'tea'}] * 10,
            'answerable': 1, 'answer_type': 'other', 'image': 'VizWiz_val_00000000.jpg'}
    manifest = {'dataset': 'official VizWiz-VQA', 'reference_policy': 'vizwiz-raw-ten-slots-valid-targets-v1',
                'train': [{}] * 5000, 'dev': [dict(row, image=f'VizWiz_val_{i:08d}.jpg') for i in range(1000)],
                # Count only in preparation. These test strings must never be reported/selected.
                'holdout': [{'image': f'HELDOUT-NEVER-USE-{i}'} for i in range(3319)]}
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def test_source_manifest_uses_exact_dev_and_keeps_raw_references(tmp_path):
    path = tmp_path / 'split_manifest.json'
    source = source_fixture(path)
    payload, audit = validate_source_manifest(path)
    assert audit['counts'] == {'train': 5000, 'dev': 1000, 'holdout': 3319}
    assert len(payload['dev']) == 1000
    assert audit['dev_ids'][0] == 'VizWiz_val_00000000.jpg'
    assert audit['dev_ids'][-1] == 'VizWiz_val_00000999.jpg'
    assert payload['dev'][0]['answers'] == source['dev'][0]['answers']
    assert len(payload['dev'][0]['answers']) == 10
    assert not any(name.startswith('HELDOUT') for name in audit['dev_ids'])


@pytest.mark.parametrize('mutation', ['counts', 'split_policy', 'duplicate', 'unsafe_name', 'bad_answers'])
def test_invalid_manifest_stops_before_images_or_model(tmp_path, mutation):
    path = tmp_path / 'split_manifest.json'
    payload = source_fixture(path)
    if mutation == 'counts':
        payload['dev'].pop()
    elif mutation == 'split_policy':
        payload['reference_policy'] = 'new-random-split'
    elif mutation == 'duplicate':
        payload['dev'][1]['image'] = payload['dev'][0]['image']
    elif mutation == 'unsafe_name':
        payload['dev'][0]['image'] = '..\\holdout.jpg'
    else:
        payload['dev'][0]['answers'] = [{'answer': 'invented'}]
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        validate_source_manifest(path)


def test_archive_requires_exact_unique_original_dev_filenames(tmp_path):
    names = ['VizWiz_val_00000001.jpg', 'VizWiz_val_00000002.jpg']
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w') as archive:
        for name in names:
            archive.writestr('val/' + name, b'not-a-valid-image-but-a-member')
    memory.seek(0)
    from pathlib import Path
    path = tmp_path / 'archive.zip'
    path.write_bytes(memory.getvalue())
    assert inspect_archive(path, names) == 2
    with pytest.raises(ValueError, match='found 0'):
        inspect_archive(path, ['absent.jpg'])


def test_archive_duplicate_and_zero_length_names_rejected(tmp_path):
    names = ['VizWiz_val_00000001.jpg']
    path = tmp_path / 'archive.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('first/' + names[0], b'jpeg')
        archive.writestr('second/' + names[0], b'jpeg')
    with pytest.raises(ValueError, match='exactly one'):
        inspect_archive(path, names)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('first/' + names[0], b'')
    with pytest.raises(ValueError, match='Empty'):
        inspect_archive(path, names)


def test_progress_downloader_preserves_final_and_checks_short_content(tmp_path):
    class Response(io.BytesIO):
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as archive:
        archive.writestr('VizWiz_val_00000001.jpg', b'jpeg')
    data = payload.getvalue()
    class ZipResponse(Response):
        headers = {'Content-Length': str(len(data))}
    with tempfile.TemporaryDirectory(prefix='.test-vizwiz-download-', dir=Path(__file__).resolve().parents[1]) as work:
        target = Path(work) / 'val.zip'
        path = download_archive(target, url='https://example.test/val.zip', opener=lambda *a, **k: ZipResponse(data))
        assert path == target and zipfile.is_zipfile(path)
        reused = download_archive(target, url='https://example.test/val.zip',
                                  opener=lambda *a, **k: (_ for _ in ()).throw(AssertionError('must reuse verified archive')))
        assert reused == target


def test_image_readability_and_corruption(tmp_path):
    good = io.BytesIO()
    Image.new('RGB', (10, 8), 'white').save(good, format='JPEG')
    good.seek(0)
    path = tmp_path / 'valid.jpg'
    path.write_bytes(good.getvalue())
    check_image(path)
    broken = tmp_path / 'broken.jpg'
    broken.write_bytes(b'not an image')
    with pytest.raises(Exception):
        check_image(broken)
