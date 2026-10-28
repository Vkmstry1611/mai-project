"""Synthetic local setup checks; no downloads or real model inference."""
import io
import json
import zipfile
from unittest.mock import patch

from PIL import Image
import pytest
from test_phase2_dev import payload, row, captured, rules
from evaluate_vqa import OfficialScorer
from evaluate_phase2_dev import replay
from prepare_phase2_local import prepare, download_archive
from visionbridge.local_phase2_runner import verify_results


def test_extract_exact_dev_only_and_preserve_completed_run(tmp_path,monkeypatch):
    # Current preparation takes an original source manifest, rules, archive and
    # NEW run directory. Do not restore the obsolete two-argument/reuse API.
    import prepare_phase2_local as setup
    from test_phase2_local_setup import source_fixture
    manifest = tmp_path / 'split_manifest.json'
    source=source_fixture(manifest)
    p={'rows':source['dev']}
    scoring=tmp_path/'rules.json';scoring.write_text(json.dumps(rules()))
    monkeypatch.setattr(setup,'REPORT_ROOT',tmp_path/'runs')
    monkeypatch.setattr(setup,'verify_checkpoint',lambda path:{'fixture':'synthetic'})
    monkeypatch.setattr(setup,'device_check',lambda:None)
    data = tmp_path / 'data'
    data.mkdir()
    buffer = io.BytesIO()
    Image.new('RGB', (10, 10)).save(buffer, format='JPEG')
    with zipfile.ZipFile(data / 'val.zip', 'w') as archive:
        for r in p['rows']:
            archive.writestr('val/' + r['image'], buffer.getvalue())
        archive.writestr('val/VizWiz_val_99999999.jpg', b'NOT AN IMAGE: must never be opened')
    run=tmp_path/'runs'/'synthetic-run'
    state=prepare(manifest,scoring,data/'val.zip',run)
    assert state['dev_examples']==1000
    images=run/'data'/'images'
    assert len(list(images.glob('*.jpg'))) == 1000
    assert not (images / 'VizWiz_val_99999999.jpg').exists()
    before = (images / p['rows'][0]['image']).read_bytes()
    with pytest.raises(FileExistsError,match='not empty'):
        prepare(manifest,scoring,data/'val.zip',run)
    assert (images / p['rows'][0]['image']).read_bytes() == before


def test_invalid_existing_archive_preserved(tmp_path):
    path = tmp_path / 'val.zip'
    path.write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='invalid'):
        download_archive(path)
    assert path.read_bytes() == b'incomplete'


def test_verify_pilot_references_and_extraction_failures(tmp_path):
    rows = [row(i) for i in range(20)]
    scorer = OfficialScorer(rules())
    records = [replay(r, captured(r), scorer, .9) for r in rows]
    for n in ('A', 'B', 'C'):
        (tmp_path / f'{n}_records.json').write_text(json.dumps(records))
    assert len(verify_results(tmp_path, rows, 'pilot')['A']) == 20
    records[0]['extraction_status']['ocr']['status'] = 'unavailable'
    (tmp_path / 'B_records.json').write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match='Extraction failures'):
        verify_results(tmp_path, rows, 'pilot')
    records[0]['answers'] = [{'answer': 'changed'}] * 10
    (tmp_path / 'B_records.json').write_text(json.dumps(records))
    with pytest.raises(AssertionError, match='references changed'):
        verify_results(tmp_path, rows, 'pilot')
