"""Offline replay integrity tests with synthetic 1,000-row fixtures."""
import copy
import json
from unittest.mock import Mock

import pytest
from test_phase2_dev import payload, captured, rules
from evaluate_phase2_dev import replay
from evaluate_vqa import OfficialScorer
from replay_phase2_dev import reselect_capture, run_replay
from modules.evidence_store import digest


def test_reselection_cannot_receive_references_and_does_not_mutate():
    p = payload()
    c = captured(p['rows'][0])
    before = copy.deepcopy(c)
    reselect_capture(c)
    assert c == before
    c['answers'] = p['rows'][0]['answers']
    with pytest.raises(ValueError, match='raw inference'):
        reselect_capture(c)


def test_full_synthetic_replay_no_inference_and_no_overwrite(tmp_path, monkeypatch):
    p = payload()
    source = tmp_path / 'source'
    source.mkdir()
    manifest = source / 'dev.json'
    manifest.write_text(json.dumps(p), encoding='utf-8')
    captures = source / 'captures'
    captures.mkdir()
    comparison = source / 'comparison'
    comparison.mkdir()
    scorer = OfficialScorer(rules())
    originals = []
    for index, row in enumerate(p['rows']):
        c = captured(row)
        (captures / f'{index:04d}.json').write_text(json.dumps({'capture': c, 'sha256': digest(c)}), encoding='utf-8')
        originals.append(replay(row, c, scorer))
    for k in 'AB':
        (comparison / f'{k}_records.json').write_text(json.dumps(originals), encoding='utf-8')
    trap = Mock(side_effect=AssertionError('No inference allowed'))
    monkeypatch.setattr('modules.vqa.answer', trap)
    monkeypatch.setattr('modules.evidence_store.extract', trap)
    monkeypatch.setattr('modules.ocr.extract', trap)
    before = manifest.read_bytes()
    output = tmp_path / 'replay-v1'
    run_replay(manifest, captures, comparison, output)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['targeted']['examples'] == 1000
    assert summary['targeted']['beneficial_overrides'] == 1000
    assert summary['gpu_inference_calls'] == 0
    assert manifest.read_bytes() == before
    trap.assert_not_called()
    with pytest.raises(FileExistsError):
        run_replay(manifest, captures, comparison, output)


def test_disabled_ocr_after_router_change_is_not_invented():
    p = payload()
    row = p['rows'][0]
    row['question'] = 'Can you tell me what this says, please?'
    c = captured(row, evidence=[])
    c['ocr_supported_question'] = False
    c['extraction_status']['ocr'] = {'status': 'disabled'}
    updated = reselect_capture(c)
    result = replay(row, updated, OfficialScorer(rules()), .9)
    assert result['ocr_supported_question']
    assert result['answer_source'] == 'blip'
    assert 'not extracted' in result['decision_reason']
