import io
import json
import zipfile
import pytest
from phase3_common import read_selected_splits,validate_splits
from prepare_phase3_data import member_bytes
from evaluate_phase3_dev import confusion_metrics,select_candidate


def test_split_parser_never_decodes_holdout_value(tmp_path,monkeypatch):
    import phase3_common
    text='{"train": [], "holdout": [{"private": "do not decode"}], "dev": []}'
    path=tmp_path/'manifest.json';path.write_text(text)
    original=json.JSONDecoder.raw_decode
    def guarded(self,s,idx=0):
        if s[idx:].startswith('[{"private"'):raise AssertionError('Holdout decoded')
        return original(self,s,idx)
    monkeypatch.setattr(json.JSONDecoder,'raw_decode',guarded)
    assert read_selected_splits(path)=={'train':[],'dev':[]}


def test_safe_official_member_crc_mismatch_rejected():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_STORED) as z:z.writestr('train/VizWiz_train_00000001.jpg',b'actual')
    data=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z:info=z.infolist()[0]
    class Remote:
        def fetch(self,a,b):return data[a:min(b+1,len(data))]
    assert member_bytes(Remote(),info)==b'actual'
    info.CRC+=1
    with pytest.raises(ValueError,match='CRC'):member_bytes(Remote(),info)


def test_confusion_orientation_answerable_positive():
    m=confusion_metrics([0,0,1,1],[0,1,0,1])
    assert (m['TN'],m['FP'],m['FN'],m['TP'])==(1,1,1,1)
    assert m['precision_percent']==m['recall_percent']==m['f1_percent']==50


def test_no_forced_improvement_selection():
    baseline={'vqa_percent':59.15,'answered_subset_risk':.3}
    candidate={'vqa_percent':59.,'answered_subset_risk':.1,'new_abstentions':5}
    chosen,best=select_candidate([candidate],baseline)
    assert chosen is None and best==candidate


def test_selection_requires_better_answered_risk_and_nonworse_vqa():
    baseline={'vqa_percent':59.15,'answered_subset_risk':.3}
    candidate={'vqa_percent':59.15,'answered_subset_risk':.2,'new_abstentions':5}
    assert select_candidate([candidate],baseline)[0]==candidate


def test_training_cost_report_refuses_changed_capture_digest(tmp_path,monkeypatch):
    import evaluate_phase3_dev as evaluator
    folder=tmp_path/'captures';folder.mkdir()
    (folder/'training-verification.json').write_text(json.dumps({'examples':5000,'capture_digest':'changed'}))
    monkeypatch.setattr(evaluator,'checked_splits',lambda:(None,[],None,None))
    monkeypatch.setattr(evaluator,'sha',lambda path:'placeholder-capture-hash')
    with pytest.raises(ValueError,match='capture digest changed'):
        evaluator.training_capture_summary(tmp_path)
