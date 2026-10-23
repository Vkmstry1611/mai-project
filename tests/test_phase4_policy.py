import json
from pathlib import Path
import pytest
from phase4_policy import admissible, pareto, run_path, RULE
from modules.answerability_gate import decide


def metric(vqa=59.15, recall=57.58, precision=90.72, answered=431, ocr=29):
    return dict(vqa_percent=vqa,recall_percent=recall,precision_percent=precision,answered_examples=answered,
                subgroups={'ocr_supported':{'answered_examples':ocr}})


def test_coverage_safeguards_reject_high_vqa_excess_abstention():
    baseline=metric()
    assert not admissible(metric(vqa=60.7,recall=43.74,precision=93.69,answered=317,ocr=5),baseline)
    assert admissible(metric(vqa=59.31,recall=57.44,precision=90.91,answered=429,ocr=27),baseline)


@pytest.mark.parametrize('change',[{'vqa':59},{'recall':50},{'precision':89},{'answered':380},{'ocr':25}])
def test_each_safeguard_is_required(change):
    assert not admissible(metric(**change),metric())


def test_pareto_tradeoff_retained_and_dominated_rejected():
    a=metric();b=metric(vqa=60,recall=55,precision=93,answered=410)
    dominated=metric(vqa=58,recall=54,precision=89,answered=400)
    assert pareto([a,b,dominated])==[a,b]


def test_zero_answerability_threshold_is_explicit_not_recovery():
    assert decide('milk',.01,.05,0,.05)['answer']=='milk'
    assert decide('milk',.99,.049,0,.05)['abstained']
    assert decide('unanswerable',1,1,0,.05)['abstained']
    assert decide('milk',None,.9,0,.05)['abstained']


def test_holdout_requires_frozen_verified_policy(monkeypatch,tmp_path):
    import phase4_holdout
    monkeypatch.setattr(phase4_holdout,'verify_policy',lambda p:(_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(phase4_holdout,'checked_splits',lambda:pytest.fail('must not read data before freeze'))
    with pytest.raises(ValueError,match='not frozen'):phase4_holdout.holdout_rows(tmp_path)


def test_output_cannot_target_previous_artifacts():
    with pytest.raises(ValueError):run_path('reports/phase3_answerability/run-local-20261012-v1')


def test_phase4_wrapper_preserves_phase2_and_final_abstention(monkeypatch):
    from phase4_pipeline import Phase4Pipeline
    from phase3_pipeline import Phase3Pipeline
    debug={'baseline_answer':'water','phase2_final_answer':'MILK','phase2_supporting_evidence_ids':['ocr-1'],
        'final_answer':'unanswerable','decision':'phase3_abstained','phase3_gate_reason':'low probability'}
    monkeypatch.setattr(Phase3Pipeline,'run',lambda self,image,q:('Please try a clearer photo',None,'OK',dict(debug),None))
    wrapper=Phase4Pipeline.__new__(Phase4Pipeline);wrapper.policy={'threshold_answerable':0,'threshold_reliable':.05}
    shown,_,_,details,_=wrapper.run(None,'read this')
    assert details['baseline_answer']=='water' and details['phase2_final_answer']=='MILK'
    assert details['phase2_supporting_evidence_ids']==['ocr-1']
    assert details['decision']=='phase4_abstained' and 'clearer photo' in shown
    assert details['phase4_gate_reason']=='low probability'
