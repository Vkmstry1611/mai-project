import json
import importlib.metadata
import pytest
from modules.answerability_gate import Gate,decide
from train_answerability import assert_fit_ids


def test_existing_abstention_never_recovers_answer():
    assert decide('unanswerable',1.,1.,.5,.5)['answer']=='unanswerable'


def test_both_heads_required_and_boundary_inclusive():
    assert decide('milk',.9,.9,.9,.9)['answer']=='milk'
    assert decide('milk',.899,.99,.9,.9)['abstained']
    assert decide('milk',.99,.899,.9,.9)['abstained']
    assert decide('milk',.99,None,.9,.9)['abstained']


def test_disabled_gate_retains_candidate():
    assert decide('milk',None,None)['answer']=='milk'


def test_invalid_probability_threshold_rejected():
    with pytest.raises(ValueError):decide('milk',.5,.5,1.01,None)


def test_invalid_scores_and_empty_candidate_abstain():
    assert decide('milk',float('nan'),.9,.5,.5)['abstained']
    assert decide('milk',.9,float('inf'),.5,.5)['abstained']
    assert decide('',1.,1.)['abstained']


def test_fit_guard_rejects_dev_and_reordered_full_train():
    with pytest.raises(ValueError):assert_fit_ids(['train-1','dev-1'],['train-1','train-2'])
    with pytest.raises(ValueError):assert_fit_ids(['train-2','train-1'],['train-1','train-2'])
    assert_fit_ids(['train-2'],['train-1','train-2'],allow_subset=True)


def test_gate_rejects_modified_model_before_joblib_load(tmp_path,monkeypatch):
    import phase3_common
    rows=[{'image':'train-1'}]
    metadata={'fitted_split':'train','sklearn_version':importlib.metadata.version('scikit-learn'),
              'train_rows_sha256':phase3_common.digest(rows),'model_sha256':'expected'}
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(phase3_common,'checked_splits',lambda:(None,rows,None,None))
    monkeypatch.setattr(phase3_common,'verify_freeze',lambda:None)
    monkeypatch.setattr(phase3_common,'sha',lambda path:'modified')
    with pytest.raises(ValueError,match='hash mismatch'):Gate(tmp_path)


def test_fitting_and_sigmoid_calibration_use_given_training_features_only():
    import numpy as np
    from train_answerability import fit_head
    x=np.asarray([[i,i%3] for i in range(40)],dtype=float)
    y=np.asarray([i%2 for i in range(40)])
    model,info=fit_head(x,y)
    final_estimator=model.calibrated_classifiers_[0].estimator
    assert np.allclose(final_estimator.named_steps['scale'].mean_,x.mean(axis=0))
    assert model.predict_proba(x[:2]).shape==(2,2)
    assert info['C'] in (.1,1.,10.)
