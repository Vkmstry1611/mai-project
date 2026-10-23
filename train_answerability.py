"""Fit tiny Phase 3 classifiers and calibration on original training only."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

from capture_phase3_features import checked_entry
from evaluate_vqa import OfficialScorer
from phase3_common import checked_splits, digest, ensure_new_or_equal, ROOT, safe_run, sha, verify_freeze, write_new


def assert_fit_ids(ids, original_ids, *, allow_subset=False):
    if len(ids)!=len(set(ids)) or not set(ids)<=set(original_ids) or (not allow_subset and list(ids)!=list(original_ids)):
        raise ValueError('Classifier fit must use original training IDs only, with exact order for full fit')


def fit_head(x,y):
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    counts=Counter(y.tolist())
    if len(counts)!=2 or min(counts.values())<5:
        return None, {'unavailable':'Too few examples in both training target classes','class_counts':dict(counts)}
    pipe=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=3000,random_state=42))])
    cv=StratifiedKFold(3,shuffle=True,random_state=42)
    grid=GridSearchCV(pipe,{'classifier__C':[.1,1.,10.]},cv=cv,scoring='neg_log_loss',n_jobs=1,error_score='raise')
    start=time.perf_counter(); grid.fit(x,y)
    # Both estimator parameters and sigmoid calibrator use training folds only.
    calibrated=CalibratedClassifierCV(grid.best_estimator_,method='sigmoid',cv=StratifiedKFold(5,shuffle=True,random_state=43),ensemble=False,n_jobs=1)
    calibrated.fit(x,y)
    info={'class_counts':dict(counts),'C':float(grid.best_params_['classifier__C']),
          'training_cv_negative_log_loss':float(grid.best_score_),'fit_seconds':time.perf_counter()-start,
          'calibration':'sigmoid on 5-fold out-of-fold original-training predictions; final estimator refit on same training rows',
          'cv_grid':[{'C':float(c),'negative_log_loss':float(s)} for c,s in zip(grid.cv_results_['param_classifier__C'],grid.cv_results_['mean_test_score'])],
          'coefficients':grid.best_estimator_.named_steps['classifier'].coef_[0].tolist()}
    return calibrated,info


def train(run_dir):
    import joblib
    import numpy as np
    root=safe_run(run_dir); verify_freeze()
    _,rows,_,payload=checked_splits()
    if not (root/'captures/training-verification.json').is_file(): raise RuntimeError('All 5,000 training captures required before fitting')
    folder=root/'models'
    if (folder/'metadata.json').exists():
        from modules.answerability_gate import Gate
        Gate(folder)
        print('Verified existing fitted classifiers; fitting reused',flush=True); return
    if folder.exists() and any(folder.iterdir()): raise FileExistsError('Partial model artifacts preserved; select a new versioned model directory')
    attempt=root/'model-fit-attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True,exist_ok=False)
    scorer=OfficialScorer(payload['official_rules']); records=[]
    for i,row in enumerate(rows):
        info=json.loads((root/'data/image-integrity'/(row['image']+'.json')).read_text())
        records.append(checked_entry(root/'captures/train'/f'{i:04d}.json',row,info['sha256']))
    assert_fit_ids([r['image'] for r in records],[r['image'] for r in rows])
    names=sorted(records[0]['features'])
    if any(sorted(r['features'])!=names for r in records): raise ValueError('Training feature schema changed')
    x=np.asarray([[r['features'][n] for n in names] for r in records],dtype=float)
    y=np.asarray([r['answerable'] for r in rows],dtype=int)
    if not np.isfinite(x).all(): raise ValueError('Invalid feature value')
    active=np.asarray([bool(r['final_answer'].strip()) and scorer.normalize(r['final_answer'])!='unanswerable' for r in records])
    # Ground truth used as targets only. It is absent from saved inference features.
    full_credit=np.asarray([int(row['answerable']==1 and scorer.score(r['final_answer'],row['answers'])>=1-1e-10) for row,r in zip(rows,records)])
    assert_fit_ids([r['image'] for r,use in zip(records,active) if use],[r['image'] for r in rows],allow_subset=True)
    print('Fitting original-training answerability head: 5000 examples',flush=True)
    answerability,a_info=fit_head(x,y)
    if answerability is None: raise RuntimeError('Answerability target classes unavailable')
    print(f'Fitting candidate-reliability head: {int(active.sum())} non-abstaining original-training examples',flush=True)
    reliability,r_info=fit_head(x[active],full_credit[active])
    joblib.dump({'answerability':answerability,'reliability':reliability},attempt/'heads.joblib',compress=3)
    metadata={'schema':1,'fitted_split':'train','train_examples':5000,'train_rows_sha256':digest(rows),
              'source_manifest_sha256':payload['source_manifest_sha256'],'fitted_ids_sha256':digest([r['image'] for r in rows]),
              'feature_names':names,'sklearn_version':importlib.metadata.version('scikit-learn'),
              'model_sha256':sha(attempt/'heads.joblib'),'answerability':a_info,'reliability':r_info,
              'reliability_examples':int(active.sum()),'reliability_target':'answerable label 1 AND full official VQA credit, restricted to non-abstaining original-training candidates',
              'head_probabilities_are_not_proof_of_visual_support':True,'holdout_accessed':False,
              'source_sha256':{p:sha(ROOT/p) for p in ('train_answerability.py','modules/answerability_features.py','modules/answerability_gate.py')}}
    write_new(attempt/'metadata.json',metadata)
    ensure_new_or_equal(root/'training-features.json',{'split':'train','rows_sha256':digest(rows),'feature_names':names,
                'rows':[{'image':r['image'],'features':r['features']} for r in records]})
    if root not in attempt.resolve().parents or root not in folder.resolve().parents or folder.exists():
        raise ValueError('Model publication target changed; partial attempt preserved')
    attempt.rename(folder)
    print('TRAINING_CLASSIFIERS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();train(args.run_dir)
