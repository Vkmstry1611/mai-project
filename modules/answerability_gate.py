"""Verified optional post-Phase-2 gate. Scores estimate labels, not grounding."""
import json
import math
from pathlib import Path


def decide(candidate, p_answerable, p_reliable=None, threshold_answerable=None, threshold_reliable=None):
    for threshold in (threshold_answerable,threshold_reliable):
        if threshold is not None and not 0<=threshold<=1:
            raise ValueError('Gate thresholds must lie in [0,1]')
    if not candidate.strip() or candidate.strip().lower()=='unanswerable':
        return {'answer':'unanswerable','abstained':True,'reason':'Phase 2 already abstained; no answer recovery attempted.'}
    if threshold_answerable is not None and (p_answerable is None or not math.isfinite(p_answerable) or not 0<=p_answerable<=1 or p_answerable<threshold_answerable):
        return {'answer':'unanswerable','abstained':True,'reason':'Estimated answerability below selected development threshold.'}
    if threshold_reliable is not None and (p_reliable is None or not math.isfinite(p_reliable) or not 0<=p_reliable<=1 or p_reliable<threshold_reliable):
        return {'answer':'unanswerable','abstained':True,'reason':'Estimated full-reference agreement below selected development threshold.'}
    return {'answer':candidate,'abstained':False,'reason':'Candidate retained by selected gate policy.'}


class Gate:
    """Load only a locally generated artifact after explicit SHA/version checks."""
    def __init__(self, folder):
        from phase3_common import sha, checked_splits, digest, verify_freeze
        import importlib.metadata
        import joblib
        folder=Path(folder)
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['fitted_split']!='train' or metadata['sklearn_version']!=importlib.metadata.version('scikit-learn'):
            raise ValueError('Gate fitted split or scikit-learn version mismatch')
        _,train,_,_=checked_splits()
        if metadata['train_rows_sha256']!=digest(train): raise ValueError('Gate trained against different manifest')
        verify_freeze()
        if sha(folder/'heads.joblib')!=metadata['model_sha256']: raise ValueError('Gate artifact hash mismatch')
        from phase3_common import ROOT
        for name,expected in metadata['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Phase 3 feature/gate/fitting source changed; use the recorded version')
        # joblib is not a safe general untrusted-file format. Only our freshly
        # generated local artifact with checked metadata/hash is supported.
        self.models=joblib.load(folder/'heads.joblib')
        self.names=metadata['feature_names']; self.metadata=metadata

    def predict(self, features, candidate):
        import numpy as np
        if set(features)!=set(self.names): raise ValueError('Inference feature schema mismatch')
        matrix=np.asarray([[features[n] for n in self.names]],dtype=float)
        if not np.isfinite(matrix).all(): raise ValueError('Non-finite gate input')
        p_answerable=float(self.models['answerability'].predict_proba(matrix)[0,1])
        reliable=self.models.get('reliability')
        p_reliable=float(reliable.predict_proba(matrix)[0,1]) if reliable is not None and candidate.strip().lower()!='unanswerable' else None
        return p_answerable,p_reliable
