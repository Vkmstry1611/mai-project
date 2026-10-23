"""Portable deployment adapter. Historical evaluation modules stay unchanged.

The manifest attests the reviewed export; it does not redo the private train/dev
audit. SHA checks protect deployment consistency, not the truth of an answer.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Invalid manifest path')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('', '.', '..') or ':' in x for x in name.split('/')):
        raise ValueError('Unsafe manifest path')
    root = Path(root).resolve()
    path = root.joinpath(*p.parts)
    if any(part.is_symlink() for part in [path, *path.parents] if part != root):
        raise ValueError('Symlink runtime asset forbidden')
    if root not in path.resolve().parents:
        raise ValueError('Manifest path escapes checkout')
    return path


def read_manifest(root=ROOT):
    from runtime_trust import MANIFEST_SHA256
    path = Path(root) / 'runtime/manifest.json'
    if sha(path) != MANIFEST_SHA256:
        raise ValueError('Public runtime manifest differs from reviewed trust anchor')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported public runtime schema')
    names = [row['path'] for row in manifest['assets']]
    if len(names) != len(set(n.casefold() for n in names)):
        raise ValueError('Duplicate runtime asset')
    for name in [*names, *manifest['source_sha256']]:
        safe_path(root, name)
    return manifest


def verify(root=ROOT, assets=True):
    root = Path(root).resolve()
    manifest = read_manifest(root)
    for name, expected in manifest['source_sha256'].items():
        if sha(safe_path(root, name)) != expected:
            raise ValueError('Public runtime source differs: ' + name)
    if assets:
        for row in manifest['assets']:
            path = safe_path(root, row['path'])
            if not path.is_file():
                raise FileNotFoundError('Missing runtime asset: ' + row['path'] + '; run setup_public_runtime.ps1')
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Runtime asset differs; preserve and investigate: ' + row['path'])
    return manifest


def guarded_paths(root=ROOT):
    m = read_manifest(root)
    return [safe_path(root, n) for n in [*m['source_sha256'], *(r['path'] for r in m['assets']),
                                        'runtime/manifest.json', 'runtime_trust.py']]


def load_gate(root=ROOT):
    # Never deserialize an arbitrary pickle/joblib, even if a user supplies a hash.
    verify(root)
    from modules.answerability_gate import Gate
    metadata = json.loads((Path(root) / 'runtime/classifier/metadata.json').read_text())
    if metadata['fitted_split'] != 'train':
        raise ValueError('Expected training-only fitted classifier')
    if metadata['sklearn_version'] != importlib.metadata.version('scikit-learn'):
        raise ValueError('Classifier requires scikit-learn ' + metadata['sklearn_version'])
    path = Path(root) / 'runtime/classifier/heads.joblib'
    if sha(path) != metadata['model_sha256']:
        raise ValueError('Classifier hash mismatch')
    import joblib
    gate = Gate.__new__(Gate)
    gate.models = joblib.load(path)
    gate.names = metadata['feature_names']
    gate.metadata = metadata
    if set(gate.models) != {'answerability', 'reliability'}:
        raise ValueError('Unexpected classifier heads')
    return gate


def pipeline(root=ROOT, session_dir=None):
    """Reuse exact Phase 3/4 run and feature methods, replacing only construction."""
    from dataclasses import replace
    from config import Config
    from evaluate_vqa import OfficialScorer
    from phase4_pipeline import Phase4Pipeline
    from pipeline import Pipeline
    root = Path(root).resolve()
    verify(root)
    instance = Phase4Pipeline.__new__(Phase4Pipeline)
    instance.gate = load_gate(root)
    instance.policy = json.loads((root / 'runtime/policy.json').read_text())
    instance.scorer = OfficialScorer(json.loads((root / 'runtime/normalization.json').read_text()))
    session_dir = Path(session_dir or root / 'runtime-local')
    cfg = replace(Config(), vqa_mode='v2', improved_blip_dir=root / 'models/blip_vizwiz_v2',
                  device='cuda', ocr_device='cpu', object_device='cpu', cache_dir=root / 'models/cache',
                  yolo_model=str(root / 'models/cache/yolov8n.pt'),
                  evidence_cache_dir=session_dir / 'application-evidence-cache', ocr_min_score=.9)
    instance.base = Pipeline(cfg)
    instance.config = instance.base.config
    return instance
