"""Deterministic private runtime bundle; no dataset images or model execution."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT=Path(__file__).resolve().parent
P4='reports/phase4_policy/run-local-20261013-v1'
P3='reports/phase3_answerability/run-local-20261012-v1'
P2LOCK='reports/phase3_answerability/plan-20261012-023211-3dbd1268/PHASE2_FREEZE.json'
CACHE='reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/model-cache'
ALLOWED_FILES={
    *('models/blip_vizwiz_v2/'+n for n in ['config.json','generation_config.json','model.safetensors','preprocessor_config.json','special_tokens_map.json','tokenizer_config.json','vocab.txt']),
    CACHE+'/yolov8n.pt',CACHE+'/easyocr/craft_mlt_25k.pth',CACHE+'/easyocr/english_g2.pth',
    'reports/phase2_local_dev_evaluation/drive_inputs/split_manifest.json',
    'reports/phase2_local_dev_evaluation/coverage-targeted-v1-20261012-021213/results/expanded_records.json',
    'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/inputs/run02-dev-only.json',
    P2LOCK,*(P3+'/'+n for n in ['models/heads.joblib','models/metadata.json','selected-policy.json']),
    *(P4+'/'+n for n in ['selected-policy.json','development_summary.json','POLICY_FREEZE.json','APPLICATION_FREEZE.json'])}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def relative(value):
    value=str(value).replace('\\','/')
    p=PurePosixPath(value)
    if p.is_absolute() or any(v in ('','.','..') or ':' in v for v in value.split('/')):
        raise ValueError('Unsafe relative path')
    return p.as_posix()


def target(root,value):
    root=Path(root).resolve();path=(root/relative(value)).resolve()
    if root not in path.parents:raise ValueError('Path escapes project')
    return path


def runtime_files():
    from phase3_common import MANIFEST,DEV_INPUT,PHASE2_RECORDS
    lock=json.loads((ROOT/P2LOCK).read_text())
    required={P2LOCK,str(MANIFEST.relative_to(ROOT)),str(DEV_INPUT.relative_to(ROOT)),str(PHASE2_RECORDS.relative_to(ROOT)),CACHE+'/yolov8n.pt'}
    for name in lock['protected_artifact_sha256']:
        path=Path(name)
        if name.startswith('models\\blip_vizwiz_v2') or ('model-cache' in path.parts and path.suffix in ('.pt','.pth')):
            required.add(name)
    required.update(P3+'/'+n for n in ['models/heads.joblib','models/metadata.json','selected-policy.json'])
    required.update(P4+'/'+n for n in ['selected-policy.json','development_summary.json','POLICY_FREEZE.json','APPLICATION_FREEZE.json'])
    result={relative(n) for n in required}
    if result!=ALLOWED_FILES:raise ValueError('Runtime dependency closure differs from reviewed paths')
    return sorted(result)


def source_pins():
    names=set(json.loads((ROOT/P2LOCK).read_text())['frozen_source_sha256'])
    for name in ('POLICY_FREEZE.json','APPLICATION_FREEZE.json'):
        names.update(json.loads((ROOT/P4/name).read_text())['source_sha256'])
    names.update(json.loads((ROOT/P3/'models/metadata.json').read_text())['source_sha256'])
    names.update(['phase3_common.py','phase3_pipeline.py','evaluate_phase3_dev.py','launch_submission.py','submission_ui_helpers.py'])
    names.update(str(p.relative_to(ROOT)) for folder in ['modules','utils'] for p in (ROOT/folder).glob('*.py'))
    return {relative(n):sha(ROOT/n) for n in sorted(names)}


def info(name):
    entry=zipfile.ZipInfo(name,date_time=(2026,10,16,0,0,0))
    entry.compress_type=zipfile.ZIP_STORED;entry.external_attr=(stat.S_IFREG|0o644)<<16
    return entry


def build(output):
    out=Path(output).resolve()
    if ROOT/'reports/artifact_bundles' not in out.parents:raise ValueError('Build inside reports/artifact_bundles')
    out.mkdir(parents=True,exist_ok=False)
    from phase4_gpu_infer import verify_application
    from phase3_common import verify_freeze
    verify_application(P4);verify_freeze()
    files=[{'path':n,'bytes':target(ROOT,n).stat().st_size,'sha256':sha(target(ROOT,n))} for n in runtime_files()]
    manifest={'schema':1,'release':'VisionBridge-frozen-runtime-20261016-v1','target':'Windows Python 3.11',
              'files':files,'source_sha256':source_pins(),'policy_sha256':sha(ROOT/P4/'selected-policy.json'),
              'sklearn_version':'1.9.1','contains_dataset_images':False,'contains_holdout_predictions':False,
              'contains_original_split_manifest_and_development_annotations':True,
              'original_blip_checkpoint_included':False,'optional_whisper_weights_included':False,
              'privacy':'Private runtime bundle; do not upload to Git/public storage without review and approval.'}
    body=json.dumps(manifest,sort_keys=True,indent=2).encode('utf-8')
    archive=out/'VisionBridge-runtime.zip'
    with zipfile.ZipFile(archive,'x',allowZip64=True) as bundle:
        bundle.writestr(info('artifact-manifest.json'),body)
        for row in files:
            h=hashlib.sha256()
            with target(ROOT,row['path']).open('rb') as source,bundle.open(info('payload/'+row['path']),'w',force_zip64=True) as dest:
                for block in iter(lambda:source.read(1024**2),b''):
                    h.update(block);dest.write(block)
            if h.hexdigest()!=row['sha256']:raise ValueError('Source changed during packaging')
    (out/'artifact-manifest.json').write_bytes(body)
    (out/'SHA256SUMS.txt').write_text(sha(archive)+'  '+archive.name+'\n'+sha(out/'artifact-manifest.json')+'  artifact-manifest.json\n',encoding='utf-8')
    verify(archive,ROOT,install=False)
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2),flush=True)


def verify(archive,project,install=False,expected_sha256=None):
    archive=Path(archive);project=Path(project).resolve()
    if expected_sha256 and sha(archive)!=expected_sha256.lower():raise ValueError('ZIP SHA256 mismatch')
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate archive paths')
        manifest=json.loads(bundle.read('artifact-manifest.json'))
        if manifest.get('schema')!=1:raise ValueError('Unsupported manifest schema')
        rows=manifest['files'];paths=[relative(r['path']) for r in rows]
        if len(paths)!=len(set(p.casefold() for p in paths)):raise ValueError('Duplicate manifest paths')
        if set(paths)!=ALLOWED_FILES:raise ValueError('Runtime bundle must contain exactly the 21 reviewed dependencies, including YOLO')
        if set(names)!={'artifact-manifest.json',*('payload/'+p for p in paths)}:raise ValueError('Unexpected ZIP members')
        for name,expected in manifest['source_sha256'].items():
            path=target(project,name)
            if not path.is_file() or sha(path)!=expected:raise ValueError('Required source differs: '+name)
        # Validate every archive byte and every existing destination BEFORE writing.
        for row in rows:
            name='payload/'+relative(row['path']);entry=bundle.getinfo(name)
            if stat.S_ISLNK(entry.external_attr>>16) or entry.file_size!=row['bytes']:raise ValueError('Invalid member type/size')
            h=hashlib.sha256()
            with bundle.open(entry) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            if h.hexdigest()!=row['sha256']:raise ValueError('Member hash mismatch')
            dest=target(project,row['path'])
            if dest.exists() and (not dest.is_file() or sha(dest)!=row['sha256']):raise FileExistsError('Refusing different destination: '+str(dest))
        if install:
            for row in rows:
                dest=target(project,row['path'])
                if dest.exists():continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open('payload/'+relative(row['path'])) as source,dest.open('xb') as output:
                    shutil.copyfileobj(source,output,1024**2)
        print('BUNDLE_VERIFIED',len(rows),'members','install='+str(install),flush=True)
        return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--output',required=True)
    for name in ['verify','install']:
        v=sub.add_parser(name);v.add_argument('--archive',required=True);v.add_argument('--project',default=str(ROOT));v.add_argument('--expected-sha256')
    a=p.parse_args()
    if a.command=='build':build(a.output)
    else:
        if a.command=='install' and not a.expected_sha256:raise ValueError('Install requires the ZIP SHA256 from a trusted source')
        verify(a.archive,a.project,a.command=='install',a.expected_sha256)
