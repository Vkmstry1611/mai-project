"""Read-only Git/source inventory; writes preparation manifests, never stages."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from visionbridge.audit_submission import PATTERNS

ROOT=Path(__file__).resolve().parents[2]
PUBLIC_DOCS={'README.md','docs/USER_MANUAL.md','docs/research/ARTIFACT_BUNDLE_GUIDE.md',
 'docs/research/MANUAL_TEST_CHECKLIST.md','docs/research/BASELINE_DISCREPANCY_UPDATE.md','docs/research/TRAINING_GUIDE.md','docs/research/COLAB_T4_GUIDE.md','docs/research/DATASET_FIX_GUIDE.md',
 'docs/research/PHASE2_DEV_EVALUATION.md','docs/research/PHASE2_LOCAL_WINDOWS.md','docs/research/PHASE3_GUIDE.md','docs/research/PHASE4_GUIDE.md',
 'docs/research/SAMPLE_PROVENANCE.md'}
PUBLIC_SUBMISSION={'submission/ARTIFACT_INDEX-20261016.json','submission/PLOT_PROVENANCE.json',
 'submission/RESULTS_AND_PROVENANCE.md','submission/notebooks/PROVENANCE.json',
 'submission/notebooks/mai_model_train_source.ipynb'}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8')


def audit(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    tracked=set(git('ls-files','-z').split('\0'))-{''}
    visible=set(git('ls-files','--cached','--others','--exclude-standard','-z').split('\0'))-{''}
    rows=[];findings=[];private_paths=[];notebooks=[];syntax=[]
    for name in sorted(visible):
        p=ROOT/name
        if not p.is_file():
            rows.append({'path':name,'classification':'Review','reason':'Tracked file missing locally','bytes':0});continue
        category='Review';reason='Historical/local documentation or sample; explicit publication review'
        if p.suffix=='.py' or p.suffix=='.ps1' or name.startswith(('modules/','utils/','scripts/','tests/','assets/')):
            category='Include';reason='Application, test or reproducibility/development source'
        if name in PUBLIC_DOCS or name in ('.gitignore','.gitattributes') or name.startswith('requirements'):
            category='Include';reason='Reviewed setup, dependency or public documentation source'
        if name in PUBLIC_SUBMISSION or name.startswith('submission/plots/'):
            category='Include';reason='Curated public submission material; not per-example data'
        if name in ('models/README.md','sample/README.md','sample/questions.json','outputs/.gitkeep'):
            category='Include';reason='Small source-controlled documentation/example configuration'
        if p.suffix=='.ipynb':
            nb=json.loads(p.read_text(encoding='utf-8'));outputs=[o for c in nb.get('cells',[]) for o in c.get('outputs',[])]
            notebooks.append({'path':name,'outputs':len(outputs),'embedded_image_outputs':sum(any(k.startswith('image/') for k in o.get('data',{})) for o in outputs)})
            category='Review' if outputs or name in ('mai_model_train.ipynb','notebooks/VisionBridge_VizWiz_Baseline_VQA (2).ipynb') else 'Include'
            reason='Preserve original/executed notebook; review outputs before publishing' if category=='Review' else 'Unexecuted improved workflow notebook'
        if name.startswith('models/blip_vizwiz/'):
            category='Review';reason='Previously tracked original checkpoint/config; unchanged, not new PR payload'
        size=p.stat().st_size
        textual=p.suffix.lower() in ('.py','.ps1','.md','.json','.txt','.ipynb','.toml','.yaml','.yml','.css','.svg','.csv') or name in ('.gitignore','.gitattributes')
        raw=p.read_bytes() if textual else None
        try:text=raw.decode('utf-8-sig') if raw is not None else None
        except UnicodeError:text=None
        if text:
            if p.suffix=='.py':ast.parse(text,filename=name);syntax.append(name)
            for rule,pattern in PATTERNS.items():
                for m in pattern.finditer(text):
                    if rule=='literal_credential' and any(s in m.group(1).lower() for s in ('placeholder','example','your_','changeme','getenv')):continue
                    findings.append({'path':name,'line':text.count('\n',0,m.start())+1,'rule':rule})
            if 'C:\\Users\\' in text or 'C:/Users/' in text:private_paths.append(name)
        # Checkpoint content is not needed for a Git-visible text/privacy scan.
        rows.append({'path':name,'classification':category,'reason':reason,'bytes':size,'tracked':name in tracked,'sha256':hashlib.sha256(raw).hexdigest() if raw is not None else ''})
    excluded=[];large=[];inventory={}
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d!='.git' and not d.startswith('.venv') and d not in ('__pycache__','.deps','.pytest_cache')]
        for name in files:
            p=Path(directory)/name;rel=p.relative_to(ROOT).as_posix()
            try:size=p.stat().st_size
            except OSError:continue
            top=rel.split('/')[0];entry=inventory.setdefault(top,{'files':0,'bytes':0});entry['files']+=1;entry['bytes']+=size
            if size>=10*1024**2:large.append({'path':rel,'bytes':size,'tracked':rel in tracked})
    for name in ['.venv/','.venv-gpu/','.venv-ocr/','.deps/','reports/','private-artifacts/','models/blip_vizwiz_v2/','example.wav','example.zip','.env','.env.example']:
        ignored=subprocess.run(['git','check-ignore','--no-index','-q',name],cwd=ROOT).returncode==0
        excluded.append({'path':name,'ignored':ignored,'classification':'Exclude' if ignored else 'Review','reason':'Local/private artifacts; exception for example configuration'})
    with (out/'FILE_CLASSIFICATION.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','classification','reason','bytes','tracked','sha256']);writer.writeheader();writer.writerows(rows)
    for category in ('Include','Review'):
        (out/(category.upper()+'.txt')).write_text('\n'.join(r['path'] for r in rows if r['classification']==category)+'\n',encoding='utf-8')
    result={'branch':git('branch','--show-current').strip(),'head':git('rev-parse','HEAD').strip(),
       'tracked_count':len(tracked),'status':git('status','--porcelain=v1'),'staged':git('diff','--cached','--name-only').splitlines(),
       'tracked_ignored':git('ls-files','-ci','--exclude-standard').splitlines(),
       'include_count':sum(r['classification']=='Include' for r in rows),'review_count':sum(r['classification']=='Review' for r in rows),
       'secret_pattern_findings':findings,'source_files_with_absolute_user_paths':private_paths,
       'large_files':large,'notebooks':notebooks,'ignore_checks':excluded,'inventory':inventory,'python_syntax_checked':len(syntax),
       'scope':'Text patterns on Git-visible files only; binary privacy, licenses and full Git history are not certified. No holdout data contents scanned.'}
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('tracked_count','include_count','review_count','secret_pattern_findings','staged','python_syntax_checked')},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);audit(parser.parse_args().output)
