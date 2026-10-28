"""Read-only reviewed source, dependency-byte and Git readiness checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
from urllib.parse import urlsplit

def git(*args):return subprocess.check_output(['git',*args],text=True,encoding='utf-8').strip()

def main():
    from visionbridge.audit_submission import PATTERNS
    from artifact_bundle import source_pins
    out=Path('reports/assistant_followup')/('audit-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    old=Path('reports/reading_upgrade/run-20261020-v1/final-source-audit/PROPOSED_SOURCE_FILES.txt')
    files=set(old.read_text().splitlines()) if old.exists() else set()
    for folder,patterns in [(Path('.'),('*.py','*.md','requirements*.txt','*.ps1','*.ipynb')),(Path('scripts'),('**/*.py',)),(Path('tests'),('*.py',)),(Path('assets/workspace'),('*.css','*.svg'))]:
        for pattern in patterns:files.update(p.as_posix() for p in folder.glob(pattern))
    files.update(['.gitignore','.gitattributes']);files=sorted(n for n in files if Path(n).is_file())
    findings=[];large=[];hashes={};notebooks=[];deferred=[]
    for name in files:
        path=Path(name);raw=path.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        if len(raw)>2*1024**2:large.append(name)
        try:text=raw.decode('utf-8-sig')
        except UnicodeError:continue
        if path.suffix=='.py':ast.parse(text,filename=name)
        if path.suffix=='.ipynb':
            notebook=json.loads(text)
            outputs=[o for c in notebook.get('cells',[]) for o in c.get('outputs',[])]
            images=sum(any(k.startswith('image/') for k in o.get('data',{})) for o in outputs)
            notebooks.append({'path':name,'output_count':len(outputs),'image_output_count':images})
            if images:deferred.append({'path':name,'reason':'Executed notebook with embedded image outputs; preserve original and review provenance or a sanitized copy before staging.'})
            for n,cell in enumerate(notebook.get('cells',[])):
                if cell.get('cell_type')=='code':
                    source=''.join(cell.get('source',[]))
                    # Existing IPython magics are valid notebook cells, not plain Python.
                    if not any(line.lstrip().startswith(('%','!')) for line in source.splitlines()):ast.parse(source,filename=f'{name}:cell{n}')
        for rule,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if rule=='literal_credential' and any(x in match.group(1).lower() for x in ('placeholder','example','your_','changeme','getenv')):continue
                findings.append({'path':name,'rule':rule,'line':text.count('\n',0,match.start())+1})
    pinned=source_pins();pinned.update(json.loads(Path('reports/persistent_demo/validation-20261019-v1/SOURCE_SHA256.json').read_text()))
    eol=[]
    for name in pinned:
        if git('hash-object','--path='+name,name)!=git('hash-object','--no-filters',name):eol.append(name)
    tracked=git('ls-files').splitlines();existing_large=[n for n in tracked if Path(n).is_file() and Path(n).stat().st_size>2*1024**2]
    remote_risks=[]
    for name in git('remote').splitlines():
        value=git('remote','get-url',name);parsed=urlsplit(value)
        if parsed.password or any(PATTERNS[k].search(value) for k in ('github_token','oauth_token')):remote_risks.append({'remote':name,'embedded_credential':True})
    proposal=[n for n in files if n not in {d['path'] for d in deferred}]
    ignore={n:subprocess.run(['git','check-ignore','-q',n]).returncode==0 for n in ['.venv/','.venv-gpu/','.venv-ocr/','.deps/','reports/','private-artifacts/','models/blip_vizwiz_v2/']}
    whitespace=subprocess.run(['git','diff','--check'],capture_output=True,text=True)
    status={'branch':git('branch','--show-current'),'candidate_count':len(proposal),'scanned_source_files':len(files),'potential_secret_findings':findings,'candidate_files_over_2mib':large,
        'staged_files':git('diff','--cached','--name-only').splitlines(),'ignored_local_paths':ignore,'pinned_source_git_filter_byte_changes':eol,
        'tracked_large_files_for_existing_lfs_review':existing_large,'notebooks':notebooks,'deferred_notebook_review':deferred,'remote_credential_risks':remote_risks,
        'whitespace_exit':whitespace.returncode,'whitespace_details':whitespace.stdout,
        'limits':'Pattern scan is not a secrecy guarantee. Historical tracked model/sample assets remain for user review; no removals or Git mutations.'}
    with (out/'AUDIT.json').open('x') as f:json.dump(status,f,indent=2)
    with (out/'SOURCE_SHA256.json').open('x') as f:json.dump(hashes,f,indent=2)
    with (out/'PROPOSED_SOURCE_FILES.txt').open('x') as f:f.write('\n'.join(proposal)+'\n')
    print('AUDIT_FOLDER',out);print(json.dumps(status,indent=2))

if __name__=='__main__':main()
