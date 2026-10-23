"""Read-only final source/launcher audit. No inference, staging or Git writes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    from investigate_reading_upgrade import OUT
    from audit_submission import PATTERNS
    previous=Path('reports/full_text_reading/run-20261020-v1/PROPOSED_SOURCE_FILES_FINAL.txt')
    candidates=set(previous.read_text(encoding='utf-8').splitlines())
    candidates.update(p.as_posix() for p in Path('.').glob('*.py'))
    candidates.update(p.as_posix() for p in Path('.').glob('*.md'))
    candidates.update(p.as_posix() for p in Path('tests').glob('*.py'))
    candidates=sorted(p for p in candidates if p and Path(p).is_file())
    findings=[];large=[];hashes={}
    for name in candidates:
        p=Path(name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stat().st_size>2*1024**2:large.append(name)
        try:text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:continue
        if p.suffix=='.py':ast.parse(text,filename=name)
        for rule,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if rule=='literal_credential' and any(s in match.group(1).lower() for s in ('placeholder','example','your_','changeme','getenv')):continue
                findings.append({'path':name,'rule':rule,'line':text.count('\n',0,match.start())+1})
    before=json.loads((OUT/'SOURCE_BEFORE.json').read_text())
    changed=[p for p,v in before.items() if Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v]
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
    whitespace=subprocess.run(['git','diff','--check'],capture_output=True,text=True)
    excluded={p:subprocess.run(['git','check-ignore','-q',p]).returncode==0 for p in ('.venv/','.venv-gpu/','.venv-ocr/','.deps/','reports/','models/blip_vizwiz_v2/')}
    result=dict(candidate_count=len(candidates),potential_secret_findings=findings,files_over_2mib=large,staged_files=staged,
        changed_preexisting_python=changed,git_diff_check_returncode=whitespace.returncode,git_diff_check=whitespace.stdout,
        runtime_paths_ignored=excluded,limits='Pattern scan is not a security guarantee; proposed source list is not approval to commit. Private photos and model/runtime artifacts excluded.')
    for name,data in [('SOURCE_AFTER.json',hashes),('GIT_AUDIT.json',result)]:
        with (OUT/name).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    with (OUT/'PROPOSED_SOURCE_FILES.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(candidates)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
