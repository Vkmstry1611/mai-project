"""Read-only size, Git and high-signal secret audit; never prints secret values."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}'),
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
    'oauth_token': re.compile(r'\bya29\.[A-Za-z0-9_-]{30,}'),
    'google_api_key': re.compile(r'\bAIza[A-Za-z0-9_-]{35}\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'literal_credential': re.compile(r'''(?im)["']?(?:api_key|access_token|client_secret|password)["']?\s*[:=]\s*["']([^"'\s]{12,})["']'''),
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8')


def audit(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tracked = set(git('ls-files', '-z').split('\0')) - {''}
    sizes = collections.Counter()
    counts = collections.Counter()
    large = []
    scan = set()
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', '.venv-gpu', '.deps'}]
        for name in files:
            path = Path(directory) / name
            rel = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            sizes[rel.split('/')[0]] += size
            counts[rel.split('/')[0]] += 1
            if size >= 10 * 1024**2:
                large.append({'path': rel, 'mib': round(size/1024**2, 2), 'tracked': rel in tracked})
            if path.suffix.lower() in {'.py', '.ps1', '.md', '.txt', '.ipynb', '.json', '.toml', '.yaml', '.yml'}:
                if rel in tracked or rel.startswith('submission/') or (rel.split('/')[0] in {'modules', 'utils', 'tests'} or '/' not in rel) or (rel.startswith('reports/') and path.suffix == '.md'):
                    scan.add(path)
    findings = []
    notebooks = []
    for path in sorted(scan):
        try:
            text = path.read_text(encoding='utf-8-sig')
        except (UnicodeError, OSError):
            continue
        for rule, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if rule == 'literal_credential' and any(s in match.group(1).lower() for s in ('placeholder', 'example', 'your_', 'changeme', 'getenv')):
                    continue
                findings.append({'path': path.relative_to(ROOT).as_posix(), 'line': text.count('\n', 0, match.start())+1,
                                 'rule': rule, 'value': '[REDACTED] — manual review required'})
        if path.suffix == '.ipynb':
            try:
                nb = json.loads(text)
                notebooks.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                                  'cells_with_outputs': sum(bool(c.get('outputs')) for c in nb.get('cells', []))})
            except json.JSONDecodeError:
                notebooks.append({'path': str(path), 'invalid_json': True})
    from urllib.parse import urlsplit
    remote_risks=[]
    for name in git('remote').splitlines():
        value=git('remote','get-url',name).strip()
        parsed=urlsplit(value)
        if parsed.password or any(PATTERNS[k].search(value) for k in ('github_token','oauth_token')):
            remote_risks.append({'remote':name,'embedded_credential':True,'url':'[REDACTED]'})
    result = {'branch': git('branch', '--show-current').strip(), 'status': git('status', '--short'),
              'staged_diff': git('diff', '--cached', '--stat'), 'tracked_files': len(tracked),
              'environment_contents_excluded_from_scan': ['.venv', '.venv-gpu', '.deps'],
              'top_level_inventory': [{'path': k, 'files': counts[k], 'mib': round(v/1024**2, 2)} for k,v in sizes.most_common()],
              'files_over_10_mib': sorted(large, key=lambda x:x['mib'], reverse=True),
              'text_files_scanned': len(scan), 'potential_secret_findings': findings, 'remote_credential_risks':remote_risks,'notebooks': notebooks,
              'limits': 'Pattern audit is not a guarantee. Dataset images, binary models, environments and arbitrary cached JSON were not scanned for private content.',
              'deleted_files': [], 'staged_files_by_this_tool': []}
    with (output/'REPOSITORY_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:result[k] for k in ['branch','tracked_files','text_files_scanned','potential_secret_findings','notebooks']}, indent=2))
    print('Largest folders:', json.dumps(result['top_level_inventory'][:8]))
    print('Largest files:', json.dumps(result['files_over_10_mib'][:12]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='reports/submission_preparation/run-local-'+time.strftime('%Y%m%d-%H%M%S'))
    audit(p.parse_args().output)
