"""Verify every release file against its checksum and byte count.

Local environments and caches are ignored: the directories .venv/, venv/,
.git/, outputs/ and __pycache__/ at any depth, and *.pyc files. Any other
extra or missing file fails the check.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES={'.venv','venv','.git','outputs','__pycache__'}
IGNORED_SUFFIXES={'.pyc'}

def ignored(relative):
    return any(part in IGNORED_DIRECTORIES for part in relative.parts[:-1]) or relative.suffix in IGNORED_SUFFIXES

def release_files(root=ROOT):
    files=set()
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if path.is_file() and not ignored(relative):
            files.add(relative.as_posix())
    return files

def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text('utf-8'))
    observed=release_files()
    expected=set(manifest['files'])|{'release_manifest.json'}
    if observed!=expected:
        raise ValueError('File list differs: '+str(sorted(observed^expected)))
    for relative,value in manifest['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError('Missing or invalid path: '+relative)
        if path.stat().st_size!=value['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=value['sha256']:
            raise ValueError('Checksum or byte count differs: '+relative)
    print(json.dumps({'release_checksums':'PASS','files':len(expected),'version':manifest['version']},indent=2))

if __name__=='__main__':main()
