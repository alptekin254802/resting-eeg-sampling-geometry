"""Verify checksums of the additional-analysis files."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
for name,expected in manifest['files'].items():
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():raise ValueError(name)
    if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('File checksum differs: '+name)
print('PASS:',len(manifest['files']),'file checksums; version',manifest['version'])
