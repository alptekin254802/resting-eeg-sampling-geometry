"""Bind resumable preparation receipts to source data and computation settings."""
from pathlib import Path
import hashlib
import json

SOURCE_NAMES = ('run_post.py','common.py','eeg_preparation.py','preregistered_analysis.py',
                'post_sampling.py','sampling_design.py','receipt_provenance.py')

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):result.update(chunk)
    return result.hexdigest()

def signature(root,row):
    record={'configuration':digest(root/'configuration.json'),
            'source':{name:digest(root/'src'/name) for name in SOURCE_NAMES},
            'participant':{key:row[key] for key in ['participant_id','path','expected_bytes','expected_sha256']}}
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_receipt(root,row,record,payload):
    if record.get('participant_id')!=row['participant_id'] or record.get('preparation_signature')!=signature(root,row):
        raise ValueError('Preparation record differs; use a fresh directory: '+row['participant_id'])
    if record.get('status') not in ('INCLUDED','EXCLUDED'):
        raise ValueError('Failed preparation cannot be resumed from its record: '+row['participant_id'])
    raw=root/'raw_post'/row['path']
    if raw.exists() and (raw.stat().st_size!=row['expected_bytes'] or digest(raw)!=row['expected_sha256']):
        raise ValueError('Raw file differs from the recorded input: '+row['participant_id'])
    if record['status']=='INCLUDED' and (not payload.is_file() or record.get('payload_sha256')!=digest(payload)):
        raise ValueError('Prepared participant payload differs: '+row['participant_id'])
    return record
