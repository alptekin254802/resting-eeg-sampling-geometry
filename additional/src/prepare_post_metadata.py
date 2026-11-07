"""Reconstruct the post-recording metadata selection from a versioned BIDS tree."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA256 = 'f0b0ec15935aa00d8120150e7eea01e59d50e3fb45a83ad63ce6d1aa5f66ae78'

def build_rows(snapshot, participant_ids):
    rows = []
    for pid in participant_ids:
        base = f'{pid}/ses-1/eeg/{pid}_ses-1_task-EyesClosed_acq-post'
        edf = snapshot / (base + '_eeg.edf')
        reasons = []
        row = dict(participant_id=str(pid), path=base+'_eeg.edf', expected_bytes=0,
                   expected_sha256='', structurally_eligible=False, reason='')
        if not edf.exists() and not edf.is_symlink():
            reasons.append('missing post EDF pointer')
        else:
            if edf.is_symlink():
                pointer_text = str(edf.readlink())
            else:
                if edf.stat().st_size > 8192:
                    raise ValueError('Expected a small annex pointer, not an EEG payload: '+str(edf))
                pointer_text = edf.read_text()
            match = re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})', pointer_text)
            if match:
                row.update(expected_bytes=int(match[1]), expected_sha256=match[2])
            else:
                reasons.append('unrecognized annex pointer')
            side = json.loads((snapshot/(base+'_eeg.json')).read_text(encoding='utf-8'))
            with (snapshot/(base+'_channels.tsv')).open(encoding='utf-8-sig') as stream:
                channels = [r['name'] for r in csv.DictReader(stream, delimiter='\t')]
            with (snapshot/(base+'_events.tsv')).open(encoding='utf-8-sig') as stream:
                events = list(csv.DictReader(stream, delimiter='\t'))
            if any(c not in channels for c in ['O1','Oz','O2','TP9','TP10']):
                reasons.append('missing required channel')
            if side.get('RecordingType') != 'continuous':
                reasons.append('not continuous metadata')
            if float(side.get('RecordingDuration',0)) < 180:
                reasons.append('duration <180s metadata')
            if not events or {e.get('type') for e in events} != {'boundary'}:
                reasons.append('event class not ONLY-INITIAL-BOUNDARY')
            row.update(event_count=len(events), boundary_onsets=';'.join(e.get('onset','') for e in events),
                       metadata_duration=side.get('RecordingDuration'))
        row.update(structurally_eligible=not reasons, reason='; '.join(reasons))
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional new empty directory for reconstructed metadata')
    args = parser.parse_args()
    source = ROOT/'inputs/scientific_results.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError('Original scientific input checksum differs')
    with np.load(source, allow_pickle=False) as archive:
        ids = archive['primary_participant_ids'].astype(str)
    rows = build_rows(args.snapshot_root, ids)
    expected = json.loads((ROOT/'metadata/post_metadata.json').read_text(encoding='utf-8'))
    if rows != expected['rows']:
        differing = [a['participant_id'] for a,b in zip(rows,expected['rows']) if a != b]
        raise ValueError('Reconstructed metadata differs: '+', '.join(differing))
    report = {k:expected[k] for k in ['dataset','version','snapshot_commit']}
    report.update(rows=rows,total_candidates=len(rows),eligible=sum(r['structurally_eligible'] for r in rows),
                  download_bytes=sum(r['expected_bytes'] for r in rows if r['structurally_eligible']))
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Choose a new empty metadata output directory')
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'post_metadata.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (args.output/'post_structural_roster.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='PASS',participants=len(rows),eligible=report['eligible'],
                         snapshot_commit=report['snapshot_commit'],scope='Metadata only; no raw EEG read')))

if __name__ == '__main__':
    main()
