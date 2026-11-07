"""Test metadata reconstruction and resumable preparation checksums."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_post_metadata import build_rows
from receipt_provenance import signature, validate_receipt, SOURCE_NAMES, digest
from sampling_design import seed64

class TestMetadata(unittest.TestCase):
    def fixture(self,root,duration=180,events='boundary\t1\n'):
        base=root/'sub-001/ses-1/eeg/sub-001_ses-1_task-EyesClosed_acq-post'
        base.parent.mkdir(parents=True)
        Path(str(base)+'_eeg.edf').write_text('SHA256E-s100--'+'a'*64+'.edf')
        Path(str(base)+'_eeg.json').write_text(json.dumps({'RecordingType':'continuous','RecordingDuration':duration}))
        Path(str(base)+'_channels.tsv').write_text('name\nO1\nOz\nO2\nTP9\nTP10\n')
        Path(str(base)+'_events.tsv').write_text('type\tonset\n'+events)
        return base
    def test_multiple_boundary_events_are_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d),events='boundary\t1\nboundary\t200\n')
            row=build_rows(Path(d),['sub-001'])[0]
            self.assertTrue(row['structurally_eligible']);self.assertEqual(row['event_count'],2)
            self.assertEqual(row['boundary_onsets'],'1;200')
    def test_duration_and_event_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d),duration=179,events='other\t1\n')
            row=build_rows(Path(d),['sub-001'])[0]
            self.assertFalse(row['structurally_eligible'])
            self.assertIn('duration <180s metadata',row['reason'])
            self.assertIn('event class',row['reason'])
    def test_payload_is_not_parsed_as_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            base=self.fixture(Path(d));Path(str(base)+'_eeg.edf').write_bytes(b'x'*8193)
            with self.assertRaisesRegex(ValueError,'small annex pointer'):build_rows(Path(d),['sub-001'])
    def test_missing_pointer_is_an_explicit_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(build_rows(Path(d),['sub-001'])[0]['reason'],'missing post EDF pointer')
    def test_broken_annex_symlink_uses_target_text_without_following_payload(self):
        with tempfile.TemporaryDirectory() as d:
            base=self.fixture(Path(d));edf=Path(str(base)+'_eeg.edf')
            target=Path('../../.git/annex/objects/SHA256E-s100--'+'a'*64+'.edf')
            original_exists=Path.exists;original_read_text=Path.read_text
            def read_text(path,*args,**kwargs):
                if path==edf:raise AssertionError('Signal payload must not be followed')
                return original_read_text(path,*args,**kwargs)
            with patch.object(Path,'exists',lambda path:False if path==edf else original_exists(path)), \
                 patch.object(Path,'is_symlink',lambda path:path==edf), \
                 patch.object(Path,'readlink',return_value=target), \
                 patch.object(Path,'read_text',read_text):
                row=build_rows(Path(d),['sub-001'])[0]
            self.assertTrue(row['structurally_eligible']);self.assertEqual(row['expected_bytes'],100)
            self.assertEqual(row['expected_sha256'],'a'*64)

class TestReceipts(unittest.TestCase):
    def fixture(self,d):
        root=Path(d);(root/'src').mkdir();(root/'configuration.json').write_text('{}')
        for name in SOURCE_NAMES:(root/'src'/name).write_text('# source\n')
        row=dict(participant_id='sub-001',path='test.edf',expected_bytes=3,expected_sha256=hashlib.sha256(b'raw').hexdigest())
        payload=root/'sub-001.npz';payload.write_bytes(b'payload')
        record=dict(participant_id='sub-001',status='INCLUDED',preparation_signature=signature(root,row),payload_sha256=digest(payload))
        return root,row,payload,record
    def test_unchanged_receipt_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d)
            self.assertEqual(validate_receipt(root,row,record,payload),record)
    def test_configuration_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d);(root/'configuration.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Preparation record differs'):validate_receipt(root,row,record,payload)
    def test_payload_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d);payload.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'payload'):validate_receipt(root,row,record,payload)
    def test_input_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d);row['expected_bytes']=4
            with self.assertRaisesRegex(ValueError,'Preparation record differs'):validate_receipt(root,row,record,payload)
    def test_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d);(root/'src/common.py').write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'Preparation record differs'):validate_receipt(root,row,record,payload)
    def test_old_unbound_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,row,payload,record=self.fixture(d);record.pop('preparation_signature')
            with self.assertRaisesRegex(ValueError,'Preparation record differs'):validate_receipt(root,row,record,payload)

class TestSeedCompatibility(unittest.TestCase):
    def test_original_stream_values(self):
        cases=[(('ds005385','sub-001','C',0,'A'),64164029775061389),
               (('ds005385','sub-001','F10',199,'B9'),2613289620052266220),
               (('ds004148','sub-59','F5',17,'B'),15499565847866722513)]
        for args,expected in cases:self.assertEqual(seed64(*args),expected)

if __name__=='__main__':unittest.main()
