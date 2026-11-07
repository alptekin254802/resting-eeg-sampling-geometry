"""Check raw-recording validation and missing-input handling."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import eeg_preparation
from preregistered_analysis import NumericalFailure, REQUIRED_CHANNELS
import run_post


class TestRawPreparation(unittest.TestCase):
    def raw(self, channels=REQUIRED_CHANNELS, duration=180, data=None):
        if data is None:
            data = np.zeros((5, 1680))
        return SimpleNamespace(ch_names=list(channels), info={'sfreq': 10.0},
                               n_times=duration*10,
                               get_data=lambda **kwargs: data)

    def check_failure(self, raw, message):
        with patch('mne.io.read_raw_edf', return_value=raw):
            with self.assertRaisesRegex(NumericalFailure, message):
                eeg_preparation._load_required_support(Path('example.edf'), 'edf')

    def test_missing_channel_has_defined_scientific_error(self):
        self.check_failure(self.raw(channels=REQUIRED_CHANNELS[:-1]), 'missing required channels')

    def test_short_recording_rejected(self):
        self.check_failure(self.raw(duration=179), 'shorter than')

    def test_invalid_support_shape_rejected(self):
        self.check_failure(self.raw(data=np.zeros((5, 1679))), 'required support shape')

    def test_nonfinite_support_rejected(self):
        data=np.zeros((5,1680));data[0,0]=np.nan
        self.check_failure(self.raw(data=data), 'nonfinite')

    def test_valid_support_unchanged(self):
        data=np.arange(8400,dtype=float).reshape(5,1680)
        with patch('mne.io.read_raw_edf', return_value=self.raw(data=data)):
            observed, rate=eeg_preparation._load_required_support(Path('example.edf'), 'edf')
        np.testing.assert_array_equal(observed,data)
        self.assertEqual(rate,10.0)

    def test_missing_recordings_fail_before_workers_or_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir();(root/'results').mkdir()
            rows=[{'participant_id':p,'path':p+'.edf','structurally_eligible':True}
                  for p in ['sub-001','sub-002']]
            (root/'metadata/post_metadata.json').write_text(json.dumps({'rows':rows}))
            with patch.object(run_post,'OUT',root), patch.object(run_post,'ThreadPoolExecutor') as pool:
                with self.assertRaises(FileNotFoundError) as caught:run_post.prepare()
            self.assertIn('2 required',str(caught.exception))
            self.assertIn('sub-001.edf',str(caught.exception));self.assertIn('sub-002.edf',str(caught.exception))
            pool.assert_not_called()


if __name__=='__main__':unittest.main()
