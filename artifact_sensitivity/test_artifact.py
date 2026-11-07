"""Short outcome-free checks; python test_artifact.py [--output PATH]."""
import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from run_artifact import DEFAULT_REPO, emg_mask, validate_schedule, lock_plan

parser=argparse.ArgumentParser()
parser.add_argument('--repo-root',type=Path,default=DEFAULT_REPO)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args,remaining=parser.parse_known_args()
sys.path.insert(0,str(args.repo_root/'scripts'))
from reproduce_inputs import rebuild_schedule, prepared_candidates


class ArtifactChecks(unittest.TestCase):
    def test_strict_subset_and_emg_burst(self):
        rate=500; t=np.arange(2000)/rate
        amplitude=np.exp(np.linspace(-0.2,0.2,42));amplitude[20]=100
        posterior=(amplitude[:,None]*np.sin(2*np.pi*70*t)).reshape(-1)*1e-6
        data=np.stack([posterior,posterior,posterior,np.zeros_like(posterior),np.zeros_like(posterior)])
        original=np.ones(42,dtype=bool);original[2]=False
        strict,flags,*_=emg_mask(data,rate,original)
        self.assertFalse((strict & ~original).any())
        self.assertTrue(flags[20])
        self.assertFalse(flags[2])
        self.assertTrue(np.array_equal(strict,original & ~flags))

    def test_schedule_mask_nonoverlap_and_geometry(self):
        mask=np.ones(42,dtype=bool);mask[41]=False
        schedule,counts=rebuild_schedule(mask,'ds005385-artifact-emg-2026-11','TEST')
        self.assertEqual(counts,[200,200,200]);validate_schedule(mask,schedule)
        self.assertTrue((np.diff(schedule[0],axis=-1)==1).all())
        self.assertTrue((np.diff(schedule[1,...,::2],axis=-1)>=3).all())
        self.assertTrue((schedule[1,...,1::2]-schedule[1,...,::2]==1).all())
        prepared_candidates.cache_clear()

    def test_validator_rejects_overlap(self):
        bad=np.broadcast_to(np.arange(10,dtype=np.uint8),(3,200,2,10)).copy()
        with self.assertRaises(AssertionError):
            validate_schedule(np.ones(42,dtype=bool),bad)

    def test_saved_artifact_invariants(self):
        if args.output is None:
            self.skipTest('No real output directory supplied')
        out=args.output
        lock_plan(out)
        stage=json.loads((out/'stage1_feasibility.json').read_text(encoding='utf-8'))
        record=json.loads((out/'PLAN_RECORD.json').read_text(encoding='utf-8'))
        self.assertLess(record['recorded_at_utc'],stage['started_at_utc'])
        for cohort in ('ds005385','ds004148'):
            with np.load(out/f'masks_emg_{cohort}.npz',allow_pickle=False) as a:
                self.assertFalse((a['strict_masks'] & ~a['original_masks']).any())
                self.assertTrue(np.array_equal(a['strict_masks'],a['original_masks'] & ~a['flagged_masks']))
                self.assertTrue(np.array_equal(a['retained_indices'],np.flatnonzero((a['valid_draw_counts']==200).all(axis=1))))
                self.assertEqual(len(a['retained_indices']),stage['cohorts'][cohort]['retained_N'])
                schedule_path=out/f'schedules_{cohort}.npz'
                if schedule_path.exists():
                    with np.load(schedule_path,allow_pickle=False) as s:
                        for mask,schedule in zip(s['strict_masks'],s['schedule']):
                            validate_schedule(mask,schedule)
        if stage['stop_rule_triggered']:
            self.assertLess(stage['cohorts']['ds005385']['retained_N'],400)
            self.assertFalse(list(out.glob('schedules_*.npz')))
            self.assertFalse(list(out.glob('inference_*.npz')))
            summary=json.loads((out/'results_summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['relative_alpha_or_rank_outcomes_calculated'])
        manifest_path=out/'OUTPUT_MANIFEST.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            for name,expected in manifest['files'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),expected,name)
        else:
            manifest=json.loads((args.repo_root/'release_manifest.json').read_text(encoding='utf-8'))
            for name,expected in manifest['files'].items():
                if name.startswith('artifact_sensitivity/'):
                    self.assertEqual(hashlib.sha256((args.repo_root/name).read_bytes()).hexdigest(),expected['sha256'],name)


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]]+remaining,verbosity=2)
