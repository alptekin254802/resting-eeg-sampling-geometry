"""Check saved muscle-artifact results from the repository root."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from run_artifact import DEFAULT_REPO, sha, lock_plan

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=DEFAULT_REPO)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--full-bootstrap',action='store_true',help='Recompute all 5000 bootstrap resamples')
    args=parser.parse_args();root=args.repo_root.resolve();out=args.output.resolve()
    sys.path.insert(0,str(root/'scripts'))
    from preregistered_analysis import (primary_bootstrap_distribution, point_statistics,
        generate_bootstrap_indices,bootstrap_index_sha256,percentile_interval)
    lock_plan(out)
    summary=json.loads((out/'results_summary.json').read_text('utf-8'))
    assert summary['status'] in ('complete','stopped_after_stage1')
    record=json.loads((out/'RUN_RECORD.json').read_text('utf-8'))
    for relative,digest in record['input_sha256'].items():
        # Historical script checksums are retained in the sanitized records.
        if not record.get('sanitized') or relative.endswith('.npz'):
            assert sha(root/relative)==digest,relative
    package=root/'artifact_sensitivity/PACKAGE_CHECKSUMS.json'
    if package.exists():
        for relative,digest in json.loads(package.read_text('utf-8'))['files'].items():
            assert sha(root/relative)==digest,relative
    tests=subprocess.run([sys.executable,'-B',str(root/'artifact_sensitivity/test_artifact.py'),
        '--repo-root',str(root),'--output',str(out)],capture_output=True,text=True,encoding='utf-8')
    print(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Artifact tests failed')
    stage=json.loads((out/'stage1_feasibility.json').read_text('utf-8'))
    checks={}
    if summary['status']=='complete':
        with np.load(root/'results/preregistered/scientific_results.npz',allow_pickle=False) as original:
            for cohort,s in summary['cohorts'].items():
                role='primary' if cohort=='ds005385' else 'replication'
                with np.load(out/f'inference_{cohort}.npz',allow_pickle=False) as data:
                    expected=generate_bootstrap_indices(s['N'],'artifact_emg_'+cohort)
                    assert np.array_equal(data['bootstrap_indices'],expected)
                    assert bootstrap_index_sha256(expected)==s['bootstrap']['indices_sha256']
                    np.testing.assert_allclose(data['paired_bootstrap_change'],data['strict_bootstrap_delta_z']-data['original_bootstrap_delta_z'],rtol=0,atol=1e-12)
                    lookup={pid:i for i,pid in enumerate(original[role+'_participant_ids'])}
                    indices=np.asarray([lookup[pid] for pid in data['participant_ids']])
                    assert np.array_equal(data['original_markers'],original[role+'_markers'][indices])
                    for kind,key in [('strict','strict'),('original','original_same_subset')]:
                        markers=data[kind+'_markers'];point=point_statistics(markers)
                        for field in ['delta_z','mcse']:
                            np.testing.assert_allclose(point[field],s[key][field],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(point['draw_z'],data[kind+'_draw_fisher_z'],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(point['back_transformed_rho'],[s[key]['rho_tilde'][r] for r in ['C','F5','F10']],rtol=0,atol=1e-12)
                        bootstrap=data[kind+'_bootstrap_delta_z']
                        np.testing.assert_allclose(percentile_interval(bootstrap),s[key]['ci95'],rtol=0,atol=1e-12)
                        chosen=np.arange(5000) if args.full_bootstrap else np.asarray([0,4999])
                        computed=primary_bootstrap_distribution(markers[:,[0,2]],expected[chosen])
                        np.testing.assert_allclose(computed,bootstrap[chosen],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(percentile_interval(data['paired_bootstrap_change']),s['paired_change']['ci95'],rtol=0,atol=1e-12)
                    difference=data['strict_draw_fisher_z'][2]-data['strict_draw_fisher_z'][0]-data['original_draw_fisher_z'][2]+data['original_draw_fisher_z'][0]
                    np.testing.assert_allclose(np.std(difference,ddof=1)/np.sqrt(200),s['paired_change']['mcse'],rtol=0,atol=1e-12)
                    with np.load(out/f'schedules_{cohort}.npz',allow_pickle=False) as schedules:
                        digest=hashlib.sha256(np.ascontiguousarray(schedules['schedule'],dtype='u1').tobytes()).hexdigest()
                    assert digest==s['schedule_hashes']['schedule_uint8_C_order_sha256']
                    assert sha(out/f'schedules_{cohort}.npz')==s['schedule_hashes']['file_sha256']
                    checks[cohort]={'N':s['N'],'all_5000_indices':'PASS','original_subset':'PASS',
                        'draw_correlations_and_points':'PASS','stored_percentiles_and_MCSE':'PASS',
                        'bootstrap_recomputed':'all 5000' if args.full_bootstrap else 'first and last resamples'}
    else:
        assert stage['stop_rule_triggered'] and stage['cohorts']['ds005385']['retained_N']<400
        assert not list(out.glob('inference_*.npz'))
    table=(out/'table_artifact_sensitivity.tex').read_text('utf-8')
    assert '\u2013' not in table and '\u2014' not in table
    assert all(line.endswith('\\\\') for line in table.splitlines() if ' & ' in line)
    assert all(sum(p['flagged'] for p in d['cell_positions'])==d['flagged_cells_total'] for d in stage['cohorts'].values())
    print(json.dumps({'status':'PASS','cohorts':checks,'plan_checksums':'PASS'},indent=2))

if __name__=='__main__':main()
