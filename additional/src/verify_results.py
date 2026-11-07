"""Verify saved additional-analysis results without a new inferential run.

The report is written to standard output. Use --output <path> to also save it
to a file; the path must lie outside the distributed files.
"""
import argparse
from common import *
from post_sampling import test_sampler

def distributed_files():
    files={(OUT/name).resolve() for name in json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))['files']}
    files.add((OUT/'manifest.json').resolve())
    release=OUT.parent/'release_manifest.json'
    if release.is_file():
        files|={(OUT.parent/name).resolve() for name in json.loads(release.read_text(encoding='utf-8'))['files']}
        files.add(release.resolve())
    return files
parser=argparse.ArgumentParser(description='Verify saved additional-analysis results.')
parser.add_argument('--output',type=Path,help='optional file for the JSON report, outside the distributed files')
args=parser.parse_args()
target=args.output.resolve() if args.output else None
if target is not None and target in distributed_files():
    raise SystemExit('Refusing to overwrite a distributed file: '+str(args.output))

checks=[]
def check(label,actual,expected,atol=1e-12):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=atol)
    checks.append(label)
def read(name):return json.loads((OUT/'results'/name).read_text())
def direct_bootstrap(mark,indices,expected,label):
    for k in [0,1,2]:check(label+f' direct bootstrap {k}',point(mark[indices[k]])[0],expected[k])
def saved_inference(mark,name):
    with np.load(OUT/'results'/f'{name}_inference.npz')as q:
        z,rho=point(mark)
        check(name+' points',z,q['mean_z']);check(name+' draw correlations',rho,q['draw_rho'])
        direct_bootstrap(mark,q['bootstrap_indices'],q['bootstrap_z'],name)
        return q['mean_z'].copy(),q['bootstrap_z'].copy(),q['bootstrap_indices'].copy()

old,cells=load_inputs()
check('complete-support marker reconstruction',markers(cells['cell_band_powers'],cells['schedule']),old['sensitivity_markers_C_F10'])
for lab,key in [('primary531','primary'),('replication59','replication')]:
    with np.load(OUT/'results'/f'agreement_{lab}.npz')as q:
        mark=old[key+'_markers'];p=np.median(100*abs(mark[...,0]-mark[...,1]),axis=2)
        np.testing.assert_array_equal(q['participant_ids'],old[key+'_participant_ids'])
        check(lab+' participant medians',p,q['participant_medians_pp'])
        b=p[q['bootstrap_indices']].mean(axis=1);check(lab+' all saved bootstrap means',b,q['bootstrap_means_pp'])
        report=read(f'agreement_{lab}.json')
        for j in [1,2]:
            check(lab+f' contrast {j}',(p[:,j]-p[:,0]).mean(),report['contrasts'][j-1]['estimate_pp'])
            check(lab+f' interval {j}',ci(b[:,j]-b[:,0]),report['contrasts'][j-1]['ci'])

with np.load(OUT/'results/normalization_markers.npz')as q:
    expected=np.concatenate([markers(cells['cell_band_powers'],cells['schedule'],x)for x in ['relative','absolute','nonalpha']],axis=1)
    mark=q['markers'];check('normalization aggregation',mark,expected)
    z,bz,_=saved_inference(mark,'normalization')
    for j,v in enumerate(read('normalization_summary.json')['contrasts']):
        check('normalization contrast '+str(j),z[j*2+1]-z[j*2],v['delta_z'])
        check('normalization interval '+str(j),ci(bz[:,j*2+1]-bz[:,j*2]),v['ci'])

with np.load(OUT/'results/pairing_prepared.npz')as q:
    mark=q['markers'];perms=q['permutations'];base=old['sensitivity_markers_C_F10'];sched=cells['schedule']
    for h in range(3):
        for g in range(2):
            check('pairing unchanged A arms',mark[:,h*2+g,:,0],base[:,g,:,0],0)
            check('pairing B permutation',mark[:,h*2+g,:,1],np.take_along_axis(base[:,g,:,1],perms[:,g,h],axis=1),0)
            check('pairing B marginal values',np.sort(mark[:,h*2+g,:,1],axis=1),np.sort(base[:,g,:,1],axis=1),0)
            new=sched[:,g].copy();new[:,:,1]=np.take_along_axis(sched[:,g,:,1],perms[:,g,h,:,None],axis=1);validate_pairs(new)
    z,bz,_=saved_inference(mark,'pairing');d=z[1::2]-z[::2];bd=bz[:,1::2]-bz[:,::2]
    values=[*d,d[1]-d[0],d[2]-d[0],d[2]-d[1]]
    boot=[bd[:,0],bd[:,1],bd[:,2],bd[:,1]-bd[:,0],bd[:,2]-bd[:,0],bd[:,2]-bd[:,1]]
    for v,e,b in zip(read('pairing_summary.json')['contrasts'],values,boot):
        check('pairing '+v['contrast'],e,v['estimate']);check('pairing interval '+v['contrast'],ci(b),v['ci'])

with np.load(OUT/'results/post_prepared.npz')as p, np.load(OUT/'results/pre_post_markers.npz')as q:
    ids=p['participant_ids'];assert len(ids)==524
    np.testing.assert_array_equal(ids,q['participant_ids'])
    post=p['markers'];check('post markers from cell powers',post,markers(p['power'],p['schedule']))
    validate_pairs(p['schedule'],p['mask'])
    lookup={pid:i for i,pid in enumerate(old['primary_participant_ids'])};pre=old['primary_markers'][[lookup[pid]for pid in ids]]
    expected=np.concatenate([pre,post,np.stack([pre[...,0],post[...,0]],axis=-1),np.stack([pre[...,1],post[...,1]],axis=-1)],axis=1)
    mark=q['markers'];check('pre/post pairing and participant join',mark,expected)
    z1,b1,i1=saved_inference(mark[:,:6],'pre_post_within');z2,b2,i2=saved_inference(mark[:,6:],'pre_post_cross')
    np.testing.assert_array_equal(i1,i2)
    z=np.r_[z1,z2];bz=np.concatenate([b1,b2],axis=1)
    report=read('pre_post_summary.json');d=z[2::3]-z[::3];bd=bz[:,2::3]-bz[:,::3]
    for v,e,b in zip(report['contrasts'],[*d,d[2]-d[0]],[bd[:,j]for j in range(4)]+[bd[:,2]-bd[:,0]]):
        check('pre/post '+v['contrast'],e,v['estimate']);check('pre/post interval '+v['contrast'],ci(b),v['ci'])
    boundary=read('pre_post_boundary_summary.json');keep=np.array([pid not in boundary['excluded']for pid in ids]);assert keep.sum()==513
    bmark=mark[keep][:,[0,2,3,5,6,8]];zz,bb,_=saved_inference(bmark,'pre_post_boundary')
    for j,v in enumerate(boundary['contrasts']):
        check('boundary '+v['contrast'],zz[j*2+1]-zz[j*2],v['estimate']);check('boundary interval '+v['contrast'],ci(bb[:,j*2+1]-bb[:,j*2]),v['ci'])

test_sampler();checks.append('post sampler equals original construction on full and gapped synthetic supports')
report=dict(status='PASS',checks=len(checks),check_descriptions=checks,absolute_tolerance=1e-12,scope='Saved-array points/intervals, pairing/join invariants, three direct ordinary-bootstrap resamples per rank analysis block, all saved agreement bootstrap means, and sampler implementation equivalence. No full raw-EEG replay or full rank-bootstrap rerun.')
print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
if target is not None:
    target.parent.mkdir(parents=True,exist_ok=True);dump(target,report)
print('PASS:',len(checks),'checks')
