from common import *
from scipy.optimize import linear_sum_assignment
import argparse
def baseline():
    old,cells=load_inputs()
    np.testing.assert_array_equal(old['sensitivity_participant_ids'],cells['participant_ids'])
    np.testing.assert_array_equal(old['sensitivity_schedule'][:,[0,2]],cells['schedule'])
    rec=markers(cells['cell_band_powers'],cells['schedule'])
    np.testing.assert_allclose(rec,old['sensitivity_markers_C_F10'],atol=1e-12,rtol=0)
    z,rho=point(rec)
    np.testing.assert_allclose(rho,old['sensitivity_draw_rho_C_F10'],atol=1e-12,rtol=0)
    dump(OUT/'metadata/input_reconstruction.json',dict(utc=now(),max_marker_difference=float(np.max(abs(rec-old['sensitivity_markers_C_F10']))),delta_z=float(z[1]-z[0]),passed=True))
    return old,cells
def agreement():
    old,_=baseline();rows=[]
    for label,key in [('primary531','primary'),('replication59','replication')]:
        m=old[key+'_markers'];p=np.median(100*np.abs(m[...,0]-m[...,1]),axis=2)
        ind=boot_indices(len(m),'agreement|'+label);b=p[ind].mean(axis=1)
        for j,name in enumerate(['C','F5','F10']):
            rows.append(dict(cohort=label,regime=name,n=len(m),mean_participant_median_pp=float(p[:,j].mean()),ci_low=ci(b[:,j])[0],ci_high=ci(b[:,j])[1]))
        contrasts=[]
        for j in [1,2]:
            contrasts.append(dict(contrast=['','F5-C','F10-C'][j],estimate_pp=float((p[:,j]-p[:,0]).mean()),ci=ci(b[:,j]-b[:,0]),fraction_participants_smaller=float(np.mean(p[:,j]<p[:,0]))))
        dump(OUT/'results'/f'agreement_{label}.json',dict(cohort=label,n=len(m),unit='percentage points',contrasts=contrasts))
        np.savez_compressed(OUT/'results'/f'agreement_{label}.npz',participant_ids=old[key+'_participant_ids'],participant_medians_pp=p,bootstrap_means_pp=b,bootstrap_indices=ind)
    csvwrite(OUT/'results/agreement_summary.csv',rows);log('agreement complete')
def pairing_prepare():
    old,c=baseline();s=c['schedule'].astype(int);n,g,d,a,k=s.shape
    permutations=np.empty((n,g,3,d),dtype=np.int16); distances=np.empty((n,g,3,d));changed=np.empty((n,g,3))
    for i,pid in enumerate(c['participant_ids']):
        for j,geom in enumerate(['C','F10']):
            arms=s[i,j];A=arms[:,0];B=arms[:,1]
            nearestA=np.abs(A[:,:,None]-np.arange(42)).min(axis=1)
            nearestB=np.abs(B[:,:,None]-np.arange(42)).min(axis=1)
            cost=nearestB[:,A].sum(axis=-1).T+nearestA[:,B].sum(axis=-1)
            bitsA=np.sum(np.left_shift(np.uint64(1),A.astype(np.uint64)),axis=1)
            bitsB=np.sum(np.left_shift(np.uint64(1),B.astype(np.uint64)),axis=1)
            valid=(bitsA[:,None]&bitsB[None,:])==0
            assert np.diag(valid).all()
            jitter=rng(f'pairing-tie|{pid}|{geom}').integers(0,1_000_000,(d,d),dtype=np.int64)
            ps=[np.arange(d)]
            for direction in [1,-1]:
                objective=direction*cost.astype(np.int64)*1_000_000_000+jitter
                objective[~valid]=10**17
                rr,cc=linear_sum_assignment(objective)
                assert np.array_equal(rr,np.arange(d)) and len(np.unique(cc))==d and valid[rr,cc].all()
                ps.append(cc)
            for h,p in enumerate(ps):
                permutations[i,j,h]=p; distances[i,j,h]=cost[np.arange(d),p]/5;changed[i,j,h]=np.mean(p!=np.arange(d))
        if (i+1)%100==0:log(f'pairing feasibility {i+1}/{n}')
    original=old['sensitivity_markers_C_F10'];new=[];schedules=[]
    for h in range(3):
        for j in range(2):
            out=original[:,j].copy();out[:,:,1]=np.take_along_axis(original[:,j,:,1],permutations[:,j,h],axis=1)
            new.append(out)
            sp=s[:,j].copy();sp[:,:,1]=np.take_along_axis(s[:,j,:,1],permutations[:,j,h,:,None],axis=1);validate_pairs(sp)
            np.testing.assert_array_equal(np.sort(out[:,:,1],axis=1),np.sort(original[:,j,:,1],axis=1));schedules.append(sp)
    mark=np.stack(new,axis=1)
    np.savez_compressed(OUT/'results/pairing_prepared.npz',participant_ids=c['participant_ids'],permutations=permutations,distances_s=distances,changed_fraction=changed,markers=mark)
    rows=[dict(regime=geom,pairing=lab,n=n,mean_distance_s=float(distances[:,j,h].mean()),mean_changed_fraction=float(changed[:,j,h].mean())) for j,geom in enumerate(['C','F10']) for h,lab in enumerate(['original','near','far'])]
    csvwrite(OUT/'results/pairing_feasibility.csv',rows)
    dump(OUT/'results/pairing_feasibility.json',dict(utc=now(),outcomes_not_used_for_matching=True,all_bijections_nonoverlapping=True,marginal_markers_unchanged=True,rows=rows))
    log('pairing feasibility written before new rank inference')
def pairing_infer():
    p=OUT/'results/pairing_prepared.npz'
    if not p.exists():pairing_prepare()
    with np.load(p) as v:m=v['markers']
    labels=[x+'_'+g for x in ['original','near','far'] for g in ['C','F10']]
    z,bz,rows=summarize(m,labels,'pairing');con=[]
    for h,lab in enumerate(['original','near','far']):
        a,b=2*h,2*h+1;con.append(dict(contrast=lab+'_F10-C',estimate=float(z[b]-z[a]),ci=ci(bz[:,b]-bz[:,a])))
    for h,lab in [(1,'near-original'),(2,'far-original')]:
        change=(bz[:,2*h+1]-bz[:,2*h])-(bz[:,1]-bz[:,0]);estimate=(z[2*h+1]-z[2*h])-(z[1]-z[0])
        con.append(dict(contrast=lab+'_delta',estimate=float(estimate),ci=ci(change)))
    con.append(dict(contrast='far-near_delta',estimate=float((z[5]-z[4])-(z[3]-z[2])),ci=ci((bz[:,5]-bz[:,4])-(bz[:,3]-bz[:,2]))))
    dump(OUT/'results/pairing_summary.json',dict(n=len(m),conditions=rows,contrasts=con));log('pairing inference complete')
def normalization():
    old,c=baseline();m=np.concatenate([markers(c['cell_band_powers'],c['schedule'],f) for f in ['relative','absolute','nonalpha']],axis=1)
    assert np.isfinite(m).all() and (m>0).all()
    labels=[f+'_'+g for f in ['relative_alpha','absolute_alpha','nonalpha'] for g in ['C','F10']]
    z,bz,rows=summarize(m,labels,'normalization');con=[]
    for h,lab in enumerate(['relative_alpha','absolute_alpha','nonalpha']):
        a,b=2*h,2*h+1;con.append(dict(feature=lab,delta_z=float(z[b]-z[a]),ci=ci(bz[:,b]-bz[:,a]),difference_from_relative=float((z[b]-z[a])-(z[1]-z[0])),difference_ci=ci((bz[:,b]-bz[:,a])-(bz[:,1]-bz[:,0]))))
    np.savez_compressed(OUT/'results/normalization_markers.npz',participant_ids=c['participant_ids'],markers=m)
    dump(OUT/'results/normalization_summary.json',dict(n=len(m),conditions=rows,contrasts=con));log('normalization complete')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['agreement','pairing_prepare','pairing','normalization']);a=parser.parse_args()
    {'agreement':agreement,'pairing_prepare':pairing_prepare,'pairing':pairing_infer,'normalization':normalization}[a.stage]()
