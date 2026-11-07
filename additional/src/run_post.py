from common import *
import time, traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from post_sampling import schedule_mask
from eeg_preparation import _load_required_support,time_domain_qc_mask
from preregistered_analysis import compute_cell_periodograms
from eeg_preparation import cell_band_powers
from receipt_provenance import signature as receipt_signature, validate_receipt
RUNTIME=OUT/'metadata/runtime_home';RUNTIME.mkdir(exist_ok=True)
os.environ['_MNE_FAKE_HOME_DIR']=str(RUNTIME)
PRE_FLAGGED={'sub-114','sub-206','sub-319','sub-493','sub-554'}
def process(row):
    pid=row['participant_id'];path=OUT/'raw_post'/row['path']
    record=dict(participant_id=pid,status='EXCLUDED',reason='',eligible_cells=0,fs_hz=None,post_boundary_flag=int(row.get('event_count',0))>1,pre_boundary_flag=pid in PRE_FLAGGED)
    try:
        if path.stat().st_size!=row['expected_bytes'] or digest(path)!=row['expected_sha256']:raise RuntimeError('Raw file pinned hash/size mismatch')
        with path.open('rb') as f:header=f.read(256)
        nrec=int(header[236:244]);dur=float(header[244:252]);reserved=header[192:236].decode('latin1').strip()
        if nrec*dur<180:raise ValueError('actual EDF duration <180s')
        if reserved.startswith('EDF+D'):raise ValueError('discontinuous EDF+D')
        data,fs=_load_required_support(path,'edf');mask=time_domain_qc_mask(data*1e6,fs)
        record.update(eligible_cells=int(mask.sum()),fs_hz=fs,actual_duration_s=nrec*dur,raw_sha256=row['expected_sha256'],mask=''.join('1' if x else '0' for x in mask))
        sch,ok,why=schedule_mask(mask,'ds005385-post-followup-2026-11-01',pid)
        record.update(C_feasible=bool(ok[0]),F5_feasible=bool(ok[1]),F10_feasible=bool(ok[2]))
        if not ok.all():record['reason']='; '.join(f'{g}: {w}' for g,w in zip(['C','F5','F10'],why) if w);return record,None
        frequency,psd=compute_cell_periodograms(data,fs);power=cell_band_powers(psd,frequency)
        mark=markers(power[None],sch[None])[0]
        if not np.isfinite(mark).all() or not np.all((mark>0)&(mark<1)):raise ValueError('Invalid relative alpha values')
        record['status']='INCLUDED'
        return record,dict(schedule=sch,mask=mask,power=power,markers=mark)
    except ValueError as ex:record['reason']=str(ex);return record,None
    except Exception as ex:record['status']='ERROR';record['reason']=str(ex);return record,None
def prepare():
    import mne
    meta=json.loads((OUT/'metadata/post_metadata.json').read_text());eligible=[r for r in meta['rows'] if r['structurally_eligible']]
    partial=OUT/'results/post_participants';partial.mkdir(exist_ok=True)
    missing=[OUT/'raw_post'/r['path'] for r in eligible
             if not (partial/(r['participant_id']+'.json')).is_file()
             and not (OUT/'raw_post'/r['path']).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} required post-recording file(s) missing. Download the versioned files before preparation. No recordings were skipped.\n'
            + '\n'.join(str(p) for p in missing))
    pending={r['participant_id']:r for r in eligible};records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        active={}
        while pending or active:
            for pid,row in list(pending.items()):
                record_path=partial/(pid+'.json')
                if record_path.exists():
                    record=validate_receipt(OUT,row,json.loads(record_path.read_text()),partial/(pid+'.npz'))
                    records.append(record);del pending[pid];continue
                path=OUT/'raw_post'/row['path']
                if path.exists() and len(active)<6:active[pool.submit(process,row)]=pid;del pending[pid]
            for future in list(active):
                if future.done():
                    pid=active.pop(future);record,payload=future.result();records.append(record)
                    row=next(r for r in eligible if r['participant_id']==pid)
                    record['preparation_signature']=receipt_signature(OUT,row)
                    if payload is not None:
                        np.savez_compressed(partial/(pid+'.npz'),**payload)
                        record['payload_sha256']=digest(partial/(pid+'.npz'))
                    dump(partial/(pid+'.json'),record)
                    if len(records)%25==0:log(f'post QC/PSD {len(records)}/{len(eligible)}')
            if pending and not active:
                raise FileNotFoundError('Post-recording files became unavailable during preparation: '
                                        + ', '.join(str(OUT/'raw_post'/r['path']) for r in pending.values()))
            if pending or active:time.sleep(.5)
    records=sorted(records,key=lambda r:r['participant_id'])
    fields=sorted(set().union(*(r.keys() for r in records)))
    csvwrite(OUT/'results/post_qc_roster.csv',[{k:r.get(k,'') for k in fields} for r in records])
    error=[r for r in records if r['status'] in ['ERROR','ACCESS_FAILED']]
    if error:dump(OUT/'results/post_processing_errors.json',error);raise RuntimeError(f'{len(error)} post files have unresolved processing/access errors')
    ids=[r['participant_id'] for r in records if r['status']=='INCLUDED'];values={k:[] for k in ['schedule','mask','power','markers']}
    for pid in ids:
        with np.load(partial/(pid+'.npz')) as q:
            for k in values:values[k].append(q[k])
    payload={k:np.array(v) for k,v in values.items()}
    np.savez_compressed(OUT/'results/post_prepared.npz',participant_ids=np.array(ids),**payload)
    dump(OUT/'results/post_preparation_summary.json',dict(completed_utc=now(),original_pre_n=531,structurally_eligible_post_n=len(eligible),included_post_n=len(ids),excluded_metadata=[r for r in meta['rows'] if not r['structurally_eligible']],excluded_qc=[r for r in records if r['status']!='INCLUDED'],all_included_raw_sha256_verified=True))
    log(f'post preparation complete: {len(ids)} common pre/post eligible participants')
def infer():
    old,_=load_inputs()
    with np.load(OUT/'results/post_prepared.npz') as q:ids=q['participant_ids'];post=q['markers'];sch=q['schedule']
    lookup={p:i for i,p in enumerate(old['primary_participant_ids'])};ind=np.array([lookup[p] for p in ids]);pre=old['primary_markers'][ind]
    validate_pairs(sch)
    conditions=[];arrays=[]
    for kind,data in [('within_pre',pre),('within_post',post),('cross_A',np.stack([pre[...,0],post[...,0]],axis=-1)),('cross_B',np.stack([pre[...,1],post[...,1]],axis=-1))]:
        for j,g in enumerate(['C','F5','F10']):conditions.append(kind+'_'+g);arrays.append(data[:,j])
    mark=np.stack(arrays,axis=1)
    # Two six-condition blocks share the exact participant bootstrap indices.
    bi=boot_indices(len(ids),'pre_post_common');zparts=[];bparts=[];rows=[]
    for lo,hi,part in [(0,6,'within'),(6,12,'cross')]:
        z,bz,r=summarize(mark[:,lo:hi],conditions[lo:hi],'pre_post_'+part,bi);zparts.append(z);bparts.append(bz);rows.extend(r)
    z=np.concatenate(zparts);bz=np.concatenate(bparts,axis=1)
    contrasts=[]
    for start,label in [(0,'within_pre'),(3,'within_post'),(6,'cross_A'),(9,'cross_B')]:
        contrasts.append(dict(contrast=label+'_F10-C',estimate=float(z[start+2]-z[start]),ci=ci(bz[:,start+2]-bz[:,start])))
    contrasts.append(dict(contrast='cross_A_minus_within_pre_delta',estimate=float((z[8]-z[6])-(z[2]-z[0])),ci=ci((bz[:,8]-bz[:,6])-(bz[:,2]-bz[:,0]))))
    dump(OUT/'results/pre_post_summary.json',dict(n=len(ids),conditions=rows,contrasts=contrasts,interpretation='Different occasions separated by cognitive battery; not a pure retest or causal fatigue experiment'))
    np.savez_compressed(OUT/'results/pre_post_markers.npz',participant_ids=ids,markers=mark,mean_z=z,bootstrap_z=bz,conditions=np.array(conditions))
    meta=json.loads((OUT/'metadata/post_metadata.json').read_text());post_flags={r['participant_id'] for r in meta['rows'] if int(r.get('event_count',0))>1}
    keep=np.array([p not in PRE_FLAGGED and p not in post_flags for p in ids]);removed=ids[~keep]
    if not len(removed):dump(OUT/'results/pre_post_boundary_summary.json',dict(status='IDENTICAL_SAMPLE',n=len(ids),excluded=[]))
    else:
        # C/F10 for both within-record comparisons and the cross-A primary transfer.
        select=[0,2,3,5,6,8];labels=[conditions[k] for k in select]
        zz,bb,rr=summarize(mark[keep][:,select],labels,'pre_post_boundary')
        con=[dict(contrast=lab+'_F10-C',estimate=float(zz[2*j+1]-zz[2*j]),ci=ci(bb[:,2*j+1]-bb[:,2*j])) for j,lab in enumerate(['within_pre','within_post','cross_A'])]
        dump(OUT/'results/pre_post_boundary_summary.json',dict(status='COMPLETE',n=int(keep.sum()),excluded=removed.tolist(),conditions=rr,contrasts=con))
    log('pre/post inference complete')
if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('prepare or infer')
    {'prepare':prepare,'infer':infer}[sys.argv[1]]()
