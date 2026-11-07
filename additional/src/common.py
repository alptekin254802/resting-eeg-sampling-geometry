"""Utilities for the separately planned 2026-11-01 exploratory follow-up."""
from pathlib import Path
import sys, os, json, hashlib, datetime, csv
sys.dont_write_bytecode = True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT
from preregistered_analysis import spearman_midrank_columns
PLAN=json.loads((OUT/'configuration.json').read_text(encoding='utf-8'))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(s): print(now(),s,flush=True)
def rng(label): return np.random.default_rng(int.from_bytes(hashlib.sha256((PLAN['namespace']+'|'+label).encode()).digest()[:8],'big'))
def dump(path,obj):
    def convert(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(path,rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def ci(x):
    a=np.asarray(x);assert np.isfinite(a).all()
    return np.percentile(a,[2.5,97.5],axis=0,method='linear').tolist()
def load_inputs():
    p=ROOT/'inputs'
    with np.load(p/'scientific_results.npz') as z: original={k:z[k] for k in z.files}
    with np.load(p/'cell_band_powers.npz') as z: cells={k:z[k] for k in z.files}
    return original,cells
def markers(powers,schedule,feature='relative'):
    idx=np.arange(len(powers)).reshape((-1,)+(1,)*(schedule.ndim-1))
    selected=powers[idx,schedule].mean(axis=-2)
    if feature=='relative': return selected[...,0]/selected[...,1]
    if feature=='absolute': return selected[...,0]
    if feature=='nonalpha': return selected[...,1]-selected[...,0]
    raise ValueError(feature)
def point(mark):
    """N x condition x draw x arm; return mean Fisher z and draw rho."""
    n,c,d,_=mark.shape
    rho=spearman_midrank_columns(mark[...,0].reshape(n,c*d),mark[...,1].reshape(n,c*d)).reshape(c,d)
    if not np.all(np.abs(rho)<1): raise ValueError('Undefined/infinite Fisher transform')
    return np.arctanh(rho).mean(axis=1),rho
def boot_indices(n,label): return rng('bootstrap|'+label).integers(0,n,(PLAN['bootstrap_resamples'],n),dtype=np.int64)
def prepare_ranks(values):
    order=np.argsort(values,axis=0,kind='stable')
    ordered=np.take_along_axis(values,order,axis=0)
    n,m=values.shape
    index=np.broadcast_to(np.arange(n)[:,None],(n,m))
    first=np.r_[np.ones((1,m),bool),ordered[1:]!=ordered[:-1]]
    last=np.r_[ordered[:-1]!=ordered[1:],np.ones((1,m),bool)]
    start=np.maximum.accumulate(np.where(first,index,0),axis=0)
    end=np.minimum.accumulate(np.where(last,index,n-1)[::-1],axis=0)[::-1]
    inverse=np.argsort(order,axis=0)
    return order,inverse,start,end,bool(first.all())
def weighted_ranks(prepared,counts):
    order,inverse,start,end,unique=prepared
    w=counts[:,order]
    cumulative=np.cumsum(w,axis=1)
    if unique:
        ranks=cumulative-.5*w+.5
    else:
        hi=np.take_along_axis(cumulative,end[None],axis=1)
        low=np.take_along_axis(cumulative-w,start[None],axis=1)
        ranks=(hi+low+1.)*.5
    return np.take_along_axis(ranks,inverse[None],axis=1)
def bootstrap_z(mark,indices,label,batch=12):
    """Exact ordinary participant bootstrap using multiplicities and midranks."""
    n,c,d,_=mark.shape
    if indices.shape[1]!=n: raise ValueError('Mismatched participant indices')
    left=prepare_ranks(mark[...,0].reshape(n,c*d)); right=prepare_ranks(mark[...,1].reshape(n,c*d))
    output=np.empty((len(indices),c))
    center=(n+1)/2
    for start in range(0,len(indices),batch):
        stop=min(start+batch,len(indices)); counts=np.array([np.bincount(i,minlength=n) for i in indices[start:stop]],dtype=float)
        x=weighted_ranks(left,counts); y=weighted_ranks(right,counts)
        w=counts[:,:,None]
        if left[-1] and right[-1]:
            # Each duplicated participant creates the same tie multiplicity in both arms.
            variance12=n*(n*n-1)-np.sum(counts**3-counts,axis=1)
            rho=1-6*np.sum(w*(x-y)**2,axis=1)/variance12[:,None]
        else:
            x-=center;y-=center
            num=np.sum(w*x*y,axis=1); den=np.sqrt(np.sum(w*x*x,axis=1)*np.sum(w*y*y,axis=1))
            rho=num/den
        if not np.isfinite(rho).all() or np.any(np.abs(rho)>=1): raise ValueError('Undefined bootstrap correlation: '+label)
        output[start:stop]=np.arctanh(rho).reshape(stop-start,c,d).mean(axis=2)
        if start==0 or stop==len(indices) or (start//batch)%50==0:log(f'{label}: bootstrap {stop}/{len(indices)}')
    return output
def summarize(mark,labels,label,indices=None):
    z,rho=point(mark)
    if indices is None: indices=boot_indices(len(mark),label)
    bz=bootstrap_z(mark,indices,label)
    np.savez_compressed(OUT/'results'/f'{label}_inference.npz',draw_rho=rho,mean_z=z,bootstrap_z=bz,bootstrap_indices=indices)
    return z,bz,[dict(condition=k,n=len(mark),rho=float(np.tanh(z[j])),rho_ci=np.tanh(np.percentile(bz[:,j],[2.5,97.5])).tolist()) for j,k in enumerate(labels)]
def validate_pairs(schedule,mask=None):
    a=np.asarray(schedule,dtype=int)
    assert np.all((a>=0)&(a<42))
    assert np.all(np.diff(np.sort(a,axis=-1),axis=-1)>0)
    assert not np.any(a[...,0,:,None]==a[...,1,None,:])
    if mask is not None:
        idx=np.arange(len(mask)).reshape((-1,)+(1,)*(a.ndim-1))
        assert mask[idx,a].all()
def pair_distance(schedule):
    a=np.asarray(schedule,dtype=int)
    diff=np.abs(a[...,0,:,None]-a[...,1,None,:])
    return 2*(diff.min(axis=-1).mean(axis=-1)+diff.min(axis=-2).mean(axis=-1))
