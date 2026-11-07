from common import *
import sampling_design as original_sampling
REGIMES=['C','F5','F10']
FULL_F5=np.asarray(original_sampling.candidates_f5(np.ones(42,bool)),dtype=np.uint8)
FULL_F5_BITS=np.sum(np.left_shift(np.uint64(1),FULL_F5.astype(np.uint64)),axis=1)
def schedule_mask(mask,dataset,pid):
    """Original construction and RNG, accelerated only for F5 candidate tests."""
    result=np.zeros((3,200,2,10),np.uint8);ok=np.zeros(3,bool);reasons=[]
    blocked=np.sum(np.left_shift(np.uint64(1),np.flatnonzero(~mask).astype(np.uint64)))
    valid=(FULL_F5_BITS&blocked)==0;f5=FULL_F5[valid];f5bits=FULL_F5_BITS[valid]
    for j,g in enumerate(REGIMES):
        try:
            if g=='F10' and any(sum(mask[list(st)])<2 for st in original_sampling.STRATA10):raise ValueError('stratum has fewer than two eligible cells')
            prepared=original_sampling.candidates_c(mask) if g=='C' else None
            if g=='F5' and not len(f5):raise ValueError('no F5 candidate arm')
            for b in range(200):
                if g!='F5':
                    result[j,b]=original_sampling.draw_pair(mask,dataset,pid,g,b,prepared)
                else:
                    selected_a_index=int(np.random.default_rng(original_sampling.seed64(dataset,pid,g,b,'A')).integers(0,len(f5)))
                    available=(f5bits&f5bits[selected_a_index])==0
                    if not available.any():raise ValueError('selected F5 A has no disjoint B')
                    gen=np.random.default_rng(original_sampling.seed64(dataset,pid,g,b,'B'));found=False
                    for start in range(0,100000,256):
                        candidate=gen.integers(0,len(f5),size=min(256,100000-start))
                        hits=np.flatnonzero(available[candidate])
                        if hits.size:
                            result[j,b,0]=f5[selected_a_index];result[j,b,1]=f5[candidate[hits[0]]];found=True;break
                    if not found:raise ValueError('100000 F5 B attempts exhausted')
            validate_pairs(result[j]);assert mask[result[j]].all();ok[j]=True;reasons.append('')
        except ValueError as ex:reasons.append(str(ex))
    return result,ok,reasons
def test_sampler():
    for pid,mask in [('SYNTHETIC',np.ones(42,bool)),('SYNTHETIC2',np.array([i not in [4,16,25] for i in range(42)]))]:
        sch,ok,_=schedule_mask(mask,'followup-test',pid)
        for j,g in enumerate(REGIMES):
            if not ok[j]:continue
            prepared=original_sampling.candidates_c(mask) if g=='C' else original_sampling.candidates_f5(mask) if g=='F5' else None
            for b in range(200):np.testing.assert_array_equal(sch[j,b],original_sampling.draw_pair(mask,'followup-test',pid,g,b,prepared))
    log('PASS: accelerated legal sampler reproduces original 200-draw schedules exactly on full and gapped synthetic supports')
