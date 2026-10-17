#!/usr/bin/env python3
"""Our independent CPU/GPU result verifier. No GPU timing is simulated.
Input: <prefix>.X.f32/.Q.f32/.labels.i32 (little endian).
Output: <output>.ids.u32/.distances.f32/.labels.i32; optional full q*N table.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def read(prefix, suffix, dtype, count):
    values=np.fromfile(str(prefix)+suffix,dtype=dtype)
    if values.size!=count:
        raise ValueError(f'{prefix}{suffix}: length {values.size}, expected {count}')
    return values


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--n',required=True,type=int)
    ap.add_argument('--d',required=True,type=int)
    ap.add_argument('--q',required=True,type=int)
    ap.add_argument('--k',required=True,type=int)
    ap.add_argument('--full-distances',type=Path)
    ap.add_argument('--repeat-output',type=Path)
    ap.add_argument('--report',required=True,type=Path)
    a=ap.parse_args()
    if min(a.n,a.d,a.q,a.k)<1 or a.k>a.n:
        raise ValueError('invalid shape or k')
    x=read(a.input,'.X.f32','<f4',a.n*a.d).reshape(a.n,a.d)
    query=read(a.input,'.Q.f32','<f4',a.q*a.d).reshape(a.q,a.d)
    labels=read(a.input,'.labels.i32','<i4',a.n)
    for array in (x,query):
        if not np.all(np.isfinite(array)) or np.any(np.abs(array)>1):
            raise ValueError('input outside finite [-1,1]')
    ids=read(a.output,'.ids.u32','<u4',a.q*a.k).reshape(a.q,a.k)
    distances=read(a.output,'.distances.f32','<f4',a.q*a.k).reshape(a.q,a.k)
    result_labels=read(a.output,'.labels.i32','<i4',a.q*a.k).reshape(a.q,a.k)
    report={'structure':True,'labels':True,'sorted':True,'distance_tolerance':True,
            'fixed_distance_selection':'not_measured','deterministic':'not_measured',
            'FP64_ID_agreement':True,'ID_mismatches':0,'max_abs_error':0,
            'nonfinite_distances':int(np.count_nonzero(~np.isfinite(distances))),
            'boundary_diagnostics':[],'contract':'1.1','atol':1e-5,'rtol':1e-4}
    full=None
    if a.full_distances:
        full=np.fromfile(a.full_distances,dtype='<f4')
        if full.size!=a.q*a.n:
            raise ValueError('full distance table length')
        full=full.reshape(a.q,a.n)
        report['fixed_distance_selection']=bool(np.all(np.isfinite(full)) and np.all(full>=0))
    for qi in range(a.q):
        # O(N) reference memory, fixed dimension-order FP64 accumulation.
        ref=np.zeros(a.n,dtype=np.float64)
        for dim in range(a.d):
            delta=np.float64(query[qi,dim])-x[:,dim].astype(np.float64)
            ref+=delta*delta
        order=np.lexsort((np.arange(a.n),ref))
        report['structure'] &= bool(np.all(ids[qi]<a.n) and np.unique(ids[qi]).size==a.k and
                                    np.all(np.isfinite(distances[qi])) and np.all(distances[qi]>=0) and
                                    not np.any((distances[qi]==0)&np.signbit(distances[qi])))
        if np.any(ids[qi]>=a.n):
            report['FP64_ID_agreement']=False
            report['labels']=False
            report['distance_tolerance']=False
            if full is not None: report['fixed_distance_selection']=False
            continue
        report['labels'] &= bool(np.array_equal(result_labels[qi],labels[ids[qi]]))
        report['sorted'] &= bool(np.array_equal(np.lexsort((ids[qi],distances[qi])),np.arange(a.k)))
        error=np.abs(distances[qi].astype(np.float64)-ref[ids[qi]])
        report['distance_tolerance'] &= bool(np.all(error<=1e-5+1e-4*np.abs(ref[ids[qi]])))
        report['max_abs_error']=max(report['max_abs_error'],float(error.max()))
        mismatch=int(np.count_nonzero(ids[qi]!=order[:a.k]))
        report['ID_mismatches']+=mismatch
        report['FP64_ID_agreement'] &= mismatch==0
        if full is not None:
            expected=np.lexsort((np.arange(a.n),full[qi]))[:a.k]
            report['fixed_distance_selection'] &= bool(np.array_equal(ids[qi],expected) and np.array_equal(distances[qi],full[qi,ids[qi]]))
            all_error=np.abs(full[qi].astype(np.float64)-ref)
            report['distance_tolerance'] &= bool(np.all(all_error<=1e-5+1e-4*np.abs(ref)))
            report['max_abs_error']=max(report['max_abs_error'],float(all_error.max()))
        if mismatch:
            relevant=sorted(set(map(int,ids[qi]))|set(map(int,order[:a.k])))
            report['boundary_diagnostics'].append({'query':qi,'kth':float(ref[order[a.k-1]]),
                'k_plus_one':float(ref[order[a.k]]) if a.k<a.n else None,
                'reference_ids':order[:a.k].tolist(),'actual_ids':ids[qi].tolist(),
                'candidate_ids':relevant,'FP64_distances':ref[relevant].tolist(),
                'method_distances':full[qi,relevant].tolist() if full is not None else None})
    if a.repeat_output:
        repeat=[read(a.repeat_output,suffix,dtype,a.q*a.k) for suffix,dtype in
                [('.ids.u32','<u4'),('.distances.f32','<f4'),('.labels.i32','<i4')]]
        report['deterministic']=all(np.array_equal(v.ravel().view('u4'),w.view('u4')) for v,w in zip((ids,distances,result_labels),repeat))
    if report['nonfinite_distances'] or (full is not None and not np.all(np.isfinite(full))):
        report['max_abs_error']=None
    report['status']=('strict_agreement' if all(report[key] is True for key in
        ('structure','labels','sorted','distance_tolerance','fixed_distance_selection','deterministic','FP64_ID_agreement'))
        else 'review_required: unexplained differences cannot support speedup claims')
    def finite_json(value):
        if isinstance(value,float) and not math.isfinite(value): return None
        if isinstance(value,list): return [finite_json(v) for v in value]
        if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
        return value
    a.report.write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(a.report)
    return 0 if all(report[key] for key in ('structure','labels','sorted','distance_tolerance')) and report['fixed_distance_selection'] is not False else 1

if __name__=='__main__':
    raise SystemExit(main())
