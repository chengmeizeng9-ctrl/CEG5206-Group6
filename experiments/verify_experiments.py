#!/usr/bin/env python3
"""Audit retained experiment data independently of the plotting pipeline."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


R0 = 'R0_scalar_full_sort'
R1 = 'R1_avx2_topk_heap'
R2 = 'R2_cuda_materialized_full_sort'
R3 = 'R3_cuda_fused_tiled_bitonic'
GRID = [(100000,64,16,10), (10000,64,16,10), (100000,32,16,10),
        (100000,128,16,10), (100000,64,1,10), (100000,64,64,10),
        (100000,64,16,5), (100000,64,16,32)]


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def quantile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), (a,b)


def grouped(data, keys):
    groups = defaultdict(list)
    for row in data:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def audit(root):
    if not __debug__:
        raise RuntimeError('Run the data audit without Python -O or PYTHONOPTIMIZE; its checks must stay enabled.')
    env = json.loads((root/'environment.json').read_text())
    assert not env['smoke'] and env['warmups'] > 0 and env['repeats'] > 0
    repeats = env['repeats']
    assert env['cuda_architecture']==61
    manifest = rows(root/'cpu_core/inputs.csv')
    assert [tuple(int(row[k]) for k in ('N','d','q','k')) for row in manifest]==GRID
    assert [row['case'] for row in manifest]==list(map(str,range(8)))
    hashes = json.loads((root/'cpu_core/input_sha256.json').read_text())
    assert len(hashes)==24
    for row, (n,d,q,k) in zip(manifest, GRID):
        for suffix, size in (('.X.f32',n*d*4),('.Q.f32',q*d*4),('.labels.i32',n*4)):
            name=row['prefix']+suffix
            path=root/'cpu_core'/name
            assert path.stat().st_size==size
            assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name], name
    for case in (4,5,6,7):
        assert hashes[f'input_{case}.X.f32']==hashes['input_0.X.f32']
    for case in (6,7):
        assert hashes[f'input_{case}.Q.f32']==hashes['input_0.Q.f32']

    cpu_correct=rows(root/'cpu_core/correctness.csv')
    assert len(cpu_correct)==16
    assert {(r['case'],r['route']) for r in cpu_correct}=={
        (str(c),route) for c in range(8) for route in (R0,R1)}
    for row in cpu_correct:
        for field in ('structure','labels','sorted','fixed_distance_selection',
                      'distance_tolerance','deterministic'):
            assert row[field]=='1', (field,row)
        assert (row['FP64_ID_agreement']=='1')==(int(row['ID_mismatches'])==0)
        shape=GRID[int(row['case'])]
        assert 0 <= int(row['ID_mismatches']) <= shape[2]*shape[3]
    cpu_raw=rows(root/'cpu_core/raw.csv')
    cpu_summary=rows(root/'cpu_core/summary.csv')
    cpu_groups=grouped(cpu_raw,('case','route','group'))
    assert len(cpu_groups)==len(cpu_summary)
    for row in cpu_summary:
        samples=cpu_groups[(row['case'],row['route'],row['group'])]
        assert sorted(int(r['round']) for r in samples)==list(range(20))
        values=[float(r['seconds']) for r in samples]
        assert all(math.isfinite(v) and v>0 for v in values)
        for field,p in (('median',.5),('Q1',.25),('Q3',.75)):
            close(row[field],quantile(values,p))
        close(row['IQR'],quantile(values,.75)-quantile(values,.25))
        close(row['queries_per_second'],int(row['q'])/statistics.median(values))
        assert int(row['unstable'])==int(float(row['IQR'])/float(row['median'])>.1)

    gpu_groups={}
    gpu_correct=[]
    mapping=env['primary_mapping']
    tile=str(env['primary_tile'])
    for index,directory in enumerate(('gpu_core','gpu_mapping_ablation','gpu_tile_ablation')):
        source=f'{index}:{directory}'
        data=rows(root/directory/'raw.csv')
        correctness=rows(root/directory/'correctness.csv')
        memory=rows(root/directory/'memory.csv')
        keys=('case','route','mapping','tile_size')
        expected=({(str(c),route,mapping,tile if route==R3 else '')
                   for c in range(8) for route in (R2,R3)} if index==0 else
                  {('0',route,m,tile if route==R3 else '')
                   for m in ('thread_per_vector','warp_per_vector') for route in (R2,R3)}
                  if index==1 else {('0',R2,mapping,'')} |
                  {('0',R3,mapping,str(t)) for t in (32,64,128,256)})
        assert {tuple(r[k] for k in keys) for r in correctness}==expected
        assert len(correctness)==len(memory)==len(expected)
        assert {tuple(r[k] for k in keys) for r in memory}==expected
        mem={tuple(r[k] for k in keys):r for r in memory}
        for row in correctness:
            for field in ('structure','labels','sorted','distance_tolerance',
                          'deterministic','materialized_match','passed'):
                assert row[field]=='1', (directory,field,row)
            assert (row['FP64_ID_agreement']=='1')==(int(row['ID_mismatches'])==0)
            shape=GRID[int(row['case'])]
            assert 0 <= int(row['ID_mismatches']) <= shape[2]*shape[3]
        gpu_correct.extend(correctness)
        metadata=dict(line.split('=',1) for line in (root/directory/'run.txt').read_text().splitlines())
        assert int(metadata['warmups'])==env['warmups'] and int(metadata['repeats'])==repeats
        assert metadata['compute_capability']=='6.1' and metadata['fmad']=='false'
        groups=grouped(data,keys+('mode',))
        assert set(groups)=={key+(mode,) for key in expected for mode in ('resident','end_to_end')}
        for key,samples in groups.items():
            assert sorted(int(r['round']) for r in samples)==list(range(repeats))
            gpu_groups[(source,)+key]=samples
            n,d,q,k=GRID[int(key[0])]
            m=mem[key[:-1]]
            assert int(m['input_bytes'])==4*(n*d+q*d+n)
            assert int(m['output_bytes'])==12*q*k
            if key[1]==R2:
                aux=20*q*n+4*(q+1)+int(m['cub_temp_bytes'])
                assert int(m['distance_buffer_bytes'])==4*q*n
                assert int(m['key_buffer_bytes'])==16*q*n
                assert int(m['merge_levels'])==0
            else:
                t=int(key[3]); count=(n+t-1)//t
                aux=16*q*count*k
                assert int(m['candidate_buffer_bytes'])==aux
                assert int(m['merge_levels'])==(count-1).bit_length()
                assert int(m['dynamic_shared_bytes'])==8*max(t,2*(1<<(k-1).bit_length()))
            assert int(m['aux_bytes'])==aux
            assert int(m['peak_device_bytes'])==aux+int(m['input_bytes'])+int(m['output_bytes'])
            for row in samples:
                assert tuple(int(row[f]) for f in ('N','d','q','k'))==(n,d,q,k)
                for f in ('aux_bytes','peak_device_bytes','merge_levels'):
                    assert row[f]==m[f]
                duration=float(row['complete_ms'])
                assert math.isfinite(duration) and duration>0
                close(duration,row['device_ms'] if key[-1]=='resident' else row['end_to_end_ms'])
                stages=('distance_ms','pack_ms','full_sort_ms','gather_ms') if key[1]==R2 else ('local_selection_ms','merge_ms','gather_ms')
                assert all(float(row[f])>=0 for f in stages)
                assert sum(float(row[f]) for f in stages)<=float(row['device_ms'])+.01
                if key[-1]=='end_to_end':
                    assert float(row['h2d_ms'])>0 and float(row['d2h_ms'])>0
                    assert duration>=float(row['device_ms'])-.01

    summaries=rows(root/'analysis/gpu_summary.csv')
    assert len(summaries)==len(gpu_groups)==50
    noisy=[]
    for row in summaries:
        key=tuple(row[k] for k in ('source','case','route','mapping','tile_size','mode'))
        values=[float(r['complete_ms']) for r in gpu_groups[key]]
        assert int(row['samples'])==repeats
        for field,p in (('median_ms',.5),('Q1_ms',.25),('Q3_ms',.75)):
            close(row[field],quantile(values,p))
        iqr=quantile(values,.75)-quantile(values,.25)
        close(row['IQR_ms'],iqr)
        close(row['IQR_over_median'],iqr/statistics.median(values))
        close(row['queries_per_second'],int(row['q'])/(statistics.median(values)/1000))
        for field in ('device_ms','end_to_end_ms','h2d_ms','distance_ms',
                      'pack_ms','local_selection_ms','full_sort_ms',
                      'merge_ms','gather_ms','d2h_ms'):
            stage=[float(r[field]) for r in gpu_groups[key] if r[field]]
            if stage:
                close(row['median_'+field],statistics.median(stage))
            else:
                assert math.isnan(float(row['median_'+field]))
        unstable=int(iqr/statistics.median(values)>.1)
        assert int(row['unstable'])==unstable
        if unstable:
            noisy.append(dict(zip(('source','case','route','mapping','tile_size','mode'),key)))
    medians={m:statistics.median(float(r['complete_ms']) for r in gpu_groups[
        ('1:gpu_mapping_ablation','0',R3,m,tile,'resident')])
        for m in ('thread_per_vector','warp_per_vector')}
    fastest_mapping = min(medians,key=medians.get)
    if env.get('primary_mapping_selection', 'auto') == 'auto':
        assert mapping==fastest_mapping
    analysis=json.loads((root/'analysis/analysis.json').read_text())
    assert analysis['gpu_raw_measurements']==50*repeats
    assert analysis['all_required_correctness_passed']
    assert analysis['cpu_correctness_records']==len(cpu_correct)
    assert analysis['gpu_correctness_records']==len(gpu_correct)==25
    assert analysis['variable_gpu_groups']==noisy
    unified=rows(root/'analysis/unified_summary.csv')
    assert len(unified)==len(cpu_summary)+len(summaries)
    assert len(rows(root/'analysis/unified_correctness.csv'))==41
    for row in unified:
        if row['backend']=='CUDA':
            match=next(r for r in summaries if all(r[k]==row[k] for k in
                ('source','case','route','mapping','tile_size','mode')))
            for field in ('median_ms','Q1_ms','Q3_ms','IQR_ms','queries_per_second'):
                close(row[field],match[field])
        else:
            match=next(r for r in cpu_summary if all(r[k]==row[k] for k in ('case','route','group')))
            for field,cpu_field in (('median_ms','median'),('Q1_ms','Q1'),('Q3_ms','Q3'),('IQR_ms','IQR')):
                close(row[field],1000*float(match[cpu_field]))
    def baseline(route, mode, source):
        return float(next(r['median_ms'] for r in unified if r['case']=='0'
            and r['route']==route and r['mode']==mode and r['source']==source
            and r['group']=='0'))
    b={'R0_ms':baseline(R0,'host_complete','cpu'),
       'R1_ms':baseline(R1,'host_complete','cpu'),
       'R2_resident_ms':baseline(R2,'resident','0:gpu_core'),
       'R3_resident_ms':baseline(R3,'resident','0:gpu_core'),
       'R2_end_to_end_ms':baseline(R2,'end_to_end','0:gpu_core'),
       'R3_end_to_end_ms':baseline(R3,'end_to_end','0:gpu_core')}
    for field,value in b.items():
        close(analysis['baseline'][field],value)
    for field,numerator,denominator in (
            ('R0_over_R1','R0_ms','R1_ms'),
            ('R0_over_R2_resident','R0_ms','R2_resident_ms'),
            ('R0_over_R3_resident','R0_ms','R3_resident_ms'),
            ('R2_over_R3_resident','R2_resident_ms','R3_resident_ms'),
            ('R2_over_R3_end_to_end','R2_end_to_end_ms','R3_end_to_end_ms')):
        close(analysis['speedups'][field],b[numerator]/b[denominator])
    report=root/'analysis/final_results.md'
    for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report.read_text()):
        assert (report.parent/name).is_file(), name
    assert len(list(report.parent.glob('*.png')))==8
    cpu_noisy=[{k:r[k] for k in ('case','route','group','median','IQR')}
               for r in cpu_summary if r['unstable']=='1']
    return {'status':'PASS','scope':'retained full-run data audit; execution logs are separate',
            'input_files_verified':24,'core_cases':8,'cpu_correctness_records':16,
            'gpu_correctness_records':25,
            'FP64_ID_mismatches':sum(int(r['ID_mismatches']) for r in cpu_correct+gpu_correct),
            'cpu_timing_samples':len(cpu_raw),'gpu_timing_samples':50*repeats,
            'gpu_timing_groups':50,'primary_mapping':mapping,
            'primary_mapping_is_fastest_this_session':mapping==fastest_mapping,
            'mapping_medians_ms':medians,'gpu_variable_groups':noisy,
            'cpu_variable_groups':cpu_noisy,'plots':8,
            'source_commit':env['git_commit'],
            'artifact_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(set(root.rglob('*.csv')) | set(root.glob('*/run.txt')) |
                                {root/'environment.json', root/'cpu_core/input_sha256.json',
                                 root/'analysis/analysis.json'})}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/full'))
    parser.add_argument('--output',type=Path,default=Path('results/verification/data_audit.json'))
    args=parser.parse_args()
    result=audit(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
