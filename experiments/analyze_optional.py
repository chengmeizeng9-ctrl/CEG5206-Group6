#!/usr/bin/env python3
"""Audit and report optional routes without mixing them into required results."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

IDENTITY=('case','N','d','q','k','backend','route','mapping','tile_size','chunk_size','buffers')
METRICS=('device_ms','h2d_ms','d2h_ms','distance_ms','pack_ms','full_sort_ms',
 'local_selection_ms','merge_ms','gather_ms','norm_x_ms','norm_q_ms','gemm_ms',
 'reconstruct_ms','radix_ms','scan_ms','compact_ms','pipeline_ms','staging_ms',
 'chunk_selection_ms','compute_service_ms','estimated_overlap_ratio',
 'dimensions_per_candidate','abandoned_candidates','near_zero_rechecks')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:return list(csv.DictReader(stream))


def write_csv(path,records):
    with path.open('w',newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(results):
    raw=read_csv(results/'raw.csv');correctness=read_csv(results/'correctness.csv');memory=read_csv(results/'memory.csv')
    require(raw and correctness and memory,'Optional CSVs must be nonempty')
    metadata=dict(line.split('=',1) for line in (results/'run.txt').read_text(encoding='utf-8').splitlines())
    repeats=int(metadata['repeats']);require(repeats>0,'Invalid repetitions')
    key=lambda row:tuple(row[k] for k in IDENTITY)
    by_correct={key(r):r for r in correctness};by_memory={key(r):r for r in memory}
    require(len(by_correct)==len(correctness) and len(by_memory)==len(memory),'Duplicate configuration evidence')
    require(by_correct.keys()==by_memory.keys(),'Correctness/memory configurations differ')
    cases=set(metadata['cases'].split(','))
    require({r['case'] for r in correctness}==cases,'Missing expected input cases')
    expected={'O5_cuda_stream_single','O5_cuda_stream_double'}
    if metadata.get('stream_only','0')!='1':
        expected|={'R0_scalar_full_sort','R1_avx2_topk_heap','C0_cpu_scalar_topk_heap',
                   'O1_cpu_avx2_tiled_bitonic','O2_cpu_scalar_early_abandon',
                   'R2_cuda_materialized_full_sort','R3_cuda_fused_tiled_bitonic',
                   'O3_cuda_radix_scan_compact','O4_cublas_gemm_reference'}
    for case in cases:
        require({r['route'] for r in correctness if r['case']==case}==expected,
                'Missing expected optional route/control configuration')
    for identity,row in by_correct.items():
        for field in ('structure','labels','sorted','distance_tolerance','deterministic','passed'):
            require(row[field]=='1',f'{identity}: {field} failed')
        if row['fixed_selection_match']:require(row['fixed_selection_match']=='1','Fixed-distance selection failed')
        if row['backend']=='CPU' or row['route'].startswith(('O3_','O4_')):
            require(row['fixed_selection_match']=='1','Missing independent fixed-distance selection check')
        if row['exact_reference_match']:require(row['exact_reference_match']=='1','Arithmetic-matched reference comparison failed')
        require((row['FP64_ID_agreement']=='1')==(int(row['ID_mismatches'])==0),'Inconsistent FP64 ID counters')
        require(0<=int(row['ID_mismatches'])<=int(row['q'])*int(row['k']),'Invalid FP64 ID counter')
        require(float(row['max_abs_error'])>=0,'Invalid numerical error record')
        m=by_memory[identity]
        if row['backend']=='CUDA':
            require(int(m['input_device_bytes'])+int(m['output_device_bytes'])+int(m['aux_bytes'])==int(m['peak_device_bytes']),
                    'Optional device allocations do not add up')
        if row['route']=='O2_cpu_scalar_early_abandon':
            require(0<float(m['dimensions_per_candidate'])<=int(row['d']),'Invalid early-abandon dimension count')
    groups=defaultdict(list)
    for row in raw:
        identity=key(row);require(identity in by_correct,'Timing without correctness evidence')
        require(math.isfinite(float(row['complete_ms'])) and float(row['complete_ms'])>0,'Invalid optional timing')
        if row['route'].startswith('O5_'):
            require(row['mode']=='end_to_end' and not row['device_ms'],'Streaming cannot be GPU-resident')
            require(0<=float(row['estimated_overlap_ratio'])<=1,'Invalid estimated overlap')
            require(float(row['complete_ms'])+.05>=float(row['pipeline_ms']),'Device pipeline exceeds wall time')
        if row['route']=='O3_cuda_radix_scan_compact':
            require(all(float(row[f])>=0 for f in ('radix_ms','scan_ms','compact_ms')),'Missing radix/scan/compact timing')
        if row['route']=='O4_cublas_gemm_reference':
            require(all(float(row[f])>=0 for f in ('norm_x_ms','norm_q_ms','gemm_ms','reconstruct_ms')),'Missing GEMM stage timing')
        if row['backend']=='CUDA':
            for field in ('aux_bytes','peak_device_bytes'):
                require(row[field]==by_memory[identity][field],'Timing/memory evidence differs')
            if not row['route'].startswith('O5_'):
                require(row['device_ms'] and float(row['device_ms'])>0,'Missing device timing')
                if row['mode']=='resident':
                    require(math.isclose(float(row['complete_ms']),float(row['device_ms']),rel_tol=1e-9),
                            'Resident complete/device timing differs')
                else:
                    require(float(row['complete_ms'])+.05>=float(row['device_ms']),
                            'Device timing exceeds transfer-inclusive wall time')
        groups[identity+(row['mode'],)].append(row)
    require({identity[:-1] for identity in groups}==by_correct.keys(),'Configuration missing timings')
    for identity in by_correct:
        row=by_correct[identity]
        modes={key[-1] for key in groups if key[:-1]==identity}
        expected={'host_complete'} if row['backend']=='CPU' else ({'end_to_end'} if row['route'].startswith('O5_') else {'resident','end_to_end'})
        require(modes==expected,'Missing or unsupported timing boundary')
    summaries=[]
    for identity,samples in sorted(groups.items()):
        require(sorted(int(r['round']) for r in samples)==list(range(repeats)),'Missing or repeated rounds')
        values=[float(r['complete_ms']) for r in samples];median=statistics.median(values)
        q1,q3=quantile(values,.25),quantile(values,.75)
        row=dict(zip(IDENTITY+('mode',),identity))
        row.update(samples=len(samples),median_ms=median,Q1_ms=q1,Q3_ms=q3,IQR_ms=q3-q1,
                   IQR_over_median=(q3-q1)/median,unstable=int((q3-q1)/median>.1),
                   queries_per_second=int(row['q'])/(median/1000),
                   aux_bytes=by_memory[identity[:-1]]['aux_bytes'],
                   peak_device_bytes=by_memory[identity[:-1]]['peak_device_bytes'],
                   pinned_host_bytes=by_memory[identity[:-1]]['pinned_host_bytes'])
        for metric in METRICS:
            values=[float(r[metric]) for r in samples if r.get(metric)]
            require(all(math.isfinite(v) and v>=0 for v in values),'Invalid optional stage/diagnostic value')
            row['median_'+metric]=statistics.median(values) if values else ''
        summaries.append(row)
    report={'all_correctness_passed':True,'configurations':len(correctness),'timing_groups':len(summaries),
            'raw_measurements':len(raw),'repeats':repeats,
            'FP64_ID_position_differences':sum(int(r['ID_mismatches']) for r in correctness),
            'variable_groups':[{k:r[k] for k in IDENTITY+('mode','IQR_over_median')} for r in summaries if r['unstable']],
            'artifact_sha256':{name:hashlib.sha256((results/name).read_bytes()).hexdigest()
                               for name in ('raw.csv','correctness.csv','memory.csv','run.txt')}}
    return summaries,memory,report


def plots(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=[]
    core=[r for r in summary if r['case']=='0' and r['mode'] in ('host_complete','end_to_end')]
    if core:
        fig,ax=plt.subplots(figsize=(11,5));labels=[r['route']+'\n'+r['mode'] for r in core]
        ax.bar(range(len(core)),[r['median_ms'] for r in core],
               yerr=[[r['median_ms']-r['Q1_ms'] for r in core],[r['Q3_ms']-r['median_ms'] for r in core]],capsize=3)
        ax.set_xticks(range(len(core)),labels,rotation=40,ha='right');ax.set_ylabel('Complete batch latency (ms)')
        fig.tight_layout();fig.savefig(output/'optional_baseline.png',dpi=160);plt.close(fig);outputs.append('optional_baseline.png')
    early=[r for r in summary if r['route']=='O2_cpu_scalar_early_abandon']
    if early:
        fig,ax=plt.subplots(figsize=(8,4));x=list(range(len(early)))
        ax.bar(x,[r['median_dimensions_per_candidate'] for r in early],label='Dimensions evaluated')
        ax.plot(x,[int(r['d']) for r in early],'o--',label='Full distance')
        ax.set_xticks(x,[r['case'] for r in early]);ax.set(xlabel='Case',ylabel='Mean dimensions per candidate')
        ax.legend();fig.tight_layout();fig.savefig(output/'early_abandon_dimensions.png',dpi=160);plt.close(fig);outputs.append('early_abandon_dimensions.png')
    gpu=[r for r in summary if r['case']=='0' and r['backend']=='CUDA' and r['mode']=='resident']
    if gpu:
        fig,ax=plt.subplots(figsize=(10,4.5));bottom=[0.0]*len(gpu)
        fields=[('distance_ms','Direct distance'),('norm_q_ms','Query norms'),('gemm_ms','GEMM'),
                ('reconstruct_ms','Reconstruction/rechecks'),('pack_ms','Packing'),('full_sort_ms','Full sort'),
                ('radix_ms','Radix refinement'),('scan_ms','Flags/scan'),('compact_ms','Compaction'),
                ('local_selection_ms','Fused local / retained-k sort'),('merge_ms','Merge'),('gather_ms','Gather')]
        for field,label in fields:
            values=[float(r['median_'+field] or 0) for r in gpu]
            if any(values):
                ax.bar(range(len(gpu)),values,bottom=bottom,label=label)
                bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_xticks(range(len(gpu)),[r['route'] for r in gpu],rotation=30,ha='right')
        ax.set_ylabel('Separately computed stage medians (ms)');ax.legend(fontsize=8,ncol=2)
        fig.tight_layout();fig.savefig(output/'optional_gpu_stages.png',dpi=160);plt.close(fig);outputs.append('optional_gpu_stages.png')
    streamed=[r for r in summary if r['route'].startswith('O5_')]
    if streamed:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for route,label in [('O5_cuda_stream_single','Single buffer'),('O5_cuda_stream_double','Double buffer')]:
            selected=sorted([r for r in streamed if r['route']==route],key=lambda r:int(r['case']))
            axes[0].plot([r['case'] for r in selected],[r['median_ms'] for r in selected],'o-',label=label)
            axes[1].plot([r['case'] for r in selected],[int(r['peak_device_bytes'])/(1024**2) for r in selected],'o-',label=label)
        axes[0].set(xlabel='Case',ylabel='End-to-end latency (ms)');axes[1].set(xlabel='Case',ylabel='Allocated device buffers (MiB)')
        axes[0].legend();fig.tight_layout();fig.savefig(output/'streaming_comparison.png',dpi=160);plt.close(fig);outputs.append('streaming_comparison.png')
    return outputs


def controlled_ratios(summary):
    pairs=[('O1_cpu_avx2_tiled_bitonic','R1_avx2_topk_heap'),
           ('O2_cpu_scalar_early_abandon','C0_cpu_scalar_topk_heap'),
           ('O3_cuda_radix_scan_compact','R2_cuda_materialized_full_sort'),
           ('O4_cublas_gemm_reference','R2_cuda_materialized_full_sort'),
           ('O5_cuda_stream_double','O5_cuda_stream_single')]
    ratios=[]
    for optional,control in pairs:
        for row in (r for r in summary if r['route']==optional):
            matches=[r for r in summary if r['route']==control and r['case']==row['case'] and r['mode']==row['mode']]
            if not matches:continue
            reference=matches[0]
            ratios.append({'case':row['case'],'optional_route':optional,'control_route':control,
                           'mode':row['mode'],'control_ms':reference['median_ms'],
                           'optional_ms':row['median_ms'],'control_over_optional':reference['median_ms']/row['median_ms'],
                           'noisy':bool(reference['unstable'] or row['unstable'])})
    return ratios


def generate(results,output,make_plots=True):
    summary,memory,report=audit(results);output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'optional_summary.csv',summary)
    figure_names=plots(output,summary) if make_plots else []
    ratios=controlled_ratios(summary)
    if ratios:write_csv(output/'optional_controlled_speedups.csv',ratios)
    report['controlled_ratios']=ratios
    report['plots']=figure_names
    (output/'optional_analysis.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    lines=['# Optional kNN routes', '',
      'All recorded configurations passed structural, numerical, deterministic and applicable selection/reference checks.',
      f"FP64 ID-position differences: **{report['FP64_ID_position_differences']}** (reported separately from exact FP32 selection).", '',
      '| Case | Route | Boundary | Median (ms) | IQR/median | Aux/device CPU buffers (MiB) |',
      '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['case']} | {r['route']} | {r['mode']} | {r['median_ms']:.3f} | "
                     f"{r['IQR_over_median']:.1%}{' flagged' if r['unstable'] else ''} | {int(r['aux_bytes'])/1024**2:.3f} |")
    lines+=['', '## Controlled comparisons', '',
      'The optional benchmark includes fresh R0/R1 CPU and R2/R3 GPU controls plus a scalar full-evaluation heap control unless stream-only mode was requested. '
      'Use these controls for comparisons within this run, rather than mixing measurements with older required-run CSVs.', '',
      '| Case | Optional route | Control | Boundary | Control / optional | Noise flag |',
      '|---|---|---|---|---:|---|']
    for row in ratios:
        lines.append(f"| {row['case']} | {row['optional_route']} | {row['control_route']} | {row['mode']} | "
                     f"{row['control_over_optional']:.3f}x | {'flagged' if row['noisy'] else ''} |")
    lines+=['', 'Ratios above one favour the optional route; ratios below one are retained as measured losses. '
      'These are within-run ratios of configuration medians, not paired-sample confidence intervals.', '',
      'CPU Bitonic uses the same AVX2 distance arithmetic as R1; early abandonment uses exactly R0 scalar arithmetic. '
      'Its timing control is C0, a scalar heap with full distance evaluation using the same heap path, so heap selection is held fixed. '
      'The early-abandon dimension counter is collected outside timing. Strict partial-sum > threshold pruning preserves ties.', '',
      'Radix selection refines a deterministic 64-bit threshold over eight 8-bit histogram passes, marks qualifying keys, '
      'uses a self-written hierarchical exclusive Blelloch scan and compacts exactly k keys per query. '
      'Only the retained k keys are finally Bitonic sorted; there is no full-N sort in this route.', '',
      'GEMM is a labelled cuBLAS FP32 reference, with query norms, GEMM, reconstruction, packing, full sorting and gathering timed. '
      'Resident calls reuse precomputed database norms; end-to-end calls upload X and recompute its norms. '
      'Database-norm setup and cuBLAS workspace allocation are disclosed in memory.csv. '
      'Near-zero reconstructed distances are re-evaluated directly to guard cancellation, and the recheck count is recorded. '
      'GEMM is validated against an independent full sort of its own reconstructed distances; bitwise direct/GEMM equality is not assumed.', '',
      'Streaming uses bounded pinned host staging and one or two reusable device slots. One compute stream serialises '
      'chunk selection and global merging; separate copy streams let the next upload overlap the preceding computation when supported. '
      'Candidate IDs retain global offsets and labels are gathered on the host inside end-to-end timing. '
      'Streaming always transfers chunks, so it has no resident-only latency. The service-time overlap ratio is an estimate, not a profiler trace.', '',
      '## Streaming storage and overlap', '',
      '| Case | Buffers | Chunk rows | Logical inputs (MiB) | Device buffers (MiB) | Pinned host (MiB) | Estimated overlap |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['route'].startswith('O5_'):
            m=next(m for m in memory if all(m[k]==r[k] for k in IDENTITY))
            lines.append(f"| {r['case']} | {r['buffers']} | {r['chunk_size']} | "
                         f"{int(m['logical_input_bytes'])/1024**2:.3f} | {int(r['peak_device_bytes'])/1024**2:.3f} | "
                         f"{int(r['pinned_host_bytes'])/1024**2:.3f} | {r['median_estimated_overlap_ratio']:.1%} |")
    lines+=['', '## Interpretation', '',
      'A small chunk-budget experiment validates bounded-memory processing; it is not evidence that the tested database exceeded '
      'physical GPU memory unless its recorded size actually did. Double buffering can lose on small inputs. '
      'The host input must fit host RAM. Actual device buffer allocation is not a DRAM-traffic measurement. '
      'All noisy groups remain retained; session repeatability and generalisation require additional evidence.', '']
    for figure in figure_names:lines.extend([f'![{figure}]({figure})',''])
    (output/'optional_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=generate(args.results,args.output or args.results/'analysis')
    print(json.dumps(report,indent=2))
