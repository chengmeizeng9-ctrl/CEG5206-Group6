#!/usr/bin/env python3
"""Hash exported inputs, independently check fixtures, and plot recorded measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    hashes = {}
    for p in sorted(root.glob('*.f32')) + sorted(root.glob('*.i32')):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[p.name] = h.hexdigest()
    (root / 'input_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    fixture_diagnostics = []
    for p in root.parent.glob('fixture_*.txt'):
        n, d, q = map(int, p.read_text().split())
        prefix = p.with_suffix('')
        x = np.fromfile(str(prefix)+'.X.f32', dtype='<f4').reshape(n, d)
        queries = np.fromfile(str(prefix)+'.Q.f32', dtype='<f4').reshape(q, d)
        labels = np.fromfile(str(prefix)+'.labels.i32', dtype='<i4')
        dist = ((queries.astype(np.float64)[:, None, :]-x.astype(np.float64)[None, :, :])**2).sum(axis=2)
        golden = rows(Path(str(prefix)+'.oracle.csv'))
        for record in golden:
            qi, rank, sample = (int(record[key]) for key in ('query', 'rank', 'id'))
            expected = np.lexsort((np.arange(n), dist[qi]))
            assert sample == expected[rank]
            assert np.isclose(float(record['distance']), dist[qi, sample], atol=1e-14, rtol=1e-14)
            assert int(record['label']) == labels[sample]
        # Emulate the scalar FP32 order without changing the original input.
        scalar = np.zeros((q, n), dtype=np.float32)
        for dim in range(d):
            delta = queries[:, None, dim]-x[None, :, dim]
            scalar = np.float32(scalar + np.float32(delta*delta))
        fixture_diagnostics.append({'fixture': p.stem,
                                    'FP64_distances': dist.tolist(),
                                    'scalar_FP32_distances': scalar.tolist(),
                                    'FP64_order': [np.lexsort((np.arange(n), row)).tolist() for row in dist],
                                    'scalar_order': [np.lexsort((np.arange(n), row)).tolist() for row in scalar]})
    (root / 'fixture_diagnostics.json').write_text(json.dumps(fixture_diagnostics, indent=2)+'\n')
    input_shapes={r['case']:r for r in rows(root/'inputs.csv')}
    summary = rows(root / 'summary.csv')
    first = [r for r in summary if r['group']=='0']
    correctness = rows(root / 'correctness.csv')
    assert all(all(r[k]=='1' for k in ('structure','labels','sorted','fixed_distance_selection','distance_tolerance','deterministic')) for r in correctness)
    raw = rows(root / 'raw.csv')
    for row in summary:
        sample = [float(r['seconds']) for r in raw if (r['case'],r['route'],r['group']) == (row['case'],row['route'],row['group'])]
        assert len(sample)==20
        q1, median, q3 = np.quantile(sample, [.25,.5,.75], method='linear')
        assert np.allclose([q1,median,q3], [float(row[k]) for k in ('Q1','median','Q3')], rtol=1e-9, atol=1e-12)
    baseline = next(r for r in first if r['case']=='0' and r['route'].startswith('R0'))
    base = [r for r in first if r['case']=='0']
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar([r['route'].split('_')[0] for r in base], [1000*float(r['median']) for r in base],
           yerr=[[1000*(float(r['median'])-float(r['Q1'])) for r in base],
                 [1000*(float(r['Q3'])-float(r['median'])) for r in base]], capsize=3)
    ax.set(ylabel='Median complete batch time (ms)', title='CPU ablation: N=100000, d=64, q=16, k=10, T=128')
    fig.tight_layout(); fig.savefig(root/'cpu_ablation.png', dpi=160); plt.close(fig)
    # Single-factor panels; R0 is deliberately reused for the T sweep.
    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    factors = [('N',['1','0']),('d',['2','0','3']),('q',['4','0','5']),('k',['6','0','7']),('T',['8','9','0','10'])]
    for ax, (factor, cases) in zip(axes.flat, factors):
        for route in sorted({r['route'] for r in first}):
            points = []
            for case in cases:
                match = [r for r in first if r['case']==case and r['route']==route]
                if match:
                    row=match[0]
                    points.append((int(input_shapes[case]['T'] if factor=='T' and row[factor]=='NA' else row[factor]), float(row['queries_per_second']) if factor=='q' else 1000*float(row['median'])))
                elif factor=='T' and route.startswith('R0'):
                    case_rows=[r for r in first if r['case']==case]
                    if case_rows: points.append((int(case_rows[0]['T']),1000*float(baseline['median'])))
            if points:
                points.sort(); ax.plot(*zip(*points),marker='o',label=route.split('_')[0])
        ax.set(xlabel=factor,ylabel='Queries / second' if factor=='q' else 'Batch time (ms)')
    axes.flat[-1].axis('off'); axes.flat[0].legend(ncol=4, fontsize=8)
    fig.tight_layout();fig.savefig(root/'parameter_sweeps.png',dpi=160);plt.close(fig)
    micro=rows(root/'selection_raw.csv')
    micro_summary=[]
    for case in sorted({r['case'] for r in micro},key=int):
        for route in ('0','1','2'):
            records=[r for r in micro if r['case']==case and r['selection']==route]
            if not records: continue
            samples=[float(r['seconds']) for r in records]
            q1,med,q3=np.quantile(samples,[.25,.5,.75])
            micro_summary.append({'case':case,'N':records[0]['N'],'k':records[0]['k'],'T':records[0]['T'],
                'selection':('sort','heap','bitonic')[int(route)],'median':float(med),'IQR':float(q3-q1),
                'unstable':bool((q3-q1)/med>.1)})
    with (root/'selection_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(micro_summary[0]));writer.writeheader();writer.writerows(micro_summary)
    fig,ax=plt.subplots(figsize=(7,5))
    for method in ('sort','heap','bitonic'):
        chosen=[r for r in micro_summary if r['case'] in ('6','0','7') and r['selection']==method]
        chosen.sort(key=lambda r:int(r['k']))
        ax.plot([int(r['k']) for r in chosen],[1000*r['median'] for r in chosen],marker='o',label=method)
    ax.set(xlabel='k',ylabel='Fixed-distance selection time (ms)',yscale='log',title='Selection diagnostic: N=100000, T=128')
    ax.legend();fig.tight_layout();fig.savefig(root/'selection_vs_k.png',dpi=160);plt.close(fig)
    report = {'input_files':len(hashes),'correctness_rows':len(correctness),
              'all_required_checks_pass':True,
              'FP64_ID_mismatches':sum(int(r['ID_mismatches']) for r in correctness),
              'noisy_groups':[{'case':r['case'],'route':r['route'],'group':r['group']} for r in summary if r['unstable']=='1'],
              'baseline_median_seconds':float(baseline['median']),
              'base_case':[{'route':r['route'],'median_seconds':float(r['median']),
                            'speedup_vs_R0':float(baseline['median'])/float(r['median']),
                            'aux_bytes':int(r['aux_bytes'])} for r in base]}
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
