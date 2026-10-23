#!/usr/bin/env python3
"""Validate retained CPU records, hash inputs, and create required-route plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def hash_inputs(directory):
    hashes = {}
    for path in sorted(directory.glob('*.f32')) + sorted(directory.glob('*.i32')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    (directory / 'input_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return hashes


def plot_results(directory, summary, inputs):
    baseline = [row for row in summary if row['case'] == '0' and row['group'] == '0']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([row['route'].split('_')[0] for row in baseline],
           [1000 * float(row['median']) for row in baseline],
           yerr=[[1000 * (float(row['median']) - float(row['Q1'])) for row in baseline],
                 [1000 * (float(row['Q3']) - float(row['median'])) for row in baseline]],
           capsize=4)
    ax.set(ylabel='Median complete batch time (ms)',
           title='Scalar full sort and AVX2 Top-K heap')
    fig.tight_layout()
    fig.savefig(directory / 'cpu_ablation.png', dpi=160)
    plt.close(fig)

    factors = [('N', ['1', '0']), ('d', ['2', '0', '3']),
               ('q', ['4', '0', '5']), ('k', ['6', '0', '7'])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    shape_by_case = {row['case']: row for row in inputs}
    for axis, (factor, cases) in zip(axes.flat, factors):
        for route in ('R0_scalar_full_sort', 'R1_avx2_topk_heap'):
            points = []
            for case in cases:
                match = next((row for row in summary
                              if row['case'] == case and row['group'] == '0'
                              and row['route'] == route), None)
                if match:
                    points.append((int(shape_by_case[case][factor]),
                                   float(match['queries_per_second']) if factor == 'q'
                                   else 1000 * float(match['median'])))
            points.sort()
            if points:
                axis.plot(*zip(*points), marker='o', label=route.split('_')[0])
        axis.set(xlabel=factor,
                 ylabel='Queries / second' if factor == 'q' else 'Batch time (ms)')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(directory / 'parameter_sweeps.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='?', type=Path, default=ROOT / 'results/main')
    args = parser.parse_args()
    directory = args.results
    hashes = hash_inputs(directory)
    input_rows = rows(directory / 'inputs.csv')
    raw = rows(directory / 'raw.csv')
    summary = rows(directory / 'summary.csv')
    correctness = rows(directory / 'correctness.csv')
    expected_routes = {'R0_scalar_full_sort', 'R1_avx2_topk_heap'}
    assert {row['route'] for row in summary} == expected_routes
    assert {row['route'] for row in correctness} == expected_routes
    assert all(all(row[field] == '1' for field in
                   ('structure', 'labels', 'sorted', 'fixed_distance_selection',
                    'distance_tolerance', 'deterministic'))
               for row in correctness)
    for record in summary:
        samples = [float(row['seconds']) for row in raw
                   if (row['case'], row['route'], row['group']) ==
                   (record['case'], record['route'], record['group'])]
        assert len(samples) == 20
        q1, median, q3 = np.quantile(samples, [.25, .5, .75], method='linear')
        assert np.allclose([q1, median, q3],
                           [float(record[key]) for key in ('Q1', 'median', 'Q3')],
                           rtol=1e-9, atol=1e-12)
    plot_results(directory, summary, input_rows)
    first_group = [row for row in summary
                   if row['case'] == '0' and row['group'] == '0']
    base = {row['route']: row for row in first_group}
    report = {
        'input_files': len(hashes),
        'configurations': len(input_rows),
        'timing_records': len(raw),
        'correctness_records': len(correctness),
        'required_correctness_checks_pass': True,
        'FP64_ID_position_differences': sum(int(row['ID_mismatches'])
                                             for row in correctness),
        'maximum_absolute_distance_error': max(float(row['max_abs_error'])
                                               for row in correctness),
        'baseline_median_seconds': {route: float(row['median'])
                                    for route, row in base.items()},
        'baseline_speedup_scalar_over_AVX2_heap':
            float(base['R0_scalar_full_sort']['median']) /
            float(base['R1_avx2_topk_heap']['median']),
        'variable_groups': [
            {'case': row['case'], 'route': row['route'], 'group': row['group']}
            for row in summary if row['unstable'] == '1'],
    }
    (directory / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
