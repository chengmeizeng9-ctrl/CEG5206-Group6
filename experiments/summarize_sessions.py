#!/usr/bin/env python3
"""Summarise separate full sessions without pooling their timing samples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

R0, R1 = 'R0_scalar_full_sort', 'R1_avx2_topk_heap'
R2, R3 = 'R2_cuda_materialized_full_sort', 'R3_cuda_fused_tiled_bitonic'
KEYS = ('case', 'N', 'd', 'q', 'k', 'backend', 'route', 'mapping', 'tile_size', 'mode')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_session(path):
    env = json.loads((path / 'environment.json').read_text())
    if env.get('status') != 'completed' or env.get('smoke'):
        raise ValueError(f'{path}: only completed full sessions may be combined')
    audit = json.loads((path / 'analysis/data_audit.json').read_text())
    if audit['status'] != 'PASS':
        raise ValueError(f'{path}: data audit failed')
    artifacts = audit.get('artifact_sha256', {})
    if not {'environment.json', 'analysis/unified_summary.csv'} <= artifacts.keys():
        raise ValueError(f'{path}: missing audited artifact hashes')
    for relative, expected in artifacts.items():
        artifact = (path / relative).resolve()
        artifact.relative_to(path.resolve())
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{path}: audited artifact changed: {relative}')
    sources = env.get('source_sha256')
    if not sources:
        raise ValueError(f'{path}: missing source hashes')
    inputs = json.loads((path / 'cpu_core/input_sha256.json').read_text())
    gpu = dict(line.split('=', 1) for line in (path / 'gpu_core/run.txt').read_text().splitlines())
    identity = {k: env[k] for k in ('hostname', 'primary_mapping', 'primary_tile',
                                   'cuda_architecture', 'nvcc', 'warmups', 'repeats')}
    identity['gpu'] = {k: gpu[k] for k in ('gpu_name', 'compute_capability',
                                          'cuda_runtime_version', 'cuda_driver_version')}
    identity['cpu_model'] = next((line.split(':', 1)[1].strip()
        for line in env.get('cpu', '').splitlines() if line.startswith('Model name:')), 'unavailable')
    identity['compiler'] = env.get('cxx', 'unavailable')
    identity['packages'] = env.get('packages', {})
    identity['build_flags_sha256'] = env.get('build_flags_sha256', {})
    cpu_meta_path = path / 'cpu_core/run.txt'
    cpu_meta = dict(line.split('=', 1) for line in cpu_meta_path.read_text().splitlines()) if cpu_meta_path.exists() else {}
    identity['cpu_pin'] = cpu_meta.get('cpu', 'unavailable')
    data = [r for r in read_csv(path / 'analysis/unified_summary.csv')
            if (r['backend'] == 'CPU' and r['group'] == '0') or r['source'] == '0:gpu_core']
    keyed = {tuple(r[k] for k in KEYS): r for r in data}
    if len(keyed) != len(data):
        raise ValueError(f'{path}: duplicate core configurations')
    return identity, sources, inputs, keyed, audit


def summarize(paths):
    loaded = [load_session(path) for path in paths]
    first_identity, first_sources, first_inputs, first_rows, _ = loaded[0]
    for path, (identity, sources, inputs, data, _) in zip(paths[1:], loaded[1:]):
        if identity != first_identity:
            raise ValueError(f'{path}: hardware, mapping or timing configuration differs')
        if sources != first_sources:
            raise ValueError(f'{path}: source hashes differ')
        if inputs != first_inputs:
            raise ValueError(f'{path}: input hashes differ')
        if data.keys() != first_rows.keys():
            raise ValueError(f'{path}: core configurations differ')
    summary = []
    for key in sorted(first_rows, key=lambda k: (int(k[0]), k[6], k[9])):
        records = [data[key] for _, _, _, data, _ in loaded]
        medians = [float(r['median_ms']) for r in records]
        for field in ('aux_bytes', 'peak_device_bytes', 'merge_levels'):
            if len({r[field] for r in records}) != 1:
                raise ValueError(f'Device buffer accounting differs: {key} / {field}')
        row = dict(zip(KEYS, key))
        row.update(sessions=len(paths), median_of_session_medians_ms=statistics.median(medians),
                   min_session_median_ms=min(medians), max_session_median_ms=max(medians),
                   session_range_over_median=(max(medians)-min(medians))/statistics.median(medians),
                   median_session_queries_per_second=statistics.median(float(r['queries_per_second']) for r in records),
                   noisy_sessions=sum(int(r['unstable']) for r in records),
                   aux_bytes=records[0]['aux_bytes'])
        summary.append(row)

    ratios = []
    definitions = [('R0_over_R1', R0, 'host_complete', R1, 'host_complete'),
                   ('R0_over_R2_resident', R0, 'host_complete', R2, 'resident'),
                   ('R0_over_R3_resident', R0, 'host_complete', R3, 'resident'),
                   ('R0_over_R2_end_to_end', R0, 'host_complete', R2, 'end_to_end'),
                   ('R0_over_R3_end_to_end', R0, 'host_complete', R3, 'end_to_end'),
                   ('R2_over_R3_resident', R2, 'resident', R3, 'resident'),
                   ('R2_over_R3_end_to_end', R2, 'end_to_end', R3, 'end_to_end')]
    for case in sorted({key[0] for key in first_rows}, key=int):
        for name, numerator, num_mode, denominator, den_mode in definitions:
            values = []
            for _, _, _, data, _ in loaded:
                def latency(route, mode):
                    return float(next(r['median_ms'] for k, r in data.items()
                                      if k[0] == case and r['route'] == route and r['mode'] == mode))
                values.append(latency(numerator, num_mode) / latency(denominator, den_mode))
            row = {'case': case, 'ratio': name, 'sessions': len(paths),
                   'median_paired_speedup': statistics.median(values),
                   'min_paired_speedup': min(values), 'max_paired_speedup': max(values)}
            row.update({f'session_{i:02d}': value for i, value in enumerate(values, 1)})
            ratios.append(row)
    metadata = {'sessions': [str(p) for p in paths], 'configuration': first_identity,
                'input_hashes_agree': True, 'source_hashes_agree': True,
                'noisy_cpu_groups_by_session': [len(item[4]['cpu_variable_groups']) for item in loaded],
                'noisy_gpu_groups_by_session': [len(item[4]['gpu_variable_groups']) for item in loaded],
                'FP64_ID_position_differences_by_session': [item[4].get('FP64_ID_mismatches', 0) for item in loaded],
                'statistics': 'Median and range of per-session medians; paired speedups per session. Samples are not pooled.'}
    return summary, ratios, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    study = json.loads((args.study / 'study.json').read_text())
    if len(study['sessions']) != study['sessions_requested']:
        raise ValueError('All requested sessions must finish before summarising')
    paths = [args.study / name for name in study['sessions']]
    if len(paths) < 2:
        raise ValueError('At least two separate sessions are required')
    summary, ratios, metadata = summarize(paths)
    write_csv(args.study / 'session_summary.csv', summary)
    write_csv(args.study / 'paired_speedups.csv', ratios)
    (args.study / 'study_summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    baseline = [r for r in summary if r['case'] == '0']
    lines = ['# Repeated CPU/CUDA study', '',
        f"{len(paths)} separate sessions used identical input hashes, source hashes, hardware, mapping and tile size.",
        'Each session retains its own raw samples, correctness records, environment, audit and report.', '',
        '## Baseline across sessions', '',
        '| Route | Boundary | Median of session medians (ms) | Session median range (ms) | Noisy sessions |',
        '|---|---|---:|---:|---:|']
    for row in baseline:
        lines.append(f"| {row['route'].split('_')[0]} | {row['mode']} | {row['median_of_session_medians_ms']:.3f} | "
                     f"{row['min_session_median_ms']:.3f}–{row['max_session_median_ms']:.3f} | {row['noisy_sessions']} |")
    lines += ['', '## Paired baseline speedups', '',
        'Speedups are calculated within each session before taking their median. They are not ratios of pooled medians.', '',
        '| Ratio | Median speedup | Session range |', '|---|---:|---:|']
    for row in ratios:
        if row['case'] == '0':
            lines.append(f"| {row['ratio']} | {row['median_paired_speedup']:.3f}x | "
                         f"{row['min_paired_speedup']:.3f}–{row['max_paired_speedup']:.3f}x |")
    lines += ['', '## Interpretation and limits', '',
        '- Within-session IQR and between-session ranges measure different kinds of variation; neither is a confidence interval.',
        '- Noisy sessions remain included. The report uses the predeclared first CPU timing group and retains additional CPU groups in each session.',
        '- Exact selection follows the declared FP32 arithmetic; FP64 ID agreement on these datasets does not guarantee agreement on every near tie.',
        '- Lower temporary allocation does not establish an equal reduction in measured DRAM traffic.',
        '- These results apply to this machine and the fixed synthetic inputs. Optional algorithms are not claimed.', '',
        f"Recorded FP64 ID-position differences per session: {metadata['FP64_ID_position_differences_by_session']}.", '',
        '## Session reports', '']
    for path, cpu_noise, gpu_noise in zip(paths, metadata['noisy_cpu_groups_by_session'], metadata['noisy_gpu_groups_by_session']):
        lines.append(f'- [{path.name}]({path.name}/analysis/final_results.md): {cpu_noise} noisy CPU groups, {gpu_noise} noisy GPU groups.')
    (args.study / 'study_summary.md').write_text('\n'.join(lines) + '\n')
    print(args.study / 'study_summary.md')


if __name__ == '__main__':
    main()
