#!/usr/bin/env python3
"""Generate the report for the retained CPU routes from recorded measurements."""
import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='?', type=Path, default=ROOT / 'results/main')
    args = parser.parse_args()
    directory = args.results
    summary = read(directory / 'summary.csv')
    correctness = read(directory / 'correctness.csv')
    raw = read(directory / 'raw.csv')
    inputs = read(directory / 'inputs.csv')
    baseline = {row['route']: row for row in summary
                if row['case'] == '0' and row['group'] == '0'}
    checks_pass = all(all(row[key] == '1' for key in
                          ('structure', 'labels', 'sorted',
                           'fixed_distance_selection', 'distance_tolerance',
                           'deterministic')) for row in correctness)
    mismatches = sum(int(row['ID_mismatches']) for row in correctness)
    maximum_error = max(float(row['max_abs_error']) for row in correctness)
    rows = []
    for route in ('R0_scalar_full_sort', 'R1_avx2_topk_heap'):
        row = baseline[route]
        rows.append(f"| {route} | {1000*float(row['median']):.3f} | "
                    f"{100*float(row['IQR'])/float(row['median']):.1f}% | "
                    f"{row['aux_bytes']} |")
    parameter_rows = ['| ' + ' | '.join(row[key] for key in ('case','N','d','q','k')) + ' |'
                      for row in inputs]
    ablation_plot = os.path.relpath(directory / 'cpu_ablation.png', ROOT / 'docs')
    sweep_plot = os.path.relpath(directory / 'parameter_sweeps.png', ROOT / 'docs')
    noisy = sum(row['unstable'] == '1' for row in summary)
    text = f'''# Required CPU kNN Implementation: Results

Date: 2026-10-19. This report covers only the required CPU work: scalar full sort with an independent FP64 oracle, AVX2 distance calculation with a Top-K heap, and one candidate-list merge method. Results refer to the retained source, build flags, input hashes, and raw records.

## Methods

R0 computes all scalar FP32 squared distances, stores N candidates, and sorts by distance then ID. R1 computes FP32 distances with AVX2 eight-lane accumulation and a scalar tail, maintaining a max-heap of at most k candidates and sorting the retained output. The candidate merge helper combines two ordered lists up to k entries; an independent unit test compares it with a full sort of the concatenated inputs.

The oracle promotes the actual FP32 inputs to FP64, computes complete distances independently, and sorts by distance then ID. Correctness records separately report structure, labels, output ordering, fixed-distance selection, distance tolerance, FP64 ID agreement, and repeat determinism. The distance criterion is atol=1e-5 and rtol=1e-4. All required structural, selection, tolerance, and determinism fields pass: **{checks_pass}**. There are {mismatches} FP64 ID-position differences across {len(correctness)} route/configuration records; the maximum recorded absolute distance error is {maximum_error:.9g}. Near-boundary ID differences are reported separately from selection correctness.

## Experimental setup

The baseline is N=100000, d=64, q=16, k=10. The eight configurations vary N, d, q, and k one factor at a time. Inputs use synthetic-v1, MT19937 seed 5206, X followed by Q, and labels equal to ID modulo 10. Input arrays and SHA-256 hashes are retained under `results/main/`.

Each route receives 10 warm-ups and 20 timed complete-search calls per group. Timing includes per-query state reset, distance computation, candidate selection, final ordering, and label gathering. It excludes input generation, full validation, I/O, oracle work, correctness checks, and initial reusable allocation. Routes alternate order each round. The guest CPU affinity, compact environment, compiler settings, and source instructions are recorded alongside the results.

## Baseline performance

Times are complete batch latency in milliseconds. Auxiliary bytes report route workspace capacity, excluding input, output, and the oracle.

| Route | Median (ms) | IQR / median | Auxiliary bytes |
|---|---:|---:|---:|
{chr(10).join(rows)}

The observed scalar-full-sort to AVX2-heap median ratio is **{float(baseline['R0_scalar_full_sort']['median'])/float(baseline['R1_avx2_topk_heap']['median']):.3f}x** on this guest. This is a measurement for the recorded virtual machine and inputs. It does not establish physical-host performance. There are {noisy} variable route/group records with IQR/median above 10%; consult the raw records before making broader performance claims.

## Parameter configurations

| Case | N | d | q | k |
|---|---:|---:|---:|---:|
{chr(10).join(parameter_rows)}

![Required-route baseline comparison]({ablation_plot})

![Single-factor CPU measurements]({sweep_plot})

## Reproduction

Build and validate using the commands in [CPU_Guide.md](../CPU_Guide.md). Run `knn_bench` into a fresh result directory, then run `scripts/analyze.py` and `scripts/report.py` against that directory. The main data, hashes, raw timing records, correctness records, and generated plots are retained under `results/main/`.
'''
    (ROOT / 'docs/report.md').write_text(text)


if __name__ == '__main__':
    main()
