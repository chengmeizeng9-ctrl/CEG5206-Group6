#!/usr/bin/env python3
"""Generate a results-section draft from validated combined experiment records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

R0 = "R0_scalar_full_sort"
R1 = "R1_avx2_topk_heap"
R2 = "R2_cuda_materialized_full_sort"
R3 = "R3_cuda_fused_tiled_bitonic"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def find(rows: list[dict[str, str]], *, route: str, mode: str,
         case: str = "0", mapping: str = "", tile: str = "",
         source_prefix: str = "") -> dict[str, str]:
    matches = []
    for row in rows:
        if source_prefix and not row["source"].startswith(source_prefix):
            continue
        if (row["case"] == case and row["route"] == route and
                row["mode"] == mode and
                (not mapping or row["mapping"] == mapping) and
                (not tile or row["tile_size"] == tile)):
            matches.append(row)
    if not matches:
        raise KeyError((route, mode, case, mapping, tile, source_prefix))
    matches.sort(key=lambda row: row.get("group", "0") != "0")
    return matches[0]


def ms(row: dict[str, str]) -> float:
    return float(row["median_ms"])


def mib(value: str) -> float:
    return int(value) / (1024 ** 2)


def format_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def tile_interpretation(selected: list[dict[str, str]], reference_tile: str) -> str:
    if not selected:
        return 'No tile comparison was recorded.'
    ordered = sorted(selected, key=lambda row: int(row['tile_size']))
    best = min(ordered, key=ms)
    first, last = ordered[0], ordered[-1]
    reference = next((row for row in ordered if row['tile_size'] == reference_tile), None)
    text = (f"Among the measured tiles, T={best['tile_size']} had the lowest resident "
            f"median ({ms(best):.3f} ms). From T={first['tile_size']} to T={last['tile_size']}, "
            f"allocated candidate storage changed from {mib(first['aux_bytes']):.3f} to "
            f"{mib(last['aux_bytes']):.3f} MiB and merge depth changed from "
            f"{first['merge_levels']} to {last['merge_levels']} levels. "
            f"The measured fused-local medians were {float(first['median_local_selection_ms']):.3f} "
            f"and {float(last['median_local_selection_ms']):.3f} ms; merge medians were "
            f"{float(first['median_merge_ms']):.3f} and {float(last['median_merge_ms']):.3f} ms. ")
    if reference:
        text += f"The predefined T={reference_tile} reference measured {ms(reference):.3f} ms in this ablation. "
    return text + ('Larger tiles produce fewer candidate lists, while their local sorting networks are larger. '
                   'The fastest measured tile is specific to this baseline, mapping and machine; '
                   'the required core grid retains its predefined reference tile.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--primary-mapping", required=True)
    parser.add_argument("--primary-tile", required=True)
    args = parser.parse_args()

    rows = read_csv(args.analysis / "unified_summary.csv")
    correctness = read_csv(args.analysis / "unified_correctness.csv")
    analysis = json.loads((args.analysis / "analysis.json").read_text())
    if not analysis.get('all_required_correctness_passed'):
        raise RuntimeError('Cannot report performance before correctness passes')
    env_path = args.analysis.parent / 'environment.json'
    environment = json.loads(env_path.read_text()) if env_path.exists() else {}
    def picture(name: str, caption: str) -> str:
        return f'![{caption}]({name})' if (args.analysis / name).exists() else f'{caption}: not collected in this run.'

    r0 = find(rows, route=R0, mode="host_complete")
    r1 = find(rows, route=R1, mode="host_complete")
    r2_res = find(rows, route=R2, mode="resident", mapping=args.primary_mapping,
                  source_prefix="0:")
    r3_res = find(rows, route=R3, mode="resident", mapping=args.primary_mapping,
                  tile=args.primary_tile, source_prefix="0:")
    r2_e2e = find(rows, route=R2, mode="end_to_end",
                  mapping=args.primary_mapping, source_prefix="0:")
    r3_e2e = find(rows, route=R3, mode="end_to_end",
                  mapping=args.primary_mapping, tile=args.primary_tile,
                  source_prefix="0:")

    gpu_rows = [row for row in rows if row["backend"] == "CUDA"]
    mappings: dict[str, set[str]] = defaultdict(set)
    tiles: dict[str, set[str]] = defaultdict(set)
    for row in gpu_rows:
        if row["case"] == "0" and row["mode"] == "resident":
            mappings[row["source"]].add(row["mapping"])
            if row["route"] == R3 and row["mapping"] == args.primary_mapping:
                tiles[row["source"]].add(row["tile_size"])
    mapping_source = next((source for source, values in mappings.items()
                           if {"thread_per_vector", "warp_per_vector"} <= values),
                          None)
    tile_source = next((source for source, values in tiles.items()
                        if len({value for value in values if value}) >= 2), None)

    mapping_table = []
    if mapping_source:
        for route in (R2, R3):
            for mapping in ("thread_per_vector", "warp_per_vector"):
                try:
                    row = find(rows, route=route, mode="resident",
                               mapping=mapping,
                               tile=args.primary_tile if route == R3 else "",
                               source_prefix=mapping_source.split(":", 1)[0] + ":")
                except KeyError:
                    continue
                mapping_table.append(
                    f"| {route.split('_')[0]} | {mapping} | "
                    f"{float(row['median_ms']):.3f} |"
                )

    tile_table = []
    selected = []
    if tile_source:
        selected = [row for row in rows
                    if row["source"] == tile_source and row["case"] == "0"
                    and row["route"] == R3 and row["mode"] == "resident"
                    and row["mapping"] == args.primary_mapping]
        selected.sort(key=lambda row: int(row["tile_size"]))
        for row in selected:
            tile_table.append(
                f"| {row['tile_size']} | {float(row['median_ms']):.3f} | "
                f"{float(row['median_local_selection_ms']):.3f} | "
                f"{float(row['median_merge_ms']):.3f} | "
                f"{mib(row['aux_bytes']):.3f} | {row['merge_levels']} |"
            )

    id_mismatches = sum(int(row.get("ID_mismatches", "0") or 0)
                        for row in correctness)
    max_error = max(float(row.get("max_abs_error", "0") or 0)
                    for row in correctness)
    gpu_pair_failures = sum(
        row.get("backend") == "CUDA" and row.get("materialized_match") == "0"
        for row in correctness
    )
    noisy = analysis["variable_gpu_groups"]
    noisy_cpu = [row for row in rows if row['backend'] == 'CPU' and row['unstable'] == '1']
    noise_table = []
    for row in rows:
        if row['unstable'] == '1':
            noise_table.append(f"| {row['backend']} | {row['source']} | {row['case']} | "
                              f"{row['route'].split('_')[0]} | {row['group']} | {row['mode']} | "
                              f"{row['mapping']} / {row['tile_size']} | {100*float(row['IQR_over_median']):.1f}% |")
    baseline_noisy = [r for r in (r0, r1, r2_res, r3_res, r2_e2e, r3_e2e) if r['unstable'] == '1']
    qualification = ('Baseline speedups include groups flagged above 10% IQR/median; '
                     'treat their precise magnitudes as variable and consult separate sessions.'
                     if baseline_noisy else
                     'No baseline group exceeded 10% IQR/median in this session; separate sessions still test repeatability.')
    workloads = []
    for case in sorted({r['case'] for r in rows if r['source'].startswith('0:')}, key=int):
        candidates = []
        try:
            for route in (R0, R1):
                candidates.append(find(rows, route=route, mode='host_complete', case=case))
            for route in (R2, R3):
                candidates.append(find(rows, route=route, mode='end_to_end', case=case,
                                       mapping=args.primary_mapping,
                                       tile=args.primary_tile if route == R3 else '', source_prefix='0:'))
        except KeyError:
            continue
        fastest = min(candidates, key=ms)
        shape = candidates[0]
        workloads.append(f"| {case} | {shape['N']} / {shape['d']} / {shape['q']} / {shape['k']} | "
                         + ' | '.join(f'{ms(r):.3f}' for r in candidates)
                         + f" | {fastest['route'].split('_')[0]} |")
    stage_text = (f"At the core baseline, R2 distance and full-sort medians were "
                  f"{float(r2_res['median_distance_ms']):.3f} and {float(r2_res['median_full_sort_ms']):.3f} ms. "
                  f"R3 fused-local and merge medians were {float(r3_res['median_local_selection_ms']):.3f} "
                  f"and {float(r3_res['median_merge_ms']):.3f} ms. "
                  'Stage medians are computed separately and need not sum to the complete median.')
    memory_text = (f"R2 allocated {mib(r2_res['aux_bytes']):.3f} MiB of auxiliary device buffers; "
                   f"R3 allocated {mib(r3_res['aux_bytes']):.3f} MiB "
                   f"({int(r2_res['aux_bytes'])/int(r3_res['aux_bytes']):.2f}x less). "
                   'R2 includes distances, two packed-key buffers, offsets and CUB scratch. '
                   'R3 includes both candidate buffers. Inputs, outputs and dynamic shared memory '
                   'are recorded separately. These allocations do not measure DRAM traffic.')
    timing_text = (f"CPU routes use ten warm-ups and twenty measured calls per retained group. "
                   f"GPU groups contain {r2_res['samples']} samples; GPU warm-ups are "
                   f"{environment.get('warmups', 'recorded in each run.txt')}. "
                   'The predeclared CPU group 0 supplies headline comparisons; any additional groups remain retained.')
    selection_text = ('The mapping was selected by the separate baseline R3 resident ablation.'
                      if environment.get('primary_mapping_selection', 'auto') == 'auto' else
                      'The mapping was explicitly fixed, including when inherited from the first session of a repeated study.')

    baseline_rows = [
        f"| R0 scalar full sort | CPU complete | {ms(r0):.3f} | "
        f"{mib(r0['aux_bytes']):.3f} |",
        f"| R1 AVX2 Top-k heap | CPU complete | {ms(r1):.3f} | "
        f"{mib(r1['aux_bytes']):.3f} |",
        f"| R2 materialised CUDA | GPU resident | {ms(r2_res):.3f} | "
        f"{mib(r2_res['aux_bytes']):.3f} |",
        f"| R3 fused CUDA | GPU resident | {ms(r3_res):.3f} | "
        f"{mib(r3_res['aux_bytes']):.3f} |",
        f"| R2 materialised CUDA | End-to-end | {ms(r2_e2e):.3f} | "
        f"{mib(r2_e2e['aux_bytes']):.3f} |",
        f"| R3 fused CUDA | End-to-end | {ms(r3_e2e):.3f} | "
        f"{mib(r3_e2e['aux_bytes']):.3f} |",
    ]

    text = f"""# Exact-kNN CPU/CUDA Experiment Results

## Experimental contract

All implementations examine every database vector and return exactly `k`
query-major results ordered by ascending computed FP32 squared-L2 distance and then ascending
sample ID. R0 is the scalar CPU full-sort baseline, R1 combines AVX2 distance
computation with a Top-k max-heap, R2 materialises and fully sorts the CUDA
distance table, and R3 fuses tiled distance computation with local Bitonic
selection and hierarchical candidate merging.

The primary CUDA configuration uses `{args.primary_mapping}` and `T =
{args.primary_tile}`. {selection_text} Reusable buffers are preallocated.
{timing_text} The environment identifies whether this is a smoke run.

## Algorithm and storage model

For C=ceil(N/T) and P=next_power_of_two(k), all required routes perform
O(qNd) distance work. R0 fully sorts N candidates per query; R1 retains a
max-heap and finally orders its k outputs. R2 uses CUB full radix sorting
of packed 64-bit keys, not a custom radix-selection implementation.

| Route | Selection work | Principal temporary storage |
|---|---|---|
| R0 | O(qN log N) comparison sorting | O(N) candidates reused per query |
| R1 | O(qN log k), plus final O(qk log k) | O(k) candidates |
| R2 | Full segmented radix sorting of fixed-width keys | 20qN + 4(q+1) + S_CUB bytes |
| R3 local | O(qCT log²T) local Bitonic sorting | Two candidate buffers: 16qCk bytes |
| R3 merge | O(q(C−1)P log²(2P)) for the implemented padded pair sorting | ceil(log2 C) merge levels |

Local pruning is exact under the declared total order: a candidate beyond a
tile's local Top-k already has at least k candidates ahead of it in that tile.
It therefore cannot enter global Top-k. Repeated correct Top-k reductions of
the candidate unions preserve that result. The unit tests compare the merge
with concatenation and full sorting, including ties and odd list counts.

## Correctness

All required correctness records passed. Across CPU and GPU outputs, the
records report **{id_mismatches}** FP64 Top-k position differences and a maximum
absolute distance error of **{max_error:.9g}**. CPU error records include all
candidate distances; GPU error records cover the returned Top-k. FP64 ID agreement is
reported separately from FP32 route-specific selection. The fused route had
**{gpu_pair_failures}** failures against the same-mapping materialised full-sort
result. Repeated-call determinism, labels, output ordering, finite non-negative
distances and the stated absolute-plus-relative tolerance were all required
before timing data entered the analysis.

Exact selection is defined under each route's documented FP32 accumulation
order. Near ties can change FP64 neighbour ordering after FP32 rounding;
universal FP64 ID equality is not implied by passing distance tolerance.
The numerical criterion is `abs(d32-d64) <= 1e-5 + 1e-4*abs(d64)`.

## Baseline performance and storage

| Route | Timing boundary | Median latency (ms) | Auxiliary storage (MiB) |
|---|---|---:|---:|
{chr(10).join(baseline_rows)}

For the baseline workload, R1 achieved **{analysis['speedups']['R0_over_R1']:.3f}x**
over R0. Relative to R2, the fused R3 route achieved
**{analysis['speedups']['R2_over_R3_resident']:.3f}x** in GPU-resident timing and
**{analysis['speedups']['R2_over_R3_end_to_end']:.3f}x** when transfers were
included. These ratios describe the recorded machine and fixed inputs; they are
not hardware-independent peak-performance claims.

{qualification}

The R0/R1 comparison combines SIMD arithmetic and heap selection. It does not
isolate the SIMD contribution. {memory_text}

{picture('baseline_resident.png', 'GPU-resident baseline')}

{picture('baseline_end_to_end.png', 'Transfer-inclusive baseline')}

## CUDA stage behaviour

R2 separately records distance computation, deterministic key packing,
segmented full sorting and final gathering. R3 records one combined fused-local
stage because distance computation and local Bitonic selection occur in the
same kernel; inventing separate values would require changing the algorithm.
Hierarchical merge and final gathering remain separate stages.

{stage_text}

{picture('gpu_stage_breakdown.png', 'CUDA stage breakdown')}

## Workload sensitivity

The one-factor-at-a-time grid changes database size, dimension, query batch and
returned `k` while holding the remaining baseline values fixed. This allows the
report to attribute each crossover to one workload factor instead of a large,
confounded Cartesian product. A crossover is claimed only within the measured
grid; an automated backend policy and unmeasured thresholds are not claimed.

The following comparison includes GPU transfers. Its fastest route is selected
from the measured CPU complete and GPU end-to-end medians, with variability
flags retained below.

| Case | N / d / q / k | R0 (ms) | R1 (ms) | R2 end-to-end (ms) | R3 end-to-end (ms) | Lowest measured median |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(workloads)}

{picture('parameter_sweeps.png', 'Parameter sweeps')}

{picture('baseline_auxiliary_memory.png', 'Auxiliary memory')}

## Mapping ablation

| Route | Mapping | Median resident latency (ms) |
|---|---|---:|
{chr(10).join(mapping_table) if mapping_table else '| N/A | N/A | N/A |'}

{picture('mapping_ablation.png', 'Mapping ablation')}

## Tile-size sensitivity

| T | Complete (ms) | Fused local (ms) | Merge (ms) | Auxiliary MiB | Merge levels |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(tile_table) if tile_table else '| N/A | N/A | N/A | N/A | N/A | N/A |'}

{tile_interpretation(selected, args.primary_tile)}

{picture('tile_sensitivity.png', 'Tile latency')}

{picture('tile_memory.png', 'Tile memory')}

## Variability and limitations

There were **{len(noisy_cpu)} CPU** and **{len(noisy)} GPU** summary groups
with IQR/median above 10%. All groups remain in the raw records.

| Backend | Source | Case | Route | CPU group | Mode | Mapping / tile | IQR/median |
|---|---|---|---|---|---|---|---:|
{chr(10).join(noise_table) if noise_table else '| None flagged | | | | | | | |'}

This single session does not establish between-session repeatability or
confidence intervals. Use the repeated-study summary when separate sessions
have completed. Synthetic inputs and this GPU/CPU limit generalisation.
Source hashes and the available Git revision are captured in the environment;
an unavailable Git revision is not described as a reviewed commit.

The materialised route uses CUB only for its labelled full-sort stage;
all required distance kernels, fused local Bitonic selection and hierarchical
candidate merging are self-written. The experiment targets exact search and
does not compare HNSW, IVF, product quantisation, multi-GPU execution or neural
network training.

Optional CPU Bitonic, custom Radix Select, GEMM-based L2, early abandonment and
out-of-core double buffering should be reported only if implemented after this
required experiment is frozen and reproduced.
"""
    (args.analysis / "final_results.md").write_text(text)
    print(args.analysis / "final_results.md")


if __name__ == "__main__":
    main()
