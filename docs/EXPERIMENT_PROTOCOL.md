# Full Exact-kNN Experimental Protocol

## 1. Required scope

The final required comparison contains four routes:

| Route | Distance | Selection and memory organisation |
|---|---|---|
| R0 | Scalar FP32 CPU | Materialise all candidates and full sort |
| R1 | AVX2 FP32 CPU | Top-k max-heap |
| R2 | CUDA FP32 | Materialise `q x N`, pack deterministic keys, segmented full sort |
| R3 | CUDA FP32 | Fused tiled distance + local Bitonic Top-k + hierarchical merge |

An independent host FP64 oracle validates numerical error and Top-k IDs. All
routes return exactly `k` query-major `(distance, ID, label)` records ordered by
ascending distance and then ascending ID.

CPU Bitonic, custom Radix Select, GEMM-based distance, early abandonment, and
out-of-core double buffering remain optional. They are not started until all
four required routes pass the complete correctness gate and the final core
experiment has been reproduced from a clean build.

## 2. Frozen inputs

The CPU benchmark creates the eight bounded configurations with synthetic-v1,
MT19937 seed 5206. It exports each array and its shape. Every CUDA experiment
loads those files directly.

| Case | N | d | q | k | Purpose |
|---|---:|---:|---:|---:|---|
| 0 | 100000 | 64 | 16 | 10 | Baseline |
| 1 | 10000 | 64 | 16 | 10 | Database scaling |
| 2 | 100000 | 32 | 16 | 10 | Lower dimension |
| 3 | 100000 | 128 | 16 | 10 | Higher dimension |
| 4 | 100000 | 64 | 1 | 10 | Single-query latency |
| 5 | 100000 | 64 | 64 | 10 | Batch throughput |
| 6 | 100000 | 64 | 16 | 5 | Smaller Top-k |
| 7 | 100000 | 64 | 16 | 32 | Larger Top-k |

The full grid is deliberately one-factor-at-a-time. A representative
`N = 10^6` GPU case may be added later, but it is not mixed into the required
core grid.

## 3. Correctness gates before timing

A route is permitted into performance plots only when all applicable checks
pass:

1. output shape and exactly `q * k` records;
2. valid, distinct IDs per query;
3. original label gathering;
4. sorted `(distance, ID)` total order;
5. finite non-negative distances with positive zero normalization;
6. absolute-plus-relative FP64 tolerance (`atol=1e-5`, `rtol=1e-4`);
7. repeated-call bitwise determinism;
8. R3 exact equality with same-mapping materialised R2;
9. hierarchical merge property tests against a full sort;
10. Compute Sanitizer memcheck and racecheck before final data collection.

FP64 Top-k ID agreement is retained as a separate field. It is not silently
collapsed into the FP32 route-specific selection check. Exact selection uses
the declared FP32 arithmetic order; near ties may differ from FP64 ranking.
The audit reports such differences separately while requiring correct FP32
selection, tolerance, determinism and same-mapping R2/R3 equality.

## 4. GPU design ablations

### Mapping ablation

Run R2 and R3 on case 0 with both mappings:

- thread-per-vector;
- warp-per-vector.

Select the faster R3 mapping by median resident latency and freeze it before
running the complete eight-case GPU grid. Record the losing route rather than
deleting it.

### Tile sensitivity

For case 0 and the selected mapping, run R3 with:

```text
T = 32, 64, 128, 256
```

Report complete resident latency, fused-local time, merge time, merge depth,
and allocated candidate storage. Keep `T=128` as the predefined reference but
do not claim it is optimal unless the measurement supports that conclusion.

## 5. Timing protocol

Use the same physical machine for R0-R3 speedup claims.

- Release builds only.
- CUDA target `sm_61` for the GTX 1050.
- CUDA 12.x, because CUDA 13 cannot compile Pascal targets.
- Ten warm-ups and twenty measured repetitions for each retained route.
- Median, Q1, Q3, IQR, and IQR/median.
- Flag IQR/median above 10% rather than hiding noisy groups.
- Apply the flag to both CPU and GPU. Retain the predeclared CPU group 0 for
  headline comparisons and retain any additional groups, including noisy ones.
- Reusable input, output, sort, and candidate buffers are preallocated.
- Input generation, file I/O, FP64 oracle construction, verification, and
  allocation are outside steady-state timing.
- Include distance, selection, merge, output ordering, label gathering, and
  synchronization required by each route.
- Report GPU-resident and H2D+GPU+D2H end-to-end latency separately.

The fused kernel reports distance plus local Bitonic selection as one stage.
Separating those operations would require a different, non-fused kernel and
would invalidate the stage interpretation.

R0/R1 changes SIMD arithmetic and selection together, so it measures their
combined design effect. It does not isolate an SIMD-only speedup. R2/R3 must
use the same mapping for exact comparisons. Record the full auxiliary buffers:
R2 uses `20qN + 4(q+1) + S_CUB` bytes; R3 uses `16q ceil(N/T) k` bytes for two
candidate buffers. Report inputs, outputs and dynamic shared memory separately.
Allocated storage is not a measurement of DRAM traffic.

Repeat the full grid in three separate sessions. Select/freeze the mapping in
session 1 and retain it thereafter, keeping subsequent mapping ablations.
Require identical input hashes, source hashes and hardware/timing configuration
before combining sessions. Report median/range of session medians and paired
within-session speedups; do not pool samples or label ranges as confidence
intervals. Consecutive sessions do not guarantee statistical independence.

## 6. Required figures and tables

1. Baseline R0-R3 median latency with IQR, GPU resident.
2. Baseline R0-R3 median latency with IQR, GPU transfer-inclusive.
3. R2/R3 CUDA stage breakdown.
4. Latency versus `N`.
5. Latency versus `d`.
6. Throughput versus `q`.
7. Selection sensitivity versus `k`.
8. Auxiliary-memory comparison.
9. Thread-versus-warp mapping ablation.
10. Tile-size latency and memory sensitivity.
11. Correctness table with every gate and FP64 ID mismatches.
12. Hardware/software environment and raw-record counts.

## 7. Predeclared hypotheses

- R1 should outperform R0 by reducing both arithmetic overhead through SIMD
  and selection work from `O(N log N)` to `O(N log k)`.
- R2 should expose the cost of writing and fully sorting `qN` candidates.
- R3 should reduce global candidate traffic to approximately
  `8 q ceil(N/T) k` bytes before ping-pong merge storage, but local sorting and
  synchronization may dominate at small workloads.
- GPU batching should improve queries per second as `q` increases, while
  single-query latency may remain CPU-favourable.
- Higher `d` should increase the fraction of time spent on distance
  computation and reduce the relative influence of selection.
- Smaller tiles reduce each local sorting network but produce more candidate
  lists and merge work; larger tiles do the reverse.

These hypotheses are written before examining final results. A route that does
not help is retained and explained rather than removed after measurement.

## 8. One-command reproduction

```bash
./run_experiment.sh --sessions 3 --clean-build
```

For an initial integration check:

```bash
./run_experiment.sh --smoke
```

The driver builds, tests and runs sanitizers before measurements. Each full
session performs CPU input generation, mapping/core/tile experiments, analysis,
report generation and an independent data audit. `--clean-build` recompiles
targets. New timestamped results preserve previous runs. Source/environment
records and complete logs accompany each session. The repeated-study root
contains `study_summary.md`; each session's `analysis/` directory supplies its
figures. Raw CSV files remain the authoritative timing evidence. See
[the README](../README.md) for commands and failure recovery.
