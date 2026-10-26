# Required CPU kNN Implementation

This directory contains only the required CPU methods: a scalar full-sort baseline, an independent FP64 oracle, AVX2 distances with a Top-K max-heap, and a separately tested sorted-list merge method. The [experiment protocol](../docs/EXPERIMENT_PROTOCOL.md) defines the shared computation and numerical contract.

## Build and validate

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The scalar full-sort baseline can run without AVX2. The Top-K route requires x86 AVX2. The test executable checks both routes against the FP64 oracle and tests heap selection and candidate merging on fixed candidate lists.

## Run the CPU experiment

Run from this directory and use a fresh output directory:

```bash
./build/knn_bench results/reproduction
python3 scripts/analyze.py results/reproduction
python3 scripts/report.py results/reproduction
```

The main experiment contains eight configurations: a baseline plus single-factor changes to N, d, q, and k. Each route receives 10 warm-ups and 20 measured calls. The benchmark records raw times, summaries, correctness, input arrays, dimensions, and hashes. See [the retained full experiment](../results/README.md) for current evidence.

## Required routes

| Route | Distance computation | Candidate selection |
|---|---|---|
| R0 | Scalar FP32, complete evaluation | Full sort |
| R1 | AVX2 FP32, eight lanes and scalar tail | Top-K max-heap |

The FP64 oracle independently ranks actual input values. Candidate order uses ascending distance, then ascending ID for exact ties. The merge helper combines two sorted candidate lists and is tested independently against a full sort.

## Source and results

| Location | Purpose |
|---|---|
| `include/knn.hpp` | Input, candidate, route, result, and validation interfaces |
| `src/scalar.cpp`, `src/simd.cpp` | Scalar and AVX2 distance functions |
| `src/knn.cpp` | Validation, two required routes, Top-K heap, list merge, and input generation |
| `src/oracle.cpp` | Independent FP64 ranking and correctness checks |
| `src/bench.cpp`, `src/export.cpp` | CPU benchmark and result export |
| `tests/` | Route, oracle, heap, merge, and boundary checks |
| `scripts/` | Input hashes, analysis plots, evidence checks, and report generation |
| `../results/full/cpu_core/` | Retained CPU records, local input arrays, hashes, and plots |

The CPU benchmark binds one thread to an available CPU and records the selected CPU. Hardware and execution details are retained with the full experiment. Compiled binaries and historical imported CPU-only measurements are not included in the cleaned repository.

## Candidate merge API

`merge_topk(a, na, b, nb, out, k)` combines two already sorted candidate
lists, writes at most k results and returns the output length. Output capacity
must be at least k and must not overlap the inputs. Ordering is ascending
distance, then ID; numerical tolerance does not participate in ordering.
Tests cover ties, unequal and empty lists, truncation and comparison with a
full sort of the concatenated candidates.
