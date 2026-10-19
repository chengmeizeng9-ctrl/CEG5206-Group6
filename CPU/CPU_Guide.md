# Required CPU kNN Implementation

This directory contains only the required CPU methods: a scalar full-sort baseline, an independent FP64 oracle, AVX2 distances with a Top-K max-heap, and a separately tested sorted-list merge method. The [CPU rules](../CPU_rules.md) define the included work; the group's shared computation contract defines common input and numerical conventions.

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

The main experiment contains eight configurations: a baseline plus single-factor changes to N, d, q, and k. Each route receives 10 warm-ups and 20 measured calls. The benchmark records raw times, summaries, correctness, input arrays, dimensions, and hashes. See `results/main/` for the retained experiment.

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
| `results/main/` | Required-route records, input arrays, hashes, and plots |
| `artifacts/` | Compiled binaries for the retained CPU implementation |

The CPU benchmark binds one thread to an available guest CPU and records the selected CPU. Virtual-machine timings describe the recorded guest and do not establish physical-host performance.
