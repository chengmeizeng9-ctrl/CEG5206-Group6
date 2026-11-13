# CPU Implementation and Intel Core Ultra 5 125H Experiment

The historical results in `results/main/` were measured on an Intel Core Ultra 5 125H reported by an openEuler/VMware guest. This directory was previously named `CPU/`; it is now named `CPU_Ultra5_125H/` to distinguish it from the two i7-12700H experiment directories. The maintained CPU implementation is also used by the unified CPU/GPU workflow, whose separate results record their own execution environments.

This directory contains the required CPU methods: a scalar full-sort baseline, an independent FP64 oracle, AVX2 distances with a Top-K max-heap, and a separately tested sorted-list merge method. The [experiment protocol](../docs/EXPERIMENT_PROTOCOL.md) defines the shared computation and numerical contract.

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
| `src/optional.cpp` | Tiled Bitonic selection, scalar early abandonment and its heap control |
| `src/bench.cpp`, `src/export.cpp` | CPU benchmark and result export |
| `tests/` | Route, oracle, heap, merge, and boundary checks |
| `scripts/` | Input hashes, analysis plots, evidence checks, and report generation |
| [results/main/](results/main/) | Historical Ultra 5 125H CPU measurements, input arrays, SHA-256 hashes, correctness records and plots |
| [results/environment_summary.json](results/environment_summary.json) | Hardware, virtual-machine environment, compiler and measurement settings for the historical CPU experiment |
| [artifacts/](artifacts/) | Retained Release, clean-build and sanitizer executables and static libraries; see [Build_Artifacts.md](artifacts/Build_Artifacts.md) |
| [../results/full/cpu_core/](../results/full/cpu_core/) | CPU records, input manifests, hashes and plots from the unified CPU/GPU study; generated arrays must be regenerated or retrieved if absent |

The CPU benchmark binds one thread to an available CPU and records the selected CPU. Historical Ultra 5 125H measurements are retained under `results/main/`, and their original compiled targets are under `artifacts/`. Unified CPU/GPU measurements and their environment records are stored separately under the repository-level `results/` directory.

The retained targets include `knn_bench`, `knn_export`, `knn_tests` and `libknn.a` under `artifacts/release/`, `artifacts/clean/` and `artifacts/sanitizer/`. They correspond to their recorded historical builds; use the build commands above for the maintained source on another machine.

## Candidate merge API

`merge_topk(a, na, b, nb, out, k)` combines two already sorted candidate
lists, writes at most k results and returns the output length. Output capacity
must be at least k and must not overlap the inputs. Ordering is ascending
distance, then ID; numerical tolerance does not participate in ordering.
Tests cover ties, unequal and empty lists, truncation and comparison with a
full sort of the concatenated candidates.

## Optional CPU routes

`Route::Avx2TiledBitonic` adds power-of-two tiles 32–256, infinity padding,
AVX2 distances, scalar local Bitonic sorting networks and packed hierarchical
candidate merging.
It supports every valid k up to N, including k larger than one tile.
`Config::tile_size` defaults to 128. Its output must match the AVX2 heap
bit-for-bit.

`Route::ScalarEarlyAbandon` maintains a scalar Top-k heap and stops a
candidate only when its partial non-negative sum is strictly above the current
worst selected distance. `Route::ScalarTopKHeap` is the full-evaluation timing
control using the same heap path. Both match R0's scalar distance arithmetic.
`Diagnostics` records examined dimensions, candidates and abandoned candidates.
The diagnostic search is outside benchmark timing.

The standalone CPU CTest includes `knn_optional_tests`. The optional experiment
command in the root README benchmarks both new routes and their controls.
