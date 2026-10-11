# Exact kNN Search on CPUs

Our group project studies how distance computation, selection algorithms, and memory organisation affect exact kNN search. This directory contains our C++17 CPU implementation, correctness framework, experiments, and reproducibility materials. We follow computation contract version 1.1 and execution rules version 1.1.

## Documentation

- [Technical report](docs/report.md)
- [Module integration and shared validation](docs/integration.md)
- [Demonstration guide](docs/demonstration.md)
- [Our CPU implementation summary](docs/contributions.md)
- [Group computation contract](../Three-Member%20Shared%20Contract.md)

This public CPU folder contains English documentation. The original Chinese source documents remain in the working project outside this GitHub submission folder.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 tests/test_external.py
```

The project requires a C++17 compiler and CMake. Our AVX2 routes require an x86 CPU with AVX2 support; scalar search does not require AVX2. The project uses x86 MXCSR controls and does not currently support ARM. Our recorded environment is openEuler with GCC 12.3.1, CMake 3.27.9, Python 3.11.6, NumPy 2.4.6, and Matplotlib 3.11.1.

For address and undefined-behaviour checks:

```bash
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON
cmake --build build-sanitize -j2
ctest --test-dir build-sanitize --output-on-failure
```

The compiler's ASan and UBSan runtime libraries must be installed. `clang-format` is an optional formatting tool, not a build dependency.

## Run and reproduce

Run commands from this directory. Use a new output directory to preserve existing results: the benchmark overwrites files with matching names in its output directory.

```bash
# Export shared correctness fixtures and run a small demonstration.
./build/knn_tests results
./build/knn_bench results/demo --smoke

# Run our full single-factor CPU experiment.
python3 scripts/environment.py results/reproduction
./build/knn_bench results/reproduction
python3 scripts/analyze.py results/reproduction

# Regenerate the report from the retained results/main records.
python3 scripts/report.py
```

Our full experiment contains 11 distinct parameter combinations, with 10 warm-up calls and 20 measured calls per route. We alternate forward and reverse route order. If a combination has an IQR/median above 10%, we retain its first measurements and perform at most one additional measurement group. During timing, do not run builds, plotting, other benchmarks, or substantial disk activity.

The benchmark binds its single computation thread to guest vCPU 1 and verifies affinity. If that CPU is unavailable, it selects one available CPU for the entire run. Guest CPU affinity does not guarantee a fixed host physical core or frequency.

Our recorded experiments are in `results/main/`: 3,400 complete-search timing records, 660 fixed-distance selection timing records, and 88 route/configuration correctness records. The large generated input arrays are included under `results/main/` with their shapes, generator, seed, and SHA-256 identifiers. Compiled targets are included under `artifacts/`; machine-local snapshots and build-system caches are excluded.

## CPU routes

| Route | Distance computation | Selection |
|---|---|---|
| R0 | Scalar, complete evaluation | Full sort |
| R1 | Tiled scalar | Full sort |
| R2 | Tiled AVX2 | Full sort |
| R3 | Tiled scalar | Max-heap |
| R4 | Tiled AVX2 | Max-heap |
| R5 | Tiled scalar | Bitonic and streaming merge |
| R6 | Tiled AVX2 | Bitonic and streaming merge |
| R7 | Scalar evaluation with exact early abandonment | Max-heap |

Inputs are row-contiguous FP32 vectors with finite components in `[-1,1]`. We return squared Euclidean distances, database row IDs, and original labels. Results are ordered by actual distance and then by ID. Tolerance is used only for verification.

AVX2 accumulates eight lane partial sums, reduces lanes in order, and handles the remaining dimensions with scalar operations. Loads support unaligned inputs and do not read beyond a row. We disable automatic vectorisation in the scalar and SIMD source files while retaining explicit SIMD intrinsics. We disable fast-math, FMA contraction, FTZ, and DAZ, and use round-to-nearest. Scalar and SIMD distances need not be bitwise identical.

Our CPU Bitonic network and streaming merge have replaceable interfaces for integration with the group's shared selection modules. Each tile retains `min(k, valid_count)` candidates; `k>N` is rejected, whereas `k=N` and `k>T` are supported.

## Source and recorded materials

| Location | Purpose |
|---|---|
| `include/knn.hpp` | APIs, candidate order, errors, and verification states |
| `src/scalar.cpp`, `src/simd.cpp` | Controlled distance kernels |
| `src/knn.cpp` | Input validation, search routes, selection, merge, and data generation |
| `src/oracle.cpp` | Independent FP64 oracle and separate correctness checks |
| `src/bench.cpp` | Complete-search and fixed-distance selection experiments |
| `src/export.cpp` | Actual search-result and full-distance export |
| `tests/` | Boundary cases, determinism, errors, and external-verifier tests |
| `scripts/` | Environment capture, verification, plots, and report generation |
| `artifacts/` | Compiled targets from Release, clean, and sanitizer builds |
| `results/main/` | Raw records, statistics, complete input arrays, hashes, and plots |
| `results/fixture_*` | Shared small inputs and complete FP64 reference rankings |
| `results/*instructions.txt` | Scalar and AVX2 instruction evidence |

Complete CPU timing includes state reset, distance computation, candidate management, selection, merging, final ordering, and label gathering. It excludes input generation, full input validation, I/O, the oracle, correctness checks, and initial reusable allocation. Preparation costs are reported separately. Auxiliary memory records actual workspace capacities; it excludes input/output storage, the oracle, and buffers belonging to other routes. Stage diagnostics and selection microbenchmarks do not replace complete-search measurements.

## API example

```cpp
#include "knn.hpp"

knn::Input input = knn::synthetic(1000, 64, 16);
knn::Result result = knn::search(input, {10, 128, 4});

// Keep input validation and allocation outside steady-state timing.
auto verified = knn::validate(input);
knn::Workspace workspace;
knn::Result output;
knn::Config config{10, 128, 4};
knn::prepare(verified, config, workspace, output);
knn::search(verified, config, workspace, output);
```

`Validated` borrows its input. We must keep the `Input` alive and immutable while using that token, and validate again after modifying data, shape, or labels. Ordinary search validates inputs on every call. Failures throw `knn::Error` with an `ErrorCode`; failed output must not be treated as a successful result. Direct calls to low-level distance kernels require validated pointers/dimensions and a prior call to `floating_environment()`.

## Export and independent verification

```bash
# Arguments: INPUT_PREFIX N d q k T route OUTPUT_PREFIX [--full]
./build/knn_export results/main/input_0 100000 64 16 10 128 4 results/cpu_R4 --full
```

Exported `.ids.u32`, `.distances.f32`, and `.labels.i32` arrays are little-endian and query-major. `--full` exports the distances actually used by that calculation form; R7 exports complete distances using its matching scalar rule. See [integration and validation](docs/integration.md) for the independent verifier command.

Our current recorded results cover CPU execution. Real GPU measurements, GEMM comparisons, and CPU/GPU crossover analysis remain pending in our group project.
