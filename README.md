# CEG5206 Group 6 — Exact kNN on CPUs and GPUs

This project compares CPU and CUDA implementations of exact k-nearest-neighbour search. Every route evaluates all database vectors and returns the nearest `k` results, ordered by squared Euclidean distance and then sample ID, with their original labels.

The complete CPU/GPU implementation and retained experiment are on the **`full-pipeline` branch**.

## Required implementations

| Route | Implementation |
|---|---|
| R0 — CPU baseline | Scalar FP32 distance computation and full sorting |
| R1 — CPU optimisation | AVX2 SIMD distance computation and a Top-K max-heap |
| R2 — CUDA baseline | Materialised distance table and CUB segmented radix full sorting |
| R3 — CUDA optimisation | Fused tiled distance computation, local Bitonic selection and hierarchical candidate merging |

An independent **FP64 oracle** checks numerical accuracy and neighbour IDs. Candidate merging is also tested against concatenation followed by full sorting. CUDA routes support both thread-per-vector and warp-per-vector distance mappings.

## Results and validation

The retained full experiment was completed on **26 October 2026**:

- [Results report and figures](results/runs/20261026T070351075233Z/analysis/final_results.md)
- [CPU/GPU correctness records](results/runs/20261026T070351075233Z/analysis/unified_correctness.csv)
- [Hardware, software and run metadata](results/runs/20261026T070351075233Z/environment.json)
- [Raw data and validation logs](results/runs/20261026T070351075233Z/)
- [Results index and data-retention notes](results/README.md)

The retained records contain 16 passing CPU and 25 passing GPU correctness records. GPU tests passed 44 cases and 2,294 checks; Compute Sanitizer reported no memory errors or race hazards.

GPU-resident and transfer-inclusive timings are reported separately. This is **one completed measurement session**; a three-session repeatability study has not been completed. Near-tie FP64 ranking differences are recorded separately from FP32 selection correctness.

## Build and reproduce

Use an x86 CPU with AVX2, a compatible NVIDIA GPU, and the CUDA toolchain. The retained experiment targets the GTX 1050 (`sm_61`) with CUDA 12.x.

On a fresh machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt

# Build, validate and run a small integration experiment.
.venv/bin/python experiments/run_full_experiment.py --repo-root . --smoke

# Run the full experiment from a clean build.
.venv/bin/python experiments/run_full_experiment.py --repo-root . --clean-build
```

Put `nvcc` and `compute-sanitizer` on PATH and make the CUDA runtime available. See [setup and reproduction details](KNN_STUDY.md#setup-on-another-machine) for toolchain requirements, the local wrapper and repeated sessions.

## Repository guide

| Location | Contents |
|---|---|
| [CPU/](CPU/) | Current scalar baseline, FP64 oracle, AVX2/heap implementation and CPU tests |
| [GPU/](GPU/) | Materialised and fused CUDA routes, candidate merging and GPU tests |
| [experiments/](experiments/) | Experiment runner, analysis, report generation and data auditing |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | Workloads, correctness checks, timing boundaries and experiment design |
| [KNN_STUDY.md](KNN_STUDY.md) | Detailed setup, reproduction and reporting instructions |
| [results/](results/) | Retained CPU/GPU experiment and evidence |

## Earlier CPU experiments

The following folders preserve earlier i7-12700H experiments separately from the current CPU/GPU comparison:

- [Original Release executable experiment](CPU_i7_12700H_OriginalRelease/)
- [Local source-build experiment](CPU_i7_12700H_LocalBuild/)
- [Chinese comparison report for the three earlier CPU experiments](docs/CEG5206三组CPU实验区别与完整结果对比.docx)

CPU Bitonic, custom Radix Select, GEMM-based distance, early abandonment and out-of-core streaming are outside the current required implementation and are not claimed as completed.
