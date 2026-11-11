# CEG5206 exact-kNN CPU/CUDA study

The required implementations are R0 scalar full sort with an independent FP64
oracle, R1 AVX2 distance with a Top-k heap, R2 materialised CUDA with labelled
CUB segmented full sort, R3 fused tiled CUDA with local Bitonic selection, and
one deterministic hierarchical candidate merge.

All five optional methods are also implemented: CPU AVX2 tiled Bitonic,
CUDA radix selection with exclusive scan/compaction, a cuBLAS GEMM reference,
scalar early abandonment, and single/double-buffered out-of-core streaming.
They use a separate experiment so their records stay distinct from the required
R0–R3 comparison.

## Reproduce on this machine

The local `.venv/` and CUDA 12.8 toolkit in `.tools/cuda-12.8/` are configured.
The wrapper selects them automatically.

```bash
# Preview without compiling, testing or measuring.
./run_experiment.sh --dry-run

# Build, correctness tests, sanitizer checks and small integration experiment.
./run_experiment.sh --smoke

# Reproduce one full eight-case experiment.
./run_experiment.sh --clean-build

# Optional: three separate sessions and a combined repeatability report.
./run_experiment.sh --sessions 3 --clean-build
```

Use AC power and consistent power settings; close unrelated workloads.
Every invocation creates a new UTC-timestamped directory. A supplied
`--results-dir` must be new or empty; existing measurements are never erased.
Build products are regenerated automatically.

## Run all optional methods

Reuse the exact CPU-exported inputs from a completed required run:

```bash
# Preview all optional routes without running them.
./run_experiment.sh --optional --dry-run

# Build and validate without collecting timing samples.
./run_experiment.sh --optional --inputs results/full/cpu_core --validate-only

# Short integration measurement, including fresh correctness/sanitizer gates.
./run_experiment.sh --optional --inputs results/full/cpu_core --smoke

# Full optional study, with contemporaneous CPU/GPU controls.
./run_experiment.sh --optional --inputs results/full/cpu_core --clean-build
```

Optional results go to `results/optional_runs/<timestamp>/`. Start with
`analysis/optional_results.md`; it includes raw-data validation, controlled
speedups, CPU dimension counts, GEMM/radix stages, buffer allocation and
streaming comparisons. Raw CSVs and test/sanitizer logs are retained. Required
and optional performance records are never silently combined.

| Optional method | Implementation and correctness control |
|---|---|
| CPU Bitonic | AVX2 distances, scalar local Bitonic networks, packed candidate lists and hierarchical merges; bitwise comparison with R1 |
| Early abandonment | Scalar partial-sum pruning only when `sum > threshold`; same heap path with full evaluation (C0) supplies the timing control |
| Radix + scan | Eight histogram threshold refinements, self-written hierarchical exclusive Blelloch scan, compaction, and sorting only the retained k; exact same-mapping R2 comparison |
| GEMM reference | Labelled cuBLAS pedantic FP32 GEMM, separate norm/reconstruction costs and near-zero direct rechecks; full-sort validation of its own distances |
| Out-of-core | One/two pinned host/device chunk slots, asynchronous H2D and one compute stream, global-ID candidate merging; single/double/R2 agreement |

The current local toolkit includes cuBLAS 12.8.4.1. If recreating the minimal
local toolkit, install its optional shared libraries with:

```bash
.venv/bin/python experiments/install_optional_cublas.py
```

This downloads NVIDIA's official approximately 939 MB archive, verifies its
SHA-256 and installs headers/shared libraries under the ignored `.tools/`
directory. A normal complete CUDA 12.x installation can supply cuBLAS directly.

`--mapping thread|warp`, `--tile 32|64|128|256`, and `--chunk-size <rows>`
control the optional study. Streaming always includes transfers and host label
gathering; it has no resident-only timing. The overlap ratio is a service-time
estimate, not a profiler trace. Its device allocation depends on the chunk
budget, while the complete input must fit host RAM. To measure inputs larger
than available device memory, supply their input manifest and arrays and use:

```bash
./run_experiment.sh --optional --inputs /path/to/cpu-export \
  --stream-only --chunk-size 32768
```

Stream-only mode avoids materialised full-N GPU controls. A small chunk-budget
run validates bounded-memory processing; it does not by itself prove that the
tested dataset exceeded physical GPU memory. CUDA optional routes support
`k <= 128`, with streaming additionally requiring `k <= tile`; CPU Bitonic
supports `k=N`, including k larger than one tile.

The current optional implementation has build/unit/sanitizer evidence under
`results/optional_validation/`. Its new performance measurements are left for
the user to run; the retained October required-run CSVs are historical results
from their recorded source version.

## Setup on another machine

Use an x86 CPU with AVX2 and a CUDA 12.x toolkit supporting the GTX 1050
(`sm_61`). Put `nvcc` and `compute-sanitizer` on PATH and make the CUDA runtime
available. CUDA 13 cannot compile this Pascal target. Its version shown by
`nvidia-smi` is driver capability, separate from the installed compiler.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/run_full_experiment.py --repo-root . --smoke
.venv/bin/python experiments/run_full_experiment.py --repo-root . --clean-build
```

For repeated required sessions add `--sessions 3`. For optional routes use
`.venv/bin/python experiments/run_optional_experiment.py --inputs <CPU_RESULTS>`.
The optional build requires cuBLAS headers and shared libraries. Use the direct
Python driver on a fresh machine; the shell wrapper expects this machine's
local toolkit layout.
Local dependencies and build output are excluded by `.gitignore`.

## Measurement and validation

The driver builds CPU/GPU modules and requires actual CPU/GPU test execution,
then runs memcheck and racecheck before measurement. The full grid varies
N, d, q and k one factor at a time. GPU mappings and four tile sizes are
measured in separate baseline ablations. Default measurements use ten
warm-ups and twenty timed calls; extra CPU timing groups are retained.

GPU-resident CUDA-event latency and transfer-inclusive host wall time are
reported separately. R3 distance and local Bitonic selection are timed as a
combined stage. Source hashes, compiler flags, versions, hardware metadata,
correctness records, raw samples and allocated-buffer accounting are retained.
The full-run audit checks input hashes, statistics, stages and plots.

Exact selection follows each route's declared FP32 arithmetic and the
(distance, ID) tie order. Numerical tolerance is
`abs(d32-d64) <= 1e-5 + 1e-4*abs(d64)`. FP64 ID differences are reported
separately; near ties can differ after FP32 rounding. Same-mapping R2/R3
results must match IDs, distance bits and labels exactly.

Both CPU and GPU groups with IQR/median above 10% are flagged. Headline CPU
comparisons use the predeclared group 0, with other groups retained. R0/R1
measures SIMD and heap selection together, not SIMD alone. Temporary buffer
allocation is not a measurement of DRAM traffic. The fastest measured tile
is specific to the tested workload and machine.

With multiple sessions, the first mapping is frozen throughout the study;
input/source hashes and hardware/configuration must match. Sanitizer evidence
is reused only within that unchanged study; tests and audits run per session.
The summary uses paired per-session speedups and ranges of session medians,
not pooled samples or confidence intervals. Consecutive sessions do not
establish statistical independence.

## Results

The completed full run is indexed in [results/README.md](results/README.md).
It is one completed measurement session, not a three-session study.

New outputs are written to:

- `results/smoke_runs/<timestamp>/`: smoke integration reports.
- `results/runs/<timestamp>/`: one full session.
- `results/studies/<timestamp>/`: repeated sessions and `study_summary.md`.

Each full session contains `environment.json`, `source_sha256.json`,
`logs/`, CPU/GPU raw-data directories, and `analysis/` with the results report,
eight plots, unified CSVs and `data_audit.json`. On failure, inspect
`environment.json` and `logs/workflow.log`, then rerun into a new directory.
Incomplete or incompatible sessions cannot enter the study summary.

Generated binary input arrays remain locally available but are ignored by
Git. Their shapes, generator seed and SHA-256 hashes are committed. A fresh
checkout must regenerate or retrieve these arrays before auditing their
hashes; reproducing the study creates them in its new results directory.

```bash
# Audit the retained local inputs and records without measuring again.
.venv/bin/python experiments/verify_experiments.py \
  --results results/full --output /tmp/knn-data-audit.json

# Regenerate a completed repeated-study summary.
.venv/bin/python experiments/summarize_sessions.py --study results/my-study

# Offline workflow tests; no CPU/GPU benchmark execution.
.venv/bin/python -m unittest discover -s experiments/tests -v
```

## Design and reporting

The [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) specifies the grid,
correctness gates, timing boundaries and work/storage hypotheses. The
[CPU guide](CPU/CPU_Guide.md) describes the standalone CPU build and merge API.
Original and revised proposals are in [docs/proposals/](docs/proposals/).

Use your generated report to explain complete latency, stage costs,
transfer overhead, candidate-buffer storage and tile/merge trade-offs. Keep
all noise flags and limit conclusions to the measured machine and synthetic
inputs. Add the actual team names, contributions and cross-review evidence.

The CPU source originated from team commit
`3198b94f6cecbbfa4daaac9d9acd156a19d7eb01`; current measured source hashes and
available Git metadata are retained with the completed run.
