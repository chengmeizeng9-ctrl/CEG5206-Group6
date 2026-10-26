# CEG5206 exact-kNN CPU/CUDA study

The required implementations are R0 scalar full sort with an independent FP64
oracle, R1 AVX2 distance with a Top-k heap, R2 materialised CUDA with labelled
CUB segmented full sort, R3 fused tiled CUDA with local Bitonic selection, and
one deterministic hierarchical candidate merge.

CPU Bitonic, radix/scan, GEMM, early abandonment and out-of-core streaming are
optional and are not claimed as implemented.

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

For repeated sessions add `--sessions 3`. Use the direct Python driver on a
fresh machine; the shell wrapper expects this machine's local toolkit layout.
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
