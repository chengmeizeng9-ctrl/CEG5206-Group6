# Retained final experiment

The retained full run is:

`runs/20261026T070351075233Z/`

`full` is a relative symlink to that directory for compatibility with the data
auditor's default path. It does not duplicate the run.

- [Results report](runs/20261026T070351075233Z/analysis/final_results.md)
- [Raw-data audit](runs/20261026T070351075233Z/analysis/data_audit.json)
- [Environment and source provenance](runs/20261026T070351075233Z/environment.json)
- [CPU/GPU correctness table](runs/20261026T070351075233Z/analysis/unified_correctness.csv)

This run completed with all 16 CPU and 25 GPU correctness records passing.
It contains 1,000 GPU timing samples, eight combined plots and clean memcheck /
racecheck logs. One CPU timing group and zero GPU groups exceeded 10%
IQR/median. This is one completed session, not a three-session repeatability
study.

## What belongs in Git

Keep source, tests, plotting/audit scripts, raw timing/correctness/memory CSVs,
reports, plots, run metadata, source hashes, input manifests/hashes, compiler
flag records and validation logs.

The generated `cpu_core/input_*.f32` and `input_*.i32` arrays remain locally
available for exact-bit verification but are ignored by Git. They account for
about 189 MiB and are reproducible from synthetic-v1 / MT19937 seed 5206 using
the CPU benchmark. Each run's `inputs.csv` and `input_sha256.json` are retained
in Git. A fresh checkout must regenerate or retrieve these arrays before
rerunning the input-hash audit; rerunning the study generates them in a new
results directory without changing this retained measurement.

Local `.venv/` and `.tools/` installations are also ignored. See
[the setup instructions](../README.md#setup-on-another-machine) on a fresh machine.

Earlier runs, imported CPU binaries/results, duplicate source archives, old
patch/manifests and build/cache files were moved out of this workspace to a
local recovery directory during repository cleanup. They are not part of this
final experiment's evidence.
