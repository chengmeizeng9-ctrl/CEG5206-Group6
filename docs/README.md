# Project Documentation

Start with the experiment protocol for the implemented study, then use the
reproduction guide and retained results to inspect how it was run.

| Document | Purpose |
|---|---|
| [Experiment protocol](EXPERIMENT_PROTOCOL.md) | Required and optional routes, shared inputs, correctness gates, timing boundaries and experiment design |
| [Reproduction guide](../KNN_STUDY.md) | Setup, execution, data auditing and repeated-session instructions |
| [Required experiment results](../results/runs/20261026T070351075233Z/analysis/final_results.md) | Retained R0–R3 measurements and interpretation |
| [Optional experiment results](../results/optional_runs/20261111T062816975198Z/analysis/optional_results.md) | Retained O1–O5 measurements with matched controls |

## Historical CPU Comparison

[CPU Experiment Comparison](CPU_Experiment_Comparison.docx)
([download Word document](CPU_Experiment_Comparison.docx?raw=true)) compares
three earlier records: the original CPU experiment, the original executable
run on an i7-12700H, and a local rebuild on that machine. It contains all eight
configurations and both CPU methods, timing variability, correctness and
build provenance. The English report retains the original numerical results
and conclusions from 23 October 2026.

The supporting records are in [CPU/results/main](../CPU/results/main/),
[OriginalRelease](../CPU_i7_12700H_OriginalRelease/) and
[LocalBuild](../CPU_i7_12700H_LocalBuild/).

## Project Proposals

The [proposal index](proposals/README.md) identifies the original v2 proposal
and revised v3 proposal. Both are retained as planning history; the revised
proposal records the five required items and separates the optional work.
