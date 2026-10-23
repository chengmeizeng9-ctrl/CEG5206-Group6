# i7-12700H Experiment: Member 1's Original Release Executable

Date: 2026-10-23. Platform: Ubuntu 26.04 / WSL2.

This experiment uses Member 1's original Release executable and follows the original procedure for one main experiment, without additional repetitions of the full experiment.

| Content | Location |
|---|---|
| Local experiment report | [docs/report.md](docs/report.md) |
| Comparison with Member 1 | [docs/comparison.md](docs/comparison.md) |
| Local raw data and plots | [results/main/](results/main/) |
| Execution notes | [docs/execution_notes.md](docs/execution_notes.md) |
| Member 1's original implementation and results | [../CPU/](../CPU/) |

The shared executable and input arrays remain in the original `CPU/` directory. This folder contains the local experiment records and comparison. From the repository root, run `cd CPU`, then `./artifacts/release/knn_bench results/new_run` and `python3 scripts/analyze.py results/new_run` to reproduce into a fresh directory.
