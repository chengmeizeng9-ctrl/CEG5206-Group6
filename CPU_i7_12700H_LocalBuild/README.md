# i7-12700H Experiment: Local Build

Date: 2026-10-23. Ubuntu 26.04 / WSL2. Build compiler: GCC 12.5.0. Configuration: Release.

This experiment builds Member 1's unchanged source code locally and follows the original procedure for one main experiment, without additional repetitions of the full experiment.

## Historical experiment snapshot

The source, retained Release binaries and measurements in this folder belong
to the experiment on **23 October 2026**, using source revision
`063769a4d6b04f6f572aa01d119c3092e4e5577d`. They are preserved together so
that the local rebuild can be inspected and reproduced.

This is a historical source snapshot. Later algorithm additions in the
current [CPU module](../CPU/) are not copied into it. Some files remain
identical to the current module, but others differ; keeping this complete
version avoids mixing source revisions when reproducing the old experiment.
Documentation and links may still be corrected without changing the
recorded source, binaries or measurements.

The input arrays are shared with [the original CPU dataset](../CPU/results/main/);
their locations and hashes are recorded in
[input_location.json](results/main/input_location.json). This folder does not
contain a second copy of those arrays.

| Content | Location |
|---|---|
| Local experiment report | [docs/report.md](docs/report.md) |
| Comparison with Member 1 | [docs/comparison.md](docs/comparison.md) |
| Raw results and plots | [results/main/](results/main/) |
| Build and run instructions | [docs/build.md](docs/build.md) |
| Build provenance and compiler options | [results/build_provenance.json](results/build_provenance.json) |
| Execution notes | [docs/execution_notes.md](docs/execution_notes.md) |

Source revision: `063769a4d6b04f6f572aa01d119c3092e4e5577d`. The retained Release executable was compiled on this computer. Member 1's original Release experiment is a separate record under `../CPU_i7_12700H_OriginalRelease/`.
