# Our CPU Implementation Summary

Date: 2026-10-09. This document summarises the CPU portion of our group project. Recorded source hashes, compiler settings, inputs, and results provide the reproducibility evidence.

| Project work | Our implementation and evidence |
|---|---|
| Task 1: Reference and correctness | Scalar full sort; independent FP64 distances/ranking; separate validation checks; shared fixtures and boundary tests |
| Task 2: CPU tiling and SIMD | R0-R6 comparisons; four tile sizes; AVX2 with eight lanes, unaligned loads, scalar tails, and instruction evidence |
| Task 2: Selection integration | Max-heap and final sorting; replaceable Bitonic network and streaming merge; fixed-distance selection validation |
| Task 7: Early abandonment | Matching scalar heap rule; strict partial-distance threshold; mean-dimension/abandonment diagnostics; R3/R7 ablation |
| Task 8: CPU experiments | Eleven single-factor combinations; warm-ups and repeated measurements; alternating route order; IQR and retained reruns; plots and analysis |
| Reproducibility | CMake; Release, sanitizer, and clean-build checks; CPU affinity; raw records; hashes; build flags and environment |
| Group integration materials | Independent CPU/GPU output verifier; shared interfaces and fixtures; technical report and demonstration guide |

Our replaceable CPU selection and benchmark modules support integration with the rest of the group project. This summary describes implemented work, rather than assigning individual authorship or duplicating credit for shared modules.

GPU kernels, real GPU measurements, GEMM comparisons, and CPU/GPU crossover conclusions are not included in our current CPU results. We have prepared validation interfaces and reference data for those experiments.
