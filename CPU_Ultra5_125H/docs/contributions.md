# Required CPU Work Summary

Our retained CPU work covers the scalar full-sort baseline, an independent FP64 oracle, AVX2 distance computation with a Top-K max-heap, and one sorted candidate-list merge method. We include the shared inputs, correctness checks, raw CPU measurements, plots, reproducibility commands, and environment summary needed to review those methods.

| Required item | Implementation and evidence |
|---|---|
| Scalar baseline and FP64 oracle | Complete scalar distances, full-sort ranking, independent FP64 ranking, correctness records, and boundary fixtures |
| SIMD distance with Top-K heap | AVX2 eight-lane distance calculation, scalar tail, max-heap selection, and instruction evidence |
| Candidate merging | `merge_topk` merges two sorted lists with deterministic ID tie-breaking; unit tests compare it with a full sort |
| CPU evaluation | Eight single-factor configurations, raw timing repetitions, summaries, and plots |

This summary covers the required CPU work and its historical Ultra 5 125H experiment. The maintained CPU source also includes optional Bitonic and early-abandonment routes and a scalar heap control; these extensions are outside this required-work summary and the historical timing records. Shared testing and report review remain group work.
