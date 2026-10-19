# CPU Interfaces and Correctness

We follow computation contract version 1.1. The required CPU implementation exposes row-contiguous FP32 inputs, signed 32-bit labels, unsigned 32-bit row IDs, and FP32 squared distances.

## Candidate ordering and merge interface

`Better` orders valid candidates first, then by ascending distance, then by ascending ID for exact ties. Padding uses positive infinity and `UINT32_MAX`; padding is never returned. Verification tolerance does not participate in ordering.

`merge_topk(a, na, b, nb, out, k)` accepts two already sorted candidate lists and their valid lengths. It writes at most k candidates in total order and returns the actual output length. The output capacity must be at least k and must not overlap either input. The helper is tested with ties, unequal inputs, empty inputs, and truncation, and its result is compared with sorting the concatenated lists.

## Independent correctness checks

`tests/test.cpp` exercises scalar full sort and AVX2 Top-K heap over zero, identical, duplicate, exact-tie, near-tie, multi-query, and AVX2-tail cases. It also compares the fixed-distance heap selector with full sorting and checks the independent FP64 oracle.

The FP64 reference promotes the actual FP32 input values and computes complete squared distances without using production distance or selection functions. It records the full ranking for small fixtures, including the kth and (k+1)th boundary when k<N. The route verifier separately reports output structure, labels, ordering, fixed-distance selection, distance tolerance, FP64 ID agreement, and determinism.

## Main input data

The complete main input arrays are retained under `results/main/`. Their shapes are in `inputs.csv`, with SHA-256 hashes in `input_sha256.json`. The small fixtures and their oracle rankings are also included.
