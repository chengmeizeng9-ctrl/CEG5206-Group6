# Our Module Integration and Shared Validation

We use computation contract version 1.1. The single-factor experiment contains 11 combinations: the baseline plus 1 + 2 + 2 + 2 + 3 variations. All specified parameter values are retained.

## Distance, selection, and workspace interfaces

Our C++17 declarations are in `include/knn.hpp`. Matrices are row-contiguous. Shape and capacity arithmetic uses `size_t` with multiplication checks. A candidate contains an FP32 distance and a `uint32_t` row ID; its measured size is 8 bytes. Labels are `int32_t`.

`Better` orders valid candidates first, then by distance, then by ID for exact ties. Padding uses positive infinity and `UINT32_MAX`. Input to selection modules must contain validated finite, non-negative distances, with infinity allowed only for padding. NaN is unsupported. Positive and negative zero compare equally; final output normalises zero to positive zero. Verification tolerance never participates in comparison.

`bitonic(Candidate*, capacity)` sorts a power-of-two capacity in place. Our integrated CPU capacities are T = 32, 64, 128, or 256. We calculate distances only for real samples and fill the remaining slots with invalid candidates. Each tile retains `min(k, valid_count)` candidates; when k exceeds T, all valid candidates in that tile are retained.

`merge_topk(a, na, b, nb, out, k)` accepts two sorted lists and their valid lengths. It writes at most k candidates and returns the actual output length. The output needs k slots and must not overlap either input. We own all input, output, and reusable workspace lifetimes.

Our CPU route streams tile results into a global list rather than storing every tile's candidates. We reset state for each query; the reset is part of complete-search timing. When replacing a shared selection or merge implementation, we first verify fixed-distance selection, then complete search, and finally performance. Results from different source revisions must not be combined without revalidation.

## Shared fixtures and main inputs

`tests/test.cpp` contains our executable correctness cases. We cover N = T-1, T, T+1, and 2T+3 for every T; dimensions 1, 7, 8, 9, and 19; k = 1, 10, N, and values above T; zero, identical, duplicate, exact-tie, and near-tie vectors; multiple queries; repeat and alternating calls; invalid inputs; and early-abandonment boundary conditions.

```bash
./build/knn_tests results
```

This exports two small fixtures. `fixture_0` contains symmetric exact ties and multiple queries. `fixture_1` contains adjacent FP32 values that expose a rounding-related ranking difference. The `.txt` file contains N, d, and q. X, Q, and labels are raw little-endian arrays. `.oracle.csv` gives every query's complete FP64 ranking, IDs, distances, and labels. We can inspect the kth and (k+1)th boundary directly from that table; k=N has no (k+1)th result.

The complete main input arrays are included under `results/main/`. Their shapes are recorded in `results/main/inputs.csv`, and their SHA-256 identifiers are in `results/main/input_sha256.json`. Small fixtures are also included; their hashes are in `results/fixture_sha256.json`. We verify actual bit patterns rather than assuming equal seeds imply equal data across languages.

## Validate CPU or GPU outputs independently

We export three query-major, little-endian q-by-k arrays:

- `<output>.ids.u32`
- `<output>.distances.f32`
- `<output>.labels.i32`

An optional q-by-N FP32 table must contain the complete distances used by that implementation. We complete device execution, transfer, and synchronisation before inspecting GPU output.

```bash
python3 scripts/verify_external.py \
  --input results/main/input_0 --output /path/to/search_result \
  --n 100000 --d 64 --q 16 --k 10 \
  --full-distances /path/to/full_distances.f32 \
  --repeat-output /path/to/repeated_result \
  --report /path/to/verification.json
```

The verifier independently computes FP64 distances and records structure, labels, sorting, distance tolerance, fixed-distance selection, FP64 ID agreement, and determinism. Without a complete distance table, selection correctness is `not_measured`; without repeated output, determinism is `not_measured`. A zero exit code alone does not imply `strict_agreement`: we inspect all reported fields.

For ranking differences, we retain the reference boundary, relevant candidate IDs, and method/reference distances. We investigate reduction order, precision settings, and GEMM reconstruction before explaining a difference. The verifier does not automatically classify an unexplained difference as acceptable.

## Remaining group experiments

We still need real NVIDIA hardware information, thread/warp/direct/GEMM configurations, GPU outputs, and GPU-resident and transfer-inclusive measurements. We also need measured GPU selection/merge resource use and integration results. We will use the same independent validation framework for those outputs and complete our numerical and CPU/GPU crossover analysis. Our CPU timing is not a substitute for GPU measurement.
