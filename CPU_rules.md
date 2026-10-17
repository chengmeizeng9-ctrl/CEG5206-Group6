# CPU Project Rules

Version 1.1, 9 October 2026

These rules describe our CPU implementation and experiments. The group computation contract remains the shared authority where applicable. The values below distinguish project requirements from environment observations; machine-specific measurements describe the recorded guest and do not establish physical-host characteristics.

## Scope and project decisions

We implement and evaluate exact k-nearest-neighbour search on CPUs. The CPU work includes scalar and FP64 reference routes, tiled scalar and AVX2 distance computation, full sort, max-heap selection, CPU Bitonic selection and merge integration, exact scalar early abandonment, correctness validation, and CPU performance analysis.

The project supports the 11 required single-factor parameter combinations. Shared changes to input generation, ordering, tolerance, or timing definitions must be reflected in the group contract and communicated to the group. The computation contract and execution rules used by the retained results are version 1.1.

## Parameters and data representation

| Name | Meaning |
|---|---|
| N | Database size, 1 to 4,294,967,295 |
| d | Vector dimension, 1 to 2,147,483,647 |
| q | Number of queries, at least 1, subject to capacity and memory limits |
| k | Number of results per query, 1 to N |
| T | Tile size: 32, 64, 128, or 256 |
| X | Row-contiguous N-by-d FP32 database matrix |
| Q | Row-contiguous q-by-d FP32 query matrix |
| ID | Unsigned 32-bit database row index, from 0 to N-1 |
| label | Signed 32-bit label array of length N |
| distance | FP32 squared Euclidean distance returned by a route |

Size, index, and byte calculations use at least 64-bit unsigned arithmetic. Check multiplication, addition, allocation sizes, and conversions to external integer types for overflow. These representation limits are not promises that every size can be allocated. Core inputs contain finite FP32 values in [-1, 1]. Reject NaN, infinity, and values outside that domain. Inputs and labels are read-only during search. Labels do not affect distance or selection.

## Recorded CPU environment

The recorded environment is an openEuler 24.03 LTS x86_64 little-endian VMware guest, with kernel 6.6.0-28.0.0.34.oe2403.x86_64. The guest reports an Intel Core Ultra 5 125H, two online vCPUs numbered 0 and 1, and AVX2, AVX, SSE, and FMA support; it does not expose AVX-512. These guest reports do not identify the host physical core or guarantee host frequency.

The recorded toolchain is GCC/G++ 12.3.1, CMake 3.27.9, Python 3.11.6, NumPy 2.4.6, and Matplotlib 3.11.1. Record the actual environment again before a new formal experiment. Host frequency and physical-core type are not observable from this guest. Do not claim physical CPU peak performance from these measurements.

## Input validation and errors

All search routes use the same input conditions. Do not silently truncate k, pad missing input data, modify input, or return partial results after failure. Distinguish invalid arguments, invalid shapes, invalid values, size overflow, resource exhaustion, unsupported configuration, and execution failure. Include the parameter or failed stage in the error report.

Ordinary calls validate all input before searching. For controlled steady-state benchmarks, validate once before timing and reuse the immutable validated input. State clearly when full validation is outside the timed region.

## Distance and floating-point rules

For each query and database row, compute the sum of squared component differences without a square root, normalization, or approximate index. The scalar reference and tiled scalar routes subtract, square, and accumulate in FP32 in increasing dimension order. Disable implicit FMA contraction, automatic vectorisation, and floating-point reassociation in these controlled routes. Use round-to-nearest and disable FTZ/DAZ for the recorded CPU comparison.

The AVX2 route processes dimensions in eight FP32 lanes, reduces the lane partial sums in a documented order, and handles the remaining dimensions with scalar operations. Support unaligned input safely and never load beyond a row. Restrict AVX2 compiler targeting to the SIMD module so scalar routes remain usable without AVX2. Record compiler options and representative disassembly; CPU feature detection alone is not evidence that SIMD was implemented.

Different reduction orders need not produce bitwise-identical distances. Compare fixed-distance selection separately from route-specific numerical results. Never use verification tolerance to alter candidate ordering.

## Candidate order and output

Order candidates by validity first, then ascending distance, then ascending ID for exact distance ties. Positive and negative zero compare equally; normalize returned zero to positive zero. Padding uses positive infinity and ID 4,294,967,295, which is never a valid database row. Carry the valid count for a partial tile and never return padding.

Return exactly k distinct valid IDs for every query, with the corresponding complete distance used by that route and the original label. Store outputs as contiguous query-major q-by-k arrays. Sort the final output; do not return heap-internal order. If a route's distance differs near a ranking boundary, retain and explain the difference rather than changing tie rules.

## CPU selection, tiling, and early abandonment

For a partial tile, compute distances only for valid rows and pad the remaining selection capacity. Each tile keeps min(k, valid_count) candidates. Support k greater than T by retaining every valid candidate from that tile; support k=N.

Full sort evaluates and sorts all candidates. The max-heap keeps the worst retained candidate at its root and sorts its final output. CPU Bitonic selection and candidate merging are part of complete-search timing. Keep workspace capacity separate from the number of valid entries and reset required state for every query.

The core early-abandonment route uses the matching scalar accumulation rule with a max-heap. After the heap is full, stop evaluating a candidate only when its partial distance is strictly greater than the current worst retained distance, tau. Equality must continue. Never return a partial distance. This route must match the corresponding unpruned scalar route exactly. SIMD early abandonment is outside the core route unless a safe bound and independent validation are supplied.

## Independent reference and correctness

Build an independent FP64 oracle from the actual FP32 input values. It computes complete distances and ranks results without calling production distance or selection kernels. For small cases, retain all reference distances, IDs, labels, and the kth and (k+1)th boundary when k<N.

For fixed distances, sort, heap, and Bitonic selection must return identical IDs in identical order. Use absolute tolerance 1e-5 and relative tolerance 1e-4 for distance checks; a distance passes when absolute error is at most atol + rtol times the absolute FP64 reference. Apply the same thresholds to all core routes. Do not silently relax them.

Record output structure, labels, ordering, fixed-distance selection, distance tolerance, FP64 ID agreement, and repeat determinism separately. Explain numerical boundary differences explicitly; an explained difference is not strict agreement. Unexplained differences or structural and selection errors cannot support an acceleration claim.

Cover zero and identical vectors, duplicate rows, exact and near ties, k=1, k=N, k>T, invalid k, partial tiles, AVX2 tails, multiple queries, repeated and alternating calls, invalid input, and early-abandonment partial sums equal to or greater than tau.

## Fixed experimental inputs

The main dataset uses synthetic-v1. It uses the standard 32-bit MT19937 sequence with seed 5206. For each value, take the high 24 bits, divide by 2^24, then map to [-1, 1) as FP32. Restart from the same seed for each parameter combination; generate all rows of X followed by all rows of Q. Labels equal row ID modulo 10.

Use identical logical input across routes within each configuration. Record dimensions, seed, generator version, little-endian input bit patterns, and SHA-256 hashes. Additional distributions or seeds must be named separately and cannot replace the main experiment.

The baseline is N=100000, d=64, q=16, k=10, T=128. The single-factor scans are N={10000,100000}, d={32,64,128}, q={1,16,64}, k={5,10,32}, and T={32,64,128,256}, deduplicated to 11 configurations. The CPU core experiment is single-threaded; do not run the full Cartesian product as a required task.

## CPU timing and statistics

Use 10 warm-up calls and 20 measured calls per route, retaining every raw measurement. Complete-search timing starts with required per-call state reset and ends when sorted IDs, distances, and labels are ready. Include distance computation, candidate management, local selection, merging, final ordering, and label gathering. Exclude input generation, full validation, file I/O, the oracle, correctness checks, and initial reusable allocation. Report preparation separately.

Use a monotonic clock. Run one benchmark process at a time. Bind the single computation thread to guest vCPU 1 and verify affinity; if unavailable, select another available vCPU for the entire group and record it. Guest affinity does not bind a host physical core. Avoid builds, plotting, other benchmarks, and heavy disk activity during formal measurements. Record observable environment changes and state unavailable guest or host properties as not observable.

Alternate route order by round, forward then reverse, to reduce fixed-order effects. For 20 sorted measurements, calculate quartiles by linear interpolation at position 19p, where p is 0.25, 0.5, or 0.75. IQR is Q3-Q1. Mark IQR/median above 10% as variable. At most one additional group may be measured for a configuration; retain both groups and all raw data. Persistent variability limits performance claims and must not be hidden by selecting only the faster group.

For very short stages below stable timer resolution, repeat the complete call within a measurement window and divide elapsed time by the repetition count. Use the same count across routes and record it. Reset state on every call. Keep microbenchmarks separate from complete-search results. Do not flush caches as part of the core route.

Throughput is q divided by median batch time. Median batch time divided by q is amortized query cost, not single-query latency. Compute speedup only from matching parameters, inputs, and timing definitions. Stage diagnostics and selection-only microbenchmarks do not replace complete-search measurements.

## Memory and records

Report input, output, and auxiliary workspace separately. Record actual allocation capacities and peak live workspace; do not equate allocated bytes with memory traffic. The logical FP32 distance plus 32-bit ID candidate payload is 8 bytes, but measure and report the actual structure and capacity.

Each result record should identify contract version, route, input hash, N/d/q/k/T, seed, source revision, CPU environment, operating system, compiler and flags, ISA and FMA configuration, thread count, timing definition, raw samples, median and quartiles, throughput, auxiliary memory, correctness fields, and numerical diagnostics. Use “not applicable” for nonexistent stages and “not measured” for unmeasured values; never substitute zero. After a fix, rerun affected validation and experiments and do not combine correctness from one source revision with timing from another.

## Team responsibilities and deliverables

CPU responsibilities include the correctness reference, scalar and FP64 routes, CPU tiling and SIMD, full sort and heap, scalar early abandonment, CPU analysis, and integration of the CPU Bitonic and merge modules. Testing, interpretation, report preparation, demonstration, and contribution records are shared group work.

The CPU deliverable includes source code, build and run instructions, reproducible inputs and hashes, raw results, plotting scripts, relevant environment details, representative commands, a technical report, demonstration materials, and contribution records.
