# CPU Project Rules

Version 1.2, 19 October 2026

These CPU-specific rules describe only the required CPU work: a scalar full-sort baseline with an independent FP64 oracle, AVX2 distance computation with a Top-K max-heap, and one correct candidate-list merging method. The shared contract governs common input and numerical conventions; the specified CPU task list governs which work is included here.

## Inputs and computation

X and Q are contiguous row-major FP32 arrays with shapes N-by-d and q-by-d. Labels are signed 32-bit values, one per database row; IDs are unsigned 32-bit database row indices. N, d, and q must be positive, 1 <= k <= N, and all size calculations must check overflow. Accept finite FP32 components in [-1, 1] and reject non-finite or out-of-domain values. Inputs and labels are read-only during search, and labels do not affect ranking.

Compute squared Euclidean distance without a square root. The scalar route subtracts, squares, and accumulates in FP32 in increasing dimension order. Disable fast-math, FMA contraction, automatic vectorisation, and floating-point reassociation for the controlled scalar reference. Use round-to-nearest and disable FTZ/DAZ.

The AVX2 route uses eight FP32 lanes, a documented lane-reduction order, and scalar handling for the remaining dimensions. Unaligned input must be safe, and loads must not pass a row boundary. Restrict AVX2 targeting to the SIMD implementation so the scalar baseline remains usable without AVX2. Record compiler options and representative instruction evidence.

## Required CPU methods

The baseline computes every scalar distance, stores all N candidates, sorts by the shared total order, and returns the first k.

The SIMD route computes each distance with AVX2 and maintains a max-heap of at most k candidates. The heap root is the worst retained candidate. Replace it only when a candidate ranks ahead of the root, then sort the retained candidates before output. Reset heap state for every query.

The candidate merge method accepts two already sorted candidate lists and their valid lengths, emits the first min(k, na+nb) candidates in total order, and handles ties by ID. Keep this method independently callable and test it against sorting the concatenated lists. Its output buffer must not overlap either input.

Candidates rank by ascending distance and then ascending ID for exact ties. Verification tolerance never changes ranking. Treat positive and negative zero as equal and normalize returned zero to positive zero. Return exactly k distinct valid IDs, their complete route-specific distances, and their original labels in query-major order.

## FP64 reference and correctness

Construct an independent oracle from the actual FP32 input values, promote those values to FP64, calculate complete squared distances, and sort without calling production distance or selection functions. For fixed candidate distances, the Top-K heap must match a full sort exactly in ID and order. Test candidate merging independently against a full sort of both input lists combined.

Use atol=1e-5 and rtol=1e-4 for route distance checks: absolute error must be at most atol + rtol times the absolute FP64 reference distance. Keep structure, labels, sorting, fixed-distance selection, distance tolerance, FP64 ID agreement, and repeat determinism as separate results. Explain ranking differences at numerical boundaries; do not describe them as strict agreement.

Cover zero and identical vectors, duplicate rows, exact and near ties, k=1 and k=N, partial AVX2 vectors, dimensions around the eight-lane boundary, multiple queries, repeated calls, invalid shapes and values, and merge inputs with ties and unequal lengths.

## Reproducible inputs and measurements

The retained main data uses synthetic-v1: standard 32-bit MT19937 with seed 5206. For each random value, take the high 24 bits, divide by 2^24, and map to [-1,1) as FP32. Restart the seed per configuration, generate X followed by Q, and set labels to row ID modulo 10. Record shapes, generator version, little-endian arrays, and SHA-256 hashes. All methods for a configuration use identical input bits.

Use the baseline N=100000, d=64, q=16, k=10 and single-factor scans over N={10000,100000}, d={32,64,128}, q={1,16,64}, and k={5,10,32}, deduplicated to eight configurations. Do not include tile-size scans, since the retained required routes do not use a tile-size parameter.

Use one CPU thread, a monotonic clock, 10 warm-ups, and 20 measured complete-search calls per route. Retain every raw sample. Complete-search time includes required state reset, distance computation, candidate management, sorting or heap updates, output ordering, and label gathering. It excludes input generation, full validation, file I/O, the oracle, correctness checks, and initial reusable allocation; report preparation separately.

Alternate route order by round. For 20 sorted measurements, use linear interpolation at position 19p for p=0.25, 0.5, and 0.75; IQR=Q3-Q1. Mark IQR/median above 10% as variable and allow at most one additional group, retaining both groups. Do not hide variability or claim physical-hardware peak performance from virtual-machine measurements.

Report input, output, and auxiliary workspace separately. Retain raw timings, summaries, correctness records, input hashes, plotting and report scripts, build instructions, and the compact environment summary. Distinguish “not applicable” from “not measured”; do not substitute zero. After a source change, regenerate affected correctness and timing records from the same source revision.
