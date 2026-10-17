# Three-Member Shared Contract

Version: 1.1  
Date: 2026-10-09  
Project: Algorithmic and Memory Optimisation of Exact kNN Search on CPUs and GPUs.

## 1. Scope and Relationship to Other Documents

This document is the sole shared contract that all three members must follow. It is based on `GP6_Project Proposal.pdf` and the user's authorization. Mandatory tasks in the proposal must not be omitted because of this document.

The proposal defines distances, ties, mandatory algorithms, and deliverables. The supported input domain, data generation method, tolerances, label type, and measurement counts defined here are team choices and must be explained in the report.

Changes to this contract must update the version, explain the reason and affected modules, and notify all three members. When changes affect numerical results or measurements, rerun the relevant data. Results produced under different contract versions must not be combined.

## 2. Inputs and Parameters

| Item | Shared specification |
|---|---|
| Database X | N×d, FP32, contiguous row-major layout, no implicit padding |
| Queries Q | q×d, FP32, contiguous row-major layout, no implicit padding |
| Labels | N signed 32-bit integers, corresponding one-to-one to database rows |
| IDs | Unsigned 32-bit database row indices, from 0 to N−1 |
| N, d, q | Each at least 1; check representable ranges, lengths, and resources |
| k | 1≤k≤N; no silent truncation |
| T | 32, 64, 128, or 256 for blocked approaches; not applicable for non-blocked approaches |
| Supported numerical domain | Finite FP32 components in [−1,1] |

The maximum ID, 4294967295, is reserved as an invalid sentinel, so N must not exceed 4294967295; d must not exceed 2147483647. These are representation limits, not guarantees that every size can be executed. Size, index, and byte calculations must use at least 64-bit representations, with overflow checks for multiplication, addition, and conversions to third-party interface types.

The proposal's main experimental parameters and reasonable size-boundary tests must be supported, especially small cases with k=N and k>T. Outside the core scope, explicit resource-exhaustion or unsupported-configuration errors are allowed, but must not be used to avoid core tasks.

Inputs are read-only. Labels do not participate in distance computation or selection, and no classification voting is added. Duplicate vectors are not deduplicated. Matrix and label lengths must match their declarations. NaN, infinity, out-of-domain values, empty inputs, and invalid parameters must be explicitly rejected.

The shared error categories are INVALID_ARGUMENT, INVALID_SHAPE, INVALID_VALUE, SIZE_OVERFLOW, RESOURCE_EXHAUSTED, UNSUPPORTED_CONFIGURATION, and EXECUTION_FAILURE. Failures must include the relevant parameter or stage and must not return a success status or partial results. The specific exception or return-value interface will be determined after coding is authorized.

## 3. Distances, Candidates, and Results

### 3.1 Distance Rules

Use squared Euclidean distance, without square roots or normalization. All samples must be considered; approximate indexes are not used.

The scalar reference performs FP32 subtraction, squaring, and accumulation in increasing dimension order, with automatic vectorization, FMA contraction, and floating-point reassociation disabled. The core direct-L2 implementation uses a controlled configuration with FMA contraction disabled; fast-math is prohibited. Explicit FMA optimization may be included as a separate comparison, with its name, settings, and numerical effects recorded.

SIMD, warp, and GEMM implementations may use different reduction orders; bitwise agreement is not required. Computation configurations must be recorded. The CPU defaults to round-to-nearest with FTZ/DAZ disabled. Record the GPU FTZ configuration, and use settings that do not actively flush tiny values to zero for the main comparison. Third-party library behavior that cannot be controlled individually must be explained separately and must not be claimed to be fully equivalent.

GEMM uses a cuBLAS FP32 reference. Core comparisons prohibit reduced-precision input computation such as TF32 and FP16/BF16. Record the compute type, math mode, and library version. Negative reconstructed distances are clamped to positive zero. Retain the count and minimum value of negative results before clamping, along with error diagnostics. Non-finite results constitute numerical failure. Report database-norm preparation separately; query norms, GEMM, reconstruction, and Top-k are included in call time.

### 3.2 Total Order and Output

Valid candidates rank ahead of invalid candidates. Among valid candidates, smaller distances rank first; when distances are exactly equal, smaller IDs rank first. Positive and negative zero are considered equal, and all output zeros are normalized to positive zero. Error tolerances must not be used in candidate comparisons.

Padding candidates have positive-infinity distance and ID=4294967295. Valid samples never use this ID. An explicit validity field is not mandatory, but the actual valid count must be passed and padding must be excluded.

Each query outputs exactly k distinct valid IDs, their corresponding FP32 distances, and their original labels, sorted in ascending total order. Results consist of three logical contiguous q×k arrays, ordered first by query and then by rank. Output distances must be the complete distances actually used for selection. Distances must not be replaced with another computation form without re-sorting.

## 4. Module Interfaces, Blocking, and Merging

| Module | Input semantics | Output semantics |
|---|---|---|
| Distance | Validated matrices, ranges, layout, and computation configuration | FP32 distances, IDs, and valid count |
| Local selection | Candidates, capacity, valid count, k, and T | Sorted local candidates and actual count |
| Merge | Sorted candidate lists, their valid lengths, and k | First min(k,total candidate count) sorted candidates |
| Complete search | Validated inputs, configuration, workspace, and output capacity | Complete q×k results and execution status |
| Validation | Same original inputs, implementation under test, and reference | Separate check results and difference diagnostics |

The caller owns the inputs, outputs, and reusable workspace. Capacity and valid count are separate. Necessary state must be reset for each call, and this reset is timed; Top-k state from previous queries must not be retained. GPU interfaces must specify the device, stream, lifetime, and completion conditions. Asynchronous results must not be read before completion.

Each block retains min(k,valid sample count) candidates. When k>T, retain all valid entries in the block. Pad the final block to T, but do not compute real distances for padding or output padding entries. Local selection, merging, and final sorting are all timed.

The heap root is the worst candidate; sort before output. Resolve radix threshold ties by ID. Refinement, flagging, exclusive scan, compaction, and final sorting are all included in timing.

The core CPU early-termination approach is the scalar heap. Once Top-k is full, terminate only when the partial distance computed under the same rules is strictly greater than the worst distance tau; continue when it equals tau. Partial distances must not be output. Results must exactly match the full heap implementation using the same rules. Combining this with SIMD requires separate safe bounds and validation evidence.

## 5. Shared Correctness Criteria

Member 1 maintains an independent FP64 oracle. It promotes the actual FP32 inputs to FP64 for computation, without generating separate data or reusing production distance or selection kernels. For small cases, retain all distances, reference IDs, labels, and the k/k+1 boundary. There is no k+1 entry when k=N.

Selection tests using identical fixed distances require exact agreement in IDs and ordering across algorithms. Differences between distance computation forms and FP64 are assessed separately.

Use atol=10⁻⁵ and rtol=10⁻⁴. A distance passes when its absolute error is ≤atol+rtol×|reference distance|. Avoid division by zero when the reference is zero. These thresholds do not guarantee that every input in the supported domain passes. Record and analyze exceedances; do not silently relax the thresholds. Any threshold change requires updating this contract and rerunning validation first.

Record separate checks for structure, labels, sorting, fixed-distance selection, distance tolerance, FP64 ID agreement, and repeated-run determinism. Conclusions may include multiple flags such as strict agreement, explained boundary differences, structural errors, selection errors, distance-tolerance exceedances, and nondeterministic results.

Explained numerical limitations may be reported as research findings, with difference statistics, but must not be labeled as strict agreement. Unexplained differences and structural or selection errors cannot support credible speedup claims.

Shared cases must cover at least zero/identical vectors, duplicate samples, exact/near ties, k=1, k=N, k>T, invalid k, final partial blocks, SIMD tails, multiple queries, repeated calls, and early termination when the partial distance equals or exceeds tau or the heap is not yet full. The three members must use the same basic expected results.

## 6. Main Experimental Inputs and Parameters

The main data generation rule is named synthetic-v1: use the standard MT19937 32-bit sequence with seed=5206. Take the upper 24 bits of each integer, divide by 2²⁴, then multiply by 2 and subtract 1 to produce FP32 components in [−1,1). Do not substitute language-specific default distribution functions.

For each parameter combination, reinitialize the same seed, generate X row by row followed by Q, and set each label to ID modulo 10. Member 3 maintains shared generation and distribution. The actual FP32 bit patterns are authoritative. Shared exports use little-endian byte order and record shape, generator version, and SHA-256. All methods within a combination must receive identical inputs. Queries need not be identical across different combinations.

| Parameter | Baseline | Sweep values |
|---|---|---|
| N | 100000 | 10000, 100000 |
| d | 64 | 32, 64, 128 |
| q | 16 | 1, 16, 64 |
| k | 10 | 5, 10, 32 |
| T | 128 | 32, 64, 128, 256 |

Change only one main factor at a time, yielding 11 main combinations after deduplication. A full Cartesian product is not required. Correctness boundary cases and additional distributions are listed separately and do not replace the main experiments.

## 7. Measurement and Statistics

Use 10 warm-up runs and 20 measured runs, retaining every run's record. Core CPU experiments use one compute thread. The main CPU environment is the current virtual machine; its specific configuration is documented in `CPU_rules.md`. GPU measurements must run on the NVIDIA environment to be determined later; CPU simulation is not an acceptable substitute.

Complete timing begins with the necessary state reset for the current call and ends when all sorted IDs, distances, and labels are ready. It includes distance computation, candidate management, local selection, merging, sorting, label gathering, required per-call conversions, and synchronization. It excludes generation, I/O, full input validation, the oracle, correctness checks, and initial reusable allocations. Normal calls still validate inputs. Steady-state experiments use shared prevalidated inputs and must state this timing boundary.

There are two main GPU timing scopes. Resident timing uses CUDA events to measure device execution; wait for completion before reading the events, and do not include host waiting in the event values. Transfer-inclusive timing uses a host monotonic clock and includes X/Q/label H2D transfers, search, and result D2H transfers until the host results are available. A pre-resident database with query-only transfers is a separate scenario.

Measure separable stages individually. Report fused or interleaved stages as combined times without forcing an artificial split. Report preparation costs separately. Run diagnostic counters separately. CPU stage times and GPU event times must not be arbitrarily added and presented as complete timing.

For the 20 sorted timing values, the quantile position is 19×p, where p=0.25, 0.5, and 0.75. Use linear interpolation to obtain Q1, the median, and Q3; IQR=Q3−Q1. Do not silently remove outliers. Handling CPU virtual-machine variability follows Member 1's execution requirements. GPU reruns and anomaly records must also be retained.

Throughput=q/median batch time. Batch time/q is an amortized cost, not single-query response latency. Speedup is the ratio of medians for identical inputs, parameters, and timing scopes. Report resident and transfer-inclusive results separately. Cross-environment conclusions must identify virtualization, hardware, and deployment differences and must not generalize to peak physical-hardware performance.

## 8. Memory, Results, and Reproducibility

Report input, output, and auxiliary allocations separately. Record peak live memory and temporary workspace using actual capacities. The logical payload of a candidate distance plus ID is 8 bytes, but verify actual structure sizes, alignment, and capacities separately. Report GPU register and shared-memory resources separately; do not include them in heap-allocation totals. Allocated bytes are not measured memory traffic.

Record the contract version, code version, data identifier, parameters, seed, hardware/system/virtualization, compiler/options, ISA/FMA, CUDA/cuBLAS/computation modes, threads/streams, timing scope, raw repeated-run records, median/IQR, stage times, throughput, auxiliary memory, and separate correctness results.

Distinguish not applicable from not measured; do not substitute zero for either. After fixes, rerun affected results. Do not combine old performance measurements with new correctness conclusions.

## 9. Responsibilities and Integration

| Member | Primary responsibilities | Deliverables to other members |
|---|---|---|
| 1 | Scalar/FP64, CPU blocking/SIMD, sort/heap, early termination, correctness | Shared cases, reference distances/results, CPU validation and analysis |
| 2 | CUDA thread/warp, materialised approach, layout reuse, direct/GEMM, GPU timing | Distance modules, computation configurations, GPU environment and numerical data |
| 3 | CPU/GPU Bitonic, Radix/Scan, sequential/tree merging, benchmark automation, memory summaries | Selection/merge modules, input data, raw records and summaries |

Member 3 delivers CPU Bitonic, and Member 1 integrates it. For fused CUDA, Member 2 handles distance computation/loading and Member 3 handles local selection/merging; both integrate it together.

Each integration proceeds in this order: fixed-distance selection validation, complete-search validation, and performance measurement. All three members share responsibility for experimental interpretation, the report, presentation, and contribution records. Testing and reporting must not all be assigned to one member.

Mandatory work includes proposal Tasks 1–8: three CPU approaches, two CUDA mappings, materialised/fused approaches, two merging methods, Radix/Scan, GEMM, early termination, ablation studies, and crossover analysis. Double buffering for data exceeding GPU memory is an optional extension once the core implementation is stable. Approximate search, multiple GPUs, and training are out of scope.

Final deliverables are source code, build/run instructions, fixed seeds and raw results, plotting scripts, environment information and representative reproduction commands, a technical report, a presentation, and contribution records for all three members. Member 1 has already received user authorization to enter implementation. Actual validation and experimental records are in member1/results and member1/docs.

## 10. Version History

2026-10-09, 1.1: Corrected the one-factor-at-a-time sweep count to 11 combinations (the baseline plus 1+2+2+2+3 variations). Parameter ranges, computation rules, and timing rules are unchanged. This affects the shared benchmark list; Members 2 and 3 must use this version during handoff. The current CPU experiments start from the corrected list, so no old performance records require rerunning.
