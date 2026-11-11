#include "optional_gpu_knn.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <stdexcept>
#include <utility>

namespace gpu_knn {
namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t s, const char *message) {
  if (s != cudaSuccess)
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(s));
}
void blas_check(cublasStatus_t s, const char *message) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(message) + ": cuBLAS status " +
                             std::to_string(s));
}
size_t ceil_div(size_t a, size_t b) { return a / b + (a % b != 0); }
size_t next_power_of_two(size_t n) {
  size_t p = 1;
  while (p < n) {
    if (p > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error("optional Top-k padding overflow");
    p *= 2;
  }
  return p;
}
#include "knn_kernels.cuh"

struct Resources {
  std::vector<void *> device, pinned;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> events;
  cublasHandle_t blas = nullptr;
  size_t device_bytes = 0, pinned_bytes = 0;
  ~Resources() {
    for (auto s : streams)
      cudaStreamSynchronize(s);
    if (blas)
      cublasDestroy(blas);
    for (auto e : events)
      cudaEventDestroy(e);
    for (auto s : streams)
      cudaStreamDestroy(s);
    for (auto p : pinned)
      cudaFreeHost(p);
    for (auto p : device)
      cudaFree(p);
  }
  template <class T> T *allocate(size_t n) {
    T *p = nullptr;
    const size_t bytes = checked_product(n, sizeof(T));
    check(cudaMalloc(reinterpret_cast<void **>(&p), bytes),
          "allocate optional device buffer");
    try {
      device.push_back(p);
    } catch (...) {
      cudaFree(p);
      throw;
    }
    device_bytes += bytes;
    return p;
  }
  template <class T> T *pin(size_t n) {
    T *p = nullptr;
    const size_t bytes = checked_product(n, sizeof(T));
    check(cudaHostAlloc(reinterpret_cast<void **>(&p), bytes,
                        cudaHostAllocDefault),
          "allocate pinned staging buffer");
    try {
      pinned.push_back(p);
    } catch (...) {
      cudaFreeHost(p);
      throw;
    }
    pinned_bytes += bytes;
    return p;
  }
  cudaStream_t stream() {
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),
          "create optional stream");
    try {
      streams.push_back(s);
    } catch (...) {
      cudaStreamDestroy(s);
      throw;
    }
    return s;
  }
  cudaEvent_t event() {
    cudaEvent_t e = nullptr;
    check(cudaEventCreate(&e), "create optional event");
    try {
      events.push_back(e);
    } catch (...) {
      cudaEventDestroy(e);
      throw;
    }
    return e;
  }
};
void record(cudaEvent_t e, cudaStream_t s) {
  check(cudaEventRecord(e, s), "record optional timing event");
}
double elapsed(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  check(cudaEventElapsedTime(&ms, a, b), "read optional timing event");
  return ms;
}
void launch_check() { check(cudaPeekAtLastError(), "launch optional kernel"); }

// A work-efficient exclusive Blelloch scan of one 256-element block.
__global__ void scan_blocks(const uint32_t *input, uint32_t *output,
                            uint32_t *sums, size_t n, size_t blocks) {
  __shared__ uint32_t temp[256];
  const size_t qi = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const unsigned lane = threadIdx.x;
  const size_t id = block * 256 + lane;
  temp[lane] = id < n ? input[qi * n + id] : 0;
  __syncthreads();
  for (unsigned stride = 1; stride < 256; stride *= 2) {
    const unsigned at = (lane + 1) * 2 * stride - 1;
    if (at < 256)
      temp[at] += temp[at - stride];
    __syncthreads();
  }
  if (lane == 0) {
    sums[qi * blocks + block] = temp[255];
    temp[255] = 0;
  }
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    const unsigned at = (lane + 1) * 2 * stride - 1;
    if (at < 256) {
      const uint32_t left = temp[at - stride];
      temp[at - stride] = temp[at];
      temp[at] += left;
    }
    __syncthreads();
  }
  if (id < n)
    output[qi * n + id] = temp[lane];
}
__global__ void add_scan_offsets(uint32_t *output, const uint32_t *offsets,
                                 size_t n, size_t q, size_t blocks) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n * q) {
    const size_t qi = index / n, id = index % n;
    output[index] += offsets[qi * blocks + id / 256];
  }
}
struct ScanPlan {
  struct Level {
    size_t n, blocks;
    uint32_t *sums, *offsets;
  };
  std::vector<Level> levels;
  size_t q;
  ScanPlan(Resources &r, size_t n, size_t queries) : q(queries) {
    for (;;) {
      size_t blocks = ceil_div(n, 256);
      levels.push_back(
          {n, blocks, r.allocate<uint32_t>(checked_product(q, blocks)),
           blocks > 1 ? r.allocate<uint32_t>(checked_product(q, blocks))
                      : nullptr});
      if (blocks == 1)
        break;
      n = blocks;
    }
  }
  void launch(const uint32_t *in, uint32_t *out, cudaStream_t stream,
              size_t index = 0) {
    const auto &level = levels[index];
    scan_blocks<<<unsigned(q * level.blocks), 256, 0, stream>>>(
        in, out, level.sums, level.n, level.blocks);
    launch_check();
    if (level.blocks > 1) {
      launch(level.sums, level.offsets, stream, index + 1);
      add_scan_offsets<<<unsigned(ceil_div(q * level.n, 256)), 256, 0,
                         stream>>>(out, level.offsets, level.n, q,
                                   level.blocks);
      launch_check();
    }
  }
};

__global__ void initialize_radix(uint64_t *prefix, uint32_t *rank, size_t q,
                                 unsigned k) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < q) {
    prefix[i] = 0;
    rank[i] = k;
  }
}
// Eight MSB-first histogram refinements identify the kth deterministic key.
// This does not sort the input or call a library Top-k implementation.
__global__ void refine_radix(const uint64_t *keys, uint64_t *prefix,
                             uint32_t *rank, size_t n, unsigned shift) {
  __shared__ uint32_t histogram[256];
  const unsigned lane = threadIdx.x;
  const size_t qi = blockIdx.x;
  histogram[lane] = 0;
  __syncthreads();
  const uint64_t higher = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
  const uint64_t selected = prefix[qi];
  for (size_t i = lane; i < n; i += 256) {
    const uint64_t key = keys[qi * n + i];
    if ((key & higher) == (selected & higher))
      atomicAdd(histogram + ((key >> shift) & 255), 1u);
  }
  __syncthreads();
  if (lane == 0) {
    uint32_t remaining = rank[qi];
    unsigned digit = 0;
    while (digit < 255 && remaining > histogram[digit]) {
      remaining -= histogram[digit];
      ++digit;
    }
    prefix[qi] = selected | (uint64_t(digit) << shift);
    rank[qi] = remaining;
  }
}
__global__ void mark_flags(const uint64_t *keys, const uint64_t *threshold,
                           uint32_t *flags, size_t n, size_t total) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total)
    flags[index] = keys[index] <= threshold[index / n];
}
__global__ void compact_keys(const uint64_t *keys, const uint32_t *flags,
                             const uint32_t *positions, uint64_t *selected,
                             size_t n, size_t k, size_t total) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total && flags[index])
    selected[(index / n) * k + positions[index]] = keys[index];
}
__global__ void sort_selected(const uint64_t *selected, Candidate *out,
                              unsigned k, unsigned padded) {
  extern __shared__ Candidate shared[];
  const unsigned i = threadIdx.x;
  Candidate value = invalid_candidate();
  if (i < k) {
    const uint64_t key = selected[size_t(blockIdx.x) * k + i];
    value = {__uint_as_float(uint32_t(key >> 32)), uint32_t(key)};
  }
  shared[i] = value;
  __syncthreads();
  bitonic_sort_shared(shared, padded);
  if (i < k)
    out[size_t(blockIdx.x) * k + i] = shared[i];
}
__global__ void norms(const float *values, float *out, size_t rows, size_t d) {
  const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0;
    for (size_t j = 0; j < d; ++j) {
      float v = values[row * d + j];
      sum += v * v;
    }
    out[row] = sum;
  }
}
__global__ void reconstruct(float *distances, const float *nx, const float *nq,
                            const float *x, const float *queries, size_t n,
                            size_t d, size_t total,
                            unsigned long long *rechecks) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    const size_t qi = index / n, id = index % n;
    float distance = (distances[index] + nx[id]) + nq[qi];
    // Norm/dot cancellation can turn nearly identical vectors into spurious
    // positive/negative distances. Recheck only this small-distance region.
    const float guard = 8.0f * FLT_EPSILON * float(d) * (nx[id] + nq[qi]);
    if (distance <= guard) {
      distance = thread_distance(queries + qi * d, x + id * d, d);
      atomicAdd(rechecks, 1ULL);
    }
    distances[index] = distance > 0 ? distance : 0.0f;
  }
}
__global__ void initialize_candidates(Candidate *out, size_t count) {
  const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id < count)
    out[id] = invalid_candidate();
}
__global__ void merge_pair(const Candidate *left, const Candidate *right,
                           Candidate *out, unsigned k, unsigned padded) {
  extern __shared__ Candidate shared[];
  const unsigned i = threadIdx.x;
  const size_t qi = blockIdx.x;
  const bool second = i >= padded;
  const unsigned rank = second ? i - padded : i;
  shared[i] = rank < k ? (second ? right[qi * k + rank] : left[qi * k + rank])
                       : invalid_candidate();
  __syncthreads();
  bitonic_sort_shared(shared, 2 * padded);
  if (i < k)
    out[qi * k + i] = shared[i];
}
} // namespace

struct OptionalRunner::Impl {
  enum E {
    START,
    H2D0,
    H2D1,
    DIST0,
    DIST1,
    PACK0,
    PACK1,
    NX0,
    NX1,
    NQ0,
    NQ1,
    GEMM0,
    GEMM1,
    REC0,
    REC1,
    SEL0,
    SEL1,
    SCAN0,
    SCAN1,
    COMPACT0,
    COMPACT1,
    SORT0,
    SORT1,
    GATHER0,
    GATHER1,
    END,
    D2H0,
    D2H1,
    EVENTS
  };
  struct ChunkEvents {
    cudaEvent_t h0, h1, c0, c1, m0, m1;
  };
  struct Slot {
    cudaStream_t copy = nullptr;
    float *host_x = nullptr, *x = nullptr;
    Candidate *a = nullptr, *b = nullptr;
    cudaEvent_t done = nullptr;
  };
  Resources resources;
  const HostInput *host;
  OptionalConfig config;
  size_t n, d, q, k, total, out_count, padded;
  cudaStream_t stream;
  cudaEvent_t events[EVENTS]{};
  OptionalMemory stats;
  float *x = nullptr, *queries = nullptr, *distances = nullptr,
        *norm_x = nullptr, *norm_q = nullptr;
  int32_t *labels = nullptr, *out_labels = nullptr;
  uint32_t *out_ids = nullptr;
  float *out_distances = nullptr;
  uint64_t *keys = nullptr, *keys_sorted = nullptr, *prefix = nullptr,
           *selected = nullptr;
  uint32_t *rank = nullptr, *flags = nullptr, *positions = nullptr;
  Candidate *selected_candidates = nullptr, *global_a = nullptr,
            *global_b = nullptr;
  int *sort_offsets = nullptr;
  void *sort_temp = nullptr;
  size_t sort_temp_bytes = 0;
  unsigned long long *rechecks = nullptr;
  std::unique_ptr<ScanPlan> scan;
  std::vector<uint32_t> h_ids;
  std::vector<float> h_distances;
  std::vector<int32_t> h_labels;
  std::vector<Slot> slots;
  std::vector<ChunkEvents> chunks;
  float *pinned_queries = nullptr;
  Candidate *pinned_output = nullptr;
  size_t chunk_size = 0;

  bool streaming() const {
    return config.route == OptionalRoute::StreamSingle ||
           config.route == OptionalRoute::StreamDouble;
  }
  Impl(const HostInput &input, OptionalConfig cfg)
      : host(&input), config(cfg), n(input.shape.n), d(input.shape.d),
        q(input.shape.q), k(input.shape.k), total(checked_product(n, q)),
        out_count(checked_product(q, k)), padded(next_power_of_two(k)) {
    const auto setup_start = Clock::now();
    validate_input(input);
    if (total > INT_MAX || n > INT_MAX || d > INT_MAX || q > 65535)
      throw std::invalid_argument(
          "optional routes require q<=65535 and N,d,q*N<=INT_MAX");
    if (config.mapping != Mapping::ThreadPerVector &&
        config.mapping != Mapping::WarpPerVector)
      throw std::invalid_argument("invalid optional mapping");
    if (config.route != OptionalRoute::RadixSelect &&
        config.route != OptionalRoute::GemmReference && !streaming())
      throw std::invalid_argument("invalid optional route");
    if (k > 128)
      throw std::invalid_argument("optional CUDA routes support k<=128");
    if (streaming() &&
        (!config.chunk_size || config.tile_size < 32 ||
         config.tile_size > 256 ||
         (config.tile_size & (config.tile_size - 1)) || k > config.tile_size))
      throw std::invalid_argument(
          "invalid streaming chunk/tile size or k>tile");
    stream = resources.stream();
    for (auto &event : events)
      event = resources.event();
    h_ids.resize(out_count);
    h_distances.resize(out_count);
    h_labels.resize(out_count);
    stats.logical_input_bytes =
        checked_product(n * d + q * d + n, sizeof(float));
    if (streaming())
      initialize_streaming();
    else
      initialize_materialized();
    stats.device.peak_allocated_bytes = resources.device_bytes;
    stats.device.auxiliary_bytes = resources.device_bytes -
                                   stats.device.input_bytes -
                                   stats.device.output_bytes;
    stats.pinned_host_bytes = resources.pinned_bytes;
    stats.setup_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - setup_start)
            .count();
  }

  void initialize_materialized() {
    x = resources.allocate<float>(checked_product(n, d));
    queries = resources.allocate<float>(checked_product(q, d));
    labels = resources.allocate<int32_t>(n);
    distances = resources.allocate<float>(total);
    keys = resources.allocate<uint64_t>(total);
    out_ids = resources.allocate<uint32_t>(out_count);
    out_distances = resources.allocate<float>(out_count);
    out_labels = resources.allocate<int32_t>(out_count);
    stats.device.input_bytes = stats.logical_input_bytes;
    stats.device.output_bytes = 12 * out_count;
    stats.device.distance_buffer_bytes = 4 * total;
    stats.device.key_buffer_bytes = 8 * total;
    if (config.route == OptionalRoute::RadixSelect) {
      prefix = resources.allocate<uint64_t>(q);
      rank = resources.allocate<uint32_t>(q);
      flags = resources.allocate<uint32_t>(total);
      positions = resources.allocate<uint32_t>(total);
      selected = resources.allocate<uint64_t>(out_count);
      selected_candidates = resources.allocate<Candidate>(out_count);
      scan = std::make_unique<ScanPlan>(resources, n, q);
      stats.device.candidate_buffer_bytes = 16 * out_count;
      stats.device.dynamic_shared_bytes =
          std::max(size_t(1024), padded * sizeof(Candidate));
    } else {
      keys_sorted = resources.allocate<uint64_t>(total);
      norm_x = resources.allocate<float>(n);
      norm_q = resources.allocate<float>(q);
      rechecks = resources.allocate<unsigned long long>(1);
      sort_offsets = resources.allocate<int>(q + 1);
      std::vector<int> offsets(q + 1);
      for (size_t i = 0; i <= q; ++i)
        offsets[i] = int(i * n);
      check(cudaMemcpyAsync(sort_offsets, offsets.data(), (q + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream),
            "upload GEMM sort offsets");
      check(cub::DeviceSegmentedRadixSort::SortKeys(
                nullptr, sort_temp_bytes, keys, keys_sorted, int(total), int(q),
                sort_offsets, sort_offsets + 1, 0, 64, stream),
            "query GEMM sort scratch");
      sort_temp = resources.allocate<unsigned char>(sort_temp_bytes);
      stats.device.cub_temp_bytes = sort_temp_bytes;
      stats.device.key_buffer_bytes = 16 * total;
      blas_check(cublasCreate(&resources.blas), "create cuBLAS handle");
      blas_check(cublasGetVersion(resources.blas, &stats.cublas_version),
                 "read cuBLAS version");
      blas_check(cublasSetStream(resources.blas, stream), "set cuBLAS stream");
      blas_check(cublasSetMathMode(resources.blas, CUBLAS_PEDANTIC_MATH),
                 "set FP32 pedantic GEMM math");
      blas_check(
          cublasSetAtomicsMode(resources.blas, CUBLAS_ATOMICS_NOT_ALLOWED),
          "disable cuBLAS atomics");
      stats.cublas_workspace_bytes = 4 * 1024 * 1024;
      void *workspace =
          resources.allocate<unsigned char>(stats.cublas_workspace_bytes);
      blas_check(cublasSetWorkspace(resources.blas, workspace,
                                    stats.cublas_workspace_bytes),
                 "set reusable cuBLAS workspace");
    }
    upload();
    if (config.route == OptionalRoute::GemmReference) {
      record(events[NX0], stream);
      norms<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(x, norm_x, n, d);
      launch_check();
      record(events[NX1], stream);
    }
    check(cudaStreamSynchronize(stream),
          "initialize materialized optional route");
    if (config.route == OptionalRoute::GemmReference)
      stats.database_norm_setup_ms = elapsed(events[NX0], events[NX1]);
  }

  void initialize_streaming() {
    chunk_size = std::min(config.chunk_size, n);
    const size_t buffers = config.route == OptionalRoute::StreamDouble ? 2 : 1;
    const size_t tiles = ceil_div(chunk_size, config.tile_size);
    const size_t candidates = checked_product(checked_product(q, tiles), k);
    queries = resources.allocate<float>(q * d);
    pinned_queries = resources.pin<float>(q * d);
    std::memcpy(pinned_queries, host->queries.data(), q * d * sizeof(float));
    global_a = resources.allocate<Candidate>(out_count);
    global_b = resources.allocate<Candidate>(out_count);
    pinned_output = resources.pin<Candidate>(out_count);
    slots.resize(buffers);
    for (auto &slot : slots) {
      slot.copy = resources.stream();
      slot.host_x = resources.pin<float>(chunk_size * d);
      slot.x = resources.allocate<float>(chunk_size * d);
      slot.a = resources.allocate<Candidate>(candidates);
      slot.b = resources.allocate<Candidate>(candidates);
    }
    chunks.resize(ceil_div(n, chunk_size));
    for (auto &event : chunks)
      event = {resources.event(), resources.event(), resources.event(),
               resources.event(), resources.event(), resources.event()};
    stats.buffers = buffers;
    stats.chunk_size = chunk_size;
    stats.device.input_bytes = 4 * (q * d + buffers * chunk_size * d);
    stats.device.output_bytes = out_count * sizeof(Candidate);
    stats.device.candidate_buffer_bytes =
        2 * (buffers * candidates + out_count) * sizeof(Candidate);
    stats.device.dynamic_shared_bytes =
        sizeof(Candidate) * std::max(config.tile_size, 2 * padded);
  }

  void upload() {
    check(cudaMemcpyAsync(x, host->x.data(), n * d * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload optional database");
    check(cudaMemcpyAsync(queries, host->queries.data(), q * d * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload optional queries");
    check(cudaMemcpyAsync(labels, host->labels.data(), n * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream),
          "upload optional labels");
  }
  void distance() {
    if (config.mapping == Mapping::ThreadPerVector)
      materialized_thread_distance_kernel<<<unsigned(ceil_div(total, 256)), 256,
                                            0, stream>>>(x, queries, distances,
                                                         n, d, q);
    else
      materialized_warp_distance_kernel<<<unsigned(ceil_div(total, 8)), 256, 0,
                                          stream>>>(x, queries, distances, n, d,
                                                    q);
    launch_check();
  }
  void copy_output() {
    check(cudaMemcpyAsync(h_ids.data(), out_ids, out_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream),
          "copy optional IDs");
    check(cudaMemcpyAsync(h_distances.data(), out_distances,
                          out_count * sizeof(float), cudaMemcpyDeviceToHost,
                          stream),
          "copy optional distances");
    check(cudaMemcpyAsync(h_labels.data(), out_labels,
                          out_count * sizeof(int32_t), cudaMemcpyDeviceToHost,
                          stream),
          "copy optional labels");
  }
  void finish_output(OptionalResult &result) {
    result.output.ids = h_ids;
    result.output.distances = h_distances;
    result.output.labels = h_labels;
  }

  OptionalResult run_materialized(bool copy_inputs, bool copy_outputs) {
    OptionalResult result;
    result.output.memory = stats.device;
    auto &t = result.output.times;
    const auto start = Clock::now();
    if (copy_inputs) {
      record(events[H2D0], stream);
      upload();
      record(events[H2D1], stream);
    }
    record(events[START], stream);
    if (config.route == OptionalRoute::RadixSelect) {
      record(events[DIST0], stream);
      distance();
      record(events[DIST1], stream);
    } else {
      if (copy_inputs) {
        record(events[NX0], stream);
        norms<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(x, norm_x, n, d);
        launch_check();
        record(events[NX1], stream);
      }
      record(events[NQ0], stream);
      norms<<<unsigned(ceil_div(q, 256)), 256, 0, stream>>>(queries, norm_q, q,
                                                            d);
      launch_check();
      record(events[NQ1], stream);
      record(events[GEMM0], stream);
      const float alpha = -2.0f, beta = 0.0f;
      // Row-major X (N*d), Q (q*d) are interpreted as column-major d*N,d*q.
      // The column-major N*q output has the required query-major q*N layout.
      blas_check(cublasSgemm(resources.blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n),
                             int(q), int(d), &alpha, x, int(d), queries, int(d),
                             &beta, distances, int(n)),
                 "GEMM squared-L2 dot products");
      record(events[GEMM1], stream);
      record(events[REC0], stream);
      check(cudaMemsetAsync(rechecks, 0, sizeof(unsigned long long), stream),
            "reset GEMM recheck counter");
      reconstruct<<<unsigned(ceil_div(total, 256)), 256, 0, stream>>>(
          distances, norm_x, norm_q, x, queries, n, d, total, rechecks);
      launch_check();
      record(events[REC1], stream);
    }
    record(events[PACK0], stream);
    pack_keys_kernel<<<unsigned(ceil_div(total, 256)), 256, 0, stream>>>(
        distances, keys, n, total);
    launch_check();
    record(events[PACK1], stream);
    if (config.route == OptionalRoute::RadixSelect) {
      record(events[SEL0], stream);
      initialize_radix<<<unsigned(ceil_div(q, 256)), 256, 0, stream>>>(
          prefix, rank, q, unsigned(k));
      launch_check();
      for (int shift = 56; shift >= 0; shift -= 8) {
        refine_radix<<<unsigned(q), 256, 0, stream>>>(keys, prefix, rank, n,
                                                      unsigned(shift));
        launch_check();
      }
      record(events[SEL1], stream);
      record(events[SCAN0], stream);
      mark_flags<<<unsigned(ceil_div(total, 256)), 256, 0, stream>>>(
          keys, prefix, flags, n, total);
      launch_check();
      scan->launch(flags, positions, stream);
      record(events[SCAN1], stream);
      record(events[COMPACT0], stream);
      compact_keys<<<unsigned(ceil_div(total, 256)), 256, 0, stream>>>(
          keys, flags, positions, selected, n, k, total);
      launch_check();
      record(events[COMPACT1], stream);
      record(events[SORT0], stream);
      sort_selected<<<unsigned(q), unsigned(padded), padded * sizeof(Candidate),
                      stream>>>(selected, selected_candidates, unsigned(k),
                                unsigned(padded));
      launch_check();
      record(events[SORT1], stream);
    } else {
      record(events[SEL0], stream);
      check(cub::DeviceSegmentedRadixSort::SortKeys(
                sort_temp, sort_temp_bytes, keys, keys_sorted, int(total),
                int(q), sort_offsets, sort_offsets + 1, 0, 64, stream),
            "sort GEMM reference keys");
      record(events[SEL1], stream);
    }
    record(events[GATHER0], stream);
    if (config.route == OptionalRoute::RadixSelect)
      gather_candidates_kernel<<<unsigned(ceil_div(out_count, 256)), 256, 0,
                                 stream>>>(selected_candidates, labels, out_ids,
                                           out_distances, out_labels,
                                           out_count);
    else
      gather_keys_kernel<<<unsigned(ceil_div(out_count, 256)), 256, 0,
                           stream>>>(keys_sorted, labels, out_ids,
                                     out_distances, out_labels, n, k, q);
    launch_check();
    record(events[GATHER1], stream);
    record(events[END], stream);
    if (copy_outputs) {
      record(events[D2H0], stream);
      copy_output();
      record(events[D2H1], stream);
    }
    check(cudaEventSynchronize(events[copy_outputs ? D2H1 : END]),
          "finish optional search");
    t.end_to_end_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    t.device_ms = elapsed(events[START], events[END]);
    t.pack_ms = elapsed(events[PACK0], events[PACK1]);
    t.gather_ms = elapsed(events[GATHER0], events[GATHER1]);
    if (copy_inputs)
      t.h2d_ms = elapsed(events[H2D0], events[H2D1]);
    if (copy_outputs)
      t.d2h_ms = elapsed(events[D2H0], events[D2H1]);
    if (config.route == OptionalRoute::RadixSelect) {
      t.distance_ms = elapsed(events[DIST0], events[DIST1]);
      result.times.radix_ms = elapsed(events[SEL0], events[SEL1]);
      result.times.scan_ms = elapsed(events[SCAN0], events[SCAN1]);
      result.times.compact_ms = elapsed(events[COMPACT0], events[COMPACT1]);
      t.local_selection_ms = elapsed(events[SORT0], events[SORT1]);
    } else {
      t.distance_ms = std::numeric_limits<double>::quiet_NaN();
      result.times.norm_x_ms =
          copy_inputs ? elapsed(events[NX0], events[NX1]) : 0;
      result.times.norm_q_ms = elapsed(events[NQ0], events[NQ1]);
      result.times.gemm_ms = elapsed(events[GEMM0], events[GEMM1]);
      result.times.reconstruct_ms = elapsed(events[REC0], events[REC1]);
      t.full_sort_ms = elapsed(events[SEL0], events[SEL1]);
      if (copy_outputs)
        check(cudaMemcpy(&result.near_zero_rechecks, rechecks,
                         sizeof(unsigned long long), cudaMemcpyDeviceToHost),
              "read GEMM recheck counter");
    }
    if (copy_outputs)
      finish_output(result);
    return result;
  }

  Candidate *select_chunk(Slot &slot, size_t rows, size_t offset) {
    const size_t tiles = ceil_div(rows, config.tile_size);
    const dim3 grid{unsigned(tiles), unsigned(q)};
    if (config.mapping == Mapping::ThreadPerVector)
      fused_thread_bitonic_kernel<<<grid, unsigned(config.tile_size),
                                    config.tile_size * sizeof(Candidate),
                                    stream>>>(slot.x, queries, slot.a, rows, d,
                                              q, unsigned(config.tile_size),
                                              unsigned(k), tiles, offset);
    else
      fused_warp_bitonic_kernel<<<
          grid, 256, config.tile_size * sizeof(Candidate), stream>>>(
          slot.x, queries, slot.a, rows, d, q, unsigned(config.tile_size),
          unsigned(k), tiles, offset);
    launch_check();
    Candidate *a = slot.a, *b = slot.b;
    size_t lists = tiles;
    while (lists > 1) {
      const size_t next = ceil_div(lists, 2);
      merge_topk_kernel<<<dim3(unsigned(next), unsigned(q)),
                          unsigned(2 * padded), 2 * padded * sizeof(Candidate),
                          stream>>>(a, b, lists, next, unsigned(k),
                                    unsigned(padded));
      launch_check();
      std::swap(a, b);
      lists = next;
    }
    return a;
  }

  OptionalResult run_streaming(bool copy_inputs, bool copy_outputs) {
    if (!copy_inputs || !copy_outputs)
      throw std::invalid_argument("streaming has end-to-end timing only; "
                                  "chunks must transfer and outputs return");
    OptionalResult result;
    result.output.memory = stats.device;
    auto &t = result.output.times;
    t.device_ms = std::numeric_limits<double>::quiet_NaN();
    t.distance_ms = std::numeric_limits<double>::quiet_NaN();
    for (auto &slot : slots)
      slot.done = nullptr;
    Candidate *current = global_a, *next = global_b;
    const auto start = Clock::now();
    std::memcpy(pinned_queries, host->queries.data(), q * d * sizeof(float));
    result.times.staging_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    record(events[START], stream);
    record(events[H2D0], stream);
    check(cudaMemcpyAsync(queries, pinned_queries, q * d * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload streaming queries");
    record(events[H2D1], stream);
    initialize_candidates<<<unsigned(ceil_div(out_count, 256)), 256, 0,
                            stream>>>(current, out_count);
    launch_check();
    for (size_t c = 0; c < chunks.size(); ++c) {
      auto &slot = slots[c % slots.size()];
      auto &event = chunks[c];
      const size_t begin = c * chunk_size,
                   rows = std::min(chunk_size, n - begin);
      if (slot.done)
        check(cudaEventSynchronize(slot.done),
              "wait for reusable streaming slot");
      const auto stage_start = Clock::now();
      std::memcpy(slot.host_x, host->x.data() + begin * d,
                  rows * d * sizeof(float));
      result.times.staging_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - stage_start)
              .count();
      check(cudaStreamWaitEvent(slot.copy, events[START], 0),
            "order streaming start");
      record(event.h0, slot.copy);
      check(cudaMemcpyAsync(slot.x, slot.host_x, rows * d * sizeof(float),
                            cudaMemcpyHostToDevice, slot.copy),
            "upload streaming chunk");
      record(event.h1, slot.copy);
      check(cudaStreamWaitEvent(stream, event.h1, 0),
            "wait for streamed chunk");
      record(event.c0, stream);
      Candidate *local = select_chunk(slot, rows, begin);
      record(event.c1, stream);
      record(event.m0, stream);
      merge_pair<<<unsigned(q), unsigned(2 * padded),
                   2 * padded * sizeof(Candidate), stream>>>(
          current, local, next, unsigned(k), unsigned(padded));
      launch_check();
      record(event.m1, stream);
      slot.done = event.m1;
      std::swap(current, next);
    }
    record(events[D2H0], stream);
    check(cudaMemcpyAsync(pinned_output, current, out_count * sizeof(Candidate),
                          cudaMemcpyDeviceToHost, stream),
          "copy streaming Top-k");
    record(events[D2H1], stream);
    record(events[END], stream);
    check(cudaEventSynchronize(events[END]), "finish streaming pipeline");
    for (size_t i = 0; i < out_count; ++i) {
      if (pinned_output[i].id >= n)
        throw std::runtime_error("streaming returned an invalid candidate");
      h_ids[i] = pinned_output[i].id;
      h_distances[i] = pinned_output[i].distance;
      h_labels[i] = host->labels[h_ids[i]];
    }
    t.end_to_end_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    t.h2d_ms = elapsed(events[H2D0], events[H2D1]);
    t.d2h_ms = elapsed(events[D2H0], events[D2H1]);
    for (const auto &e : chunks) {
      t.h2d_ms += elapsed(e.h0, e.h1);
      result.times.chunk_selection_ms += elapsed(e.c0, e.c1);
      t.merge_ms += elapsed(e.m0, e.m1);
    }
    result.times.pipeline_ms = elapsed(events[START], events[END]);
    result.times.compute_service_ms =
        result.times.chunk_selection_ms + t.merge_ms;
    const double transfer = t.h2d_ms + t.d2h_ms,
                 compute = result.times.compute_service_ms;
    const double overlap =
        std::max(0.0, transfer + compute - result.times.pipeline_ms);
    result.times.estimated_overlap_ratio =
        std::min(transfer, compute) > 0
            ? std::min(1.0, overlap / std::min(transfer, compute))
            : 0;
    finish_output(result);
    return result;
  }
};

const char *route_name(OptionalRoute route) {
  switch (route) {
  case OptionalRoute::RadixSelect:
    return "O3_cuda_radix_scan_compact";
  case OptionalRoute::GemmReference:
    return "O4_cublas_gemm_reference";
  case OptionalRoute::StreamSingle:
    return "O5_cuda_stream_single";
  case OptionalRoute::StreamDouble:
    return "O5_cuda_stream_double";
  }
  return "invalid_optional_route";
}
OptionalRunner::OptionalRunner(const HostInput &input, OptionalConfig config)
    : impl_(std::make_unique<Impl>(input, config)) {}
OptionalRunner::~OptionalRunner() = default;
OptionalRunner::OptionalRunner(OptionalRunner &&) noexcept = default;
OptionalRunner &OptionalRunner::operator=(OptionalRunner &&) noexcept = default;
OptionalResult OptionalRunner::run(bool copy_inputs, bool copy_outputs) {
  if (!impl_)
    throw std::logic_error("moved-from optional runner");
  return impl_->streaming()
             ? impl_->run_streaming(copy_inputs, copy_outputs)
             : impl_->run_materialized(copy_inputs, copy_outputs);
}
bool OptionalRunner::supports_resident() const {
  if (!impl_)
    throw std::logic_error("moved-from optional runner");
  return !impl_->streaming();
}
const OptionalMemory &OptionalRunner::memory() const {
  if (!impl_)
    throw std::logic_error("moved-from optional runner");
  return impl_->stats;
}
std::vector<float> OptionalRunner::materialized_distances() const {
  if (!impl_ || impl_->streaming())
    throw std::logic_error(
        "streaming does not materialize the database distance matrix");
  std::vector<float> output(impl_->total);
  check(cudaMemcpy(output.data(), impl_->distances,
                   output.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download optional distance matrix");
  return output;
}
std::vector<uint32_t> debug_exclusive_scan(const std::vector<uint32_t> &flags,
                                           size_t n, size_t q) {
  if (!n || !q || checked_product(n, q) != flags.size() ||
      flags.size() > INT_MAX ||
      std::any_of(flags.begin(), flags.end(), [](uint32_t v) { return v > 1; }))
    throw std::invalid_argument(
        "exclusive-scan input must contain q nonempty segments of 0/1 flags");
  Resources resources;
  auto stream = resources.stream();
  auto input = resources.allocate<uint32_t>(flags.size()),
       output = resources.allocate<uint32_t>(flags.size());
  ScanPlan plan(resources, n, q);
  check(cudaMemcpyAsync(input, flags.data(), flags.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream),
        "upload scan flags");
  plan.launch(input, output, stream);
  std::vector<uint32_t> result(flags.size());
  check(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "download scan positions");
  check(cudaStreamSynchronize(stream), "finish debug exclusive scan");
  return result;
}
} // namespace gpu_knn
