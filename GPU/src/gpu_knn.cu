#include "gpu_knn.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpu_knn {
namespace {

void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

template <typename T> void device_allocate(T **pointer, std::size_t count) {
  const std::size_t bytes = checked_product(count, sizeof(T));
  cuda_check(cudaMalloc(reinterpret_cast<void **>(pointer), bytes),
             "cudaMalloc");
}

template <typename T> void device_release(T *pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
  }
}

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t next_power_of_two(std::size_t value) {
  if (value == 0) {
    return 1;
  }
  std::size_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("next power of two overflow");
    }
    result *= 2;
  }
  return result;
}

std::size_t ceil_div(std::size_t a, std::size_t b) {
  return a / b + static_cast<std::size_t>(a % b != 0);
}

__device__ __forceinline__ Candidate invalid_candidate() {
  return Candidate{CUDART_INF_F, invalid_id};
}

__host__ __device__ __forceinline__ bool candidate_better(const Candidate &a,
                                                           const Candidate &b) {
  if (a.id == invalid_id || b.id == invalid_id) {
    return a.id != invalid_id && b.id == invalid_id;
  }
  return a.distance < b.distance ||
         (a.distance == b.distance && a.id < b.id);
}

__device__ __forceinline__ float
thread_distance(const float *query, const float *row, std::size_t d) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < d; ++j) {
    const float delta = query[j] - row[j];
    sum += delta * delta;
  }
  return sum == 0.0f ? 0.0f : sum;
}

__device__ __forceinline__ float
warp_distance(const float *query, const float *row, std::size_t d,
              unsigned lane) {
  float sum = 0.0f;
  for (std::size_t j = lane; j < d; j += 32) {
    const float delta = query[j] - row[j];
    sum += delta * delta;
  }
  constexpr unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  return sum == 0.0f ? 0.0f : sum;
}

__global__ void materialized_thread_distance_kernel(
    const float *x, const float *queries, float *distances, std::size_t n,
    std::size_t d, std::size_t q) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = n * q;
  if (index >= total) {
    return;
  }
  const std::size_t query_id = index / n;
  const std::size_t sample_id = index - query_id * n;
  distances[index] =
      thread_distance(queries + query_id * d, x + sample_id * d, d);
}

__global__ void materialized_warp_distance_kernel(
    const float *x, const float *queries, float *distances, std::size_t n,
    std::size_t d, std::size_t q) {
  const std::size_t global_thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t warp_id = global_thread / 32;
  const unsigned lane = threadIdx.x & 31u;
  const std::size_t total = n * q;
  if (warp_id >= total) {
    return;
  }
  const std::size_t query_id = warp_id / n;
  const std::size_t sample_id = warp_id - query_id * n;
  const float sum = warp_distance(queries + query_id * d,
                                  x + sample_id * d, d, lane);
  if (lane == 0) {
    distances[warp_id] = sum;
  }
}

__global__ void pack_keys_kernel(const float *distances, std::uint64_t *keys,
                                 std::size_t n, std::size_t total) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float distance = distances[index];
  const std::uint32_t bits = __float_as_uint(distance);
  const std::uint32_t id = static_cast<std::uint32_t>(index % n);
  keys[index] = (static_cast<std::uint64_t>(bits) << 32) | id;
}

__global__ void gather_keys_kernel(const std::uint64_t *sorted_keys,
                                   const std::int32_t *labels,
                                   std::uint32_t *output_ids,
                                   float *output_distances,
                                   std::int32_t *output_labels, std::size_t n,
                                   std::size_t k, std::size_t q) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = q * k;
  if (index >= total) {
    return;
  }
  const std::size_t query_id = index / k;
  const std::size_t rank = index - query_id * k;
  const std::uint64_t key = sorted_keys[query_id * n + rank];
  const std::uint32_t id = static_cast<std::uint32_t>(key);
  const std::uint32_t distance_bits = static_cast<std::uint32_t>(key >> 32);
  output_ids[index] = id;
  output_distances[index] = __uint_as_float(distance_bits);
  output_labels[index] = labels[id];
}

__device__ void bitonic_sort_shared(Candidate *values, unsigned count) {
  const unsigned index = threadIdx.x;
  for (unsigned size = 2; size <= count; size <<= 1) {
    for (unsigned stride = size >> 1; stride > 0; stride >>= 1) {
      if (index < count) {
        const unsigned partner = index ^ stride;
        if (partner > index) {
          const bool ascending = (index & size) == 0;
          const Candidate left = values[index];
          const Candidate right = values[partner];
          const bool swap = ascending ? candidate_better(right, left)
                                      : candidate_better(left, right);
          if (swap) {
            values[index] = right;
            values[partner] = left;
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void fused_thread_bitonic_kernel(
    const float *x, const float *queries, Candidate *local_topk,
    std::size_t n, std::size_t d, std::size_t q, unsigned tile_size,
    unsigned k, std::size_t tile_count) {
  extern __shared__ Candidate shared[];
  const std::size_t tile_id = blockIdx.x;
  const std::size_t query_id = blockIdx.y;
  if (tile_id >= tile_count || query_id >= q) {
    return;
  }
  const unsigned local = threadIdx.x;
  if (local < tile_size) {
    const std::size_t sample_id = tile_id * tile_size + local;
    Candidate candidate = invalid_candidate();
    if (sample_id < n) {
      candidate.distance = thread_distance(queries + query_id * d,
                                           x + sample_id * d, d);
      candidate.id = static_cast<std::uint32_t>(sample_id);
    }
    shared[local] = candidate;
  }
  __syncthreads();
  bitonic_sort_shared(shared, tile_size);
  if (local < k) {
    const std::size_t output =
        (query_id * tile_count + tile_id) * k + local;
    local_topk[output] = shared[local];
  }
}

__global__ void fused_warp_bitonic_kernel(
    const float *x, const float *queries, Candidate *local_topk,
    std::size_t n, std::size_t d, std::size_t q, unsigned tile_size,
    unsigned k, std::size_t tile_count) {
  extern __shared__ Candidate shared[];
  const std::size_t tile_id = blockIdx.x;
  const std::size_t query_id = blockIdx.y;
  if (tile_id >= tile_count || query_id >= q) {
    return;
  }
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x / 32;
  const unsigned warps_per_block = blockDim.x / 32;
  for (unsigned local = warp; local < tile_size;
       local += warps_per_block) {
    const std::size_t sample_id = tile_id * tile_size + local;
    Candidate candidate = invalid_candidate();
    if (sample_id < n) {
      const float sum = warp_distance(queries + query_id * d,
                                      x + sample_id * d, d, lane);
      if (lane == 0) {
        candidate.distance = sum;
        candidate.id = static_cast<std::uint32_t>(sample_id);
      }
    }
    if (lane == 0) {
      shared[local] = candidate;
    }
  }
  __syncthreads();
  bitonic_sort_shared(shared, tile_size);
  if (threadIdx.x < k) {
    const std::size_t output =
        (query_id * tile_count + tile_id) * k + threadIdx.x;
    local_topk[output] = shared[threadIdx.x];
  }
}

__global__ void merge_topk_kernel(const Candidate *input, Candidate *output,
                                  std::size_t input_list_count,
                                  std::size_t output_list_count, unsigned k,
                                  unsigned padded_k) {
  extern __shared__ Candidate shared[];
  const std::size_t output_list = blockIdx.x;
  const std::size_t query_id = blockIdx.y;
  const unsigned index = threadIdx.x;
  const unsigned count = 2 * padded_k;
  if (output_list >= output_list_count || index >= count) {
    return;
  }

  const bool right_half = index >= padded_k;
  const std::size_t source_list =
      output_list * 2 + static_cast<std::size_t>(right_half);
  const unsigned rank = right_half ? index - padded_k : index;
  Candidate candidate = invalid_candidate();
  if (source_list < input_list_count && rank < k) {
    candidate = input[(query_id * input_list_count + source_list) * k + rank];
  }
  shared[index] = candidate;
  __syncthreads();
  bitonic_sort_shared(shared, count);
  if (index < k) {
    output[(query_id * output_list_count + output_list) * k + index] =
        shared[index];
  }
}

__global__ void gather_candidates_kernel(
    const Candidate *candidates, const std::int32_t *labels,
    std::uint32_t *output_ids, float *output_distances,
    std::int32_t *output_labels, std::size_t total) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const Candidate candidate = candidates[index];
  output_ids[index] = candidate.id;
  output_distances[index] = candidate.distance;
  output_labels[index] = labels[candidate.id];
}

void check_launch(const char *operation) {
  cuda_check(cudaPeekAtLastError(), operation);
}

std::size_t merge_level_count(std::size_t lists) {
  std::size_t levels = 0;
  while (lists > 1) {
    lists = ceil_div(lists, std::size_t{2});
    ++levels;
  }
  return levels;
}

} // namespace

struct Runner::Impl {
  enum EventIndex {
    H2D_START,
    H2D_END,
    DEVICE_START,
    DISTANCE_START,
    DISTANCE_END,
    PACK_START,
    PACK_END,
    SELECT_START,
    SELECT_END,
    MERGE_START,
    MERGE_END,
    GATHER_START,
    GATHER_END,
    DEVICE_END,
    D2H_START,
    D2H_END,
    EVENT_COUNT
  };

  const HostInput *host = nullptr;
  SearchConfig config;
  std::size_t n = 0, d = 0, q = 0, k = 0, total = 0;
  std::size_t tile_count = 0, padded_k = 0, levels = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t events[EVENT_COUNT]{};

  float *d_x = nullptr;
  float *d_queries = nullptr;
  std::int32_t *d_labels = nullptr;
  std::uint32_t *d_output_ids = nullptr;
  float *d_output_distances = nullptr;
  std::int32_t *d_output_labels = nullptr;

  float *d_all_distances = nullptr;
  std::uint64_t *d_keys_in = nullptr;
  std::uint64_t *d_keys_out = nullptr;
  int *d_offsets = nullptr;
  void *d_cub_temp = nullptr;
  std::size_t cub_temp_bytes = 0;

  Candidate *d_candidates_a = nullptr;
  Candidate *d_candidates_b = nullptr;

  MemoryStats memory_stats;
  std::vector<std::uint32_t> h_output_ids;
  std::vector<float> h_output_distances;
  std::vector<std::int32_t> h_output_labels;

  Impl(const HostInput &input, SearchConfig cfg)
      : host(&input), config(cfg), n(input.shape.n), d(input.shape.d),
        q(input.shape.q), k(input.shape.k), total(checked_product(q, n)) {
    validate_input(input);
    if (total > static_cast<std::size_t>(INT_MAX) || n > INT_MAX ||
        q > INT_MAX) {
      throw std::overflow_error(
          "CUB segmented sort and launch metadata require values <= INT_MAX");
    }
    if (config.route == Route::FusedTiledBitonic) {
      if (!is_power_of_two(config.tile_size) || config.tile_size < 32 ||
          config.tile_size > 256) {
        throw std::invalid_argument(
            "fused tile size must be a power of two in [32, 256]");
      }
      if (k > config.tile_size || k > 128) {
        throw std::invalid_argument(
            "fused route requires k <= tile size and k <= 128");
      }
      tile_count = ceil_div(n, config.tile_size);
      padded_k = next_power_of_two(k);
      levels = merge_level_count(tile_count);
    }

    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");
      for (auto &event : events) {
        cuda_check(cudaEventCreate(&event), "cudaEventCreate");
      }

      device_allocate(&d_x, checked_product(n, d));
    device_allocate(&d_queries, checked_product(q, d));
    device_allocate(&d_labels, n);
    const std::size_t output_count = checked_product(q, k);
    device_allocate(&d_output_ids, output_count);
    device_allocate(&d_output_distances, output_count);
    device_allocate(&d_output_labels, output_count);
    h_output_ids.resize(output_count);
    h_output_distances.resize(output_count);
    h_output_labels.resize(output_count);

    if (config.route == Route::MaterializedFullSort) {
      device_allocate(&d_all_distances, total);
      device_allocate(&d_keys_in, total);
      device_allocate(&d_keys_out, total);
      device_allocate(&d_offsets, q + 1);
      std::vector<int> offsets(q + 1);
      for (std::size_t index = 0; index <= q; ++index) {
        offsets[index] = static_cast<int>(index * n);
      }
      cuda_check(cudaMemcpyAsync(d_offsets, offsets.data(),
                                 (q + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
                 "copy segmented-sort offsets");
      cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                     nullptr, cub_temp_bytes, d_keys_in, d_keys_out,
                     static_cast<int>(total), static_cast<int>(q), d_offsets,
                     d_offsets + 1, 0, 64, stream),
                 "query CUB segmented radix sort storage");
      if (cub_temp_bytes != 0) {
        cuda_check(cudaMalloc(&d_cub_temp, cub_temp_bytes),
                   "allocate CUB segmented radix sort storage");
      }
    } else {
      const std::size_t candidates =
          checked_product(checked_product(q, tile_count), k);
      device_allocate(&d_candidates_a, candidates);
      device_allocate(&d_candidates_b, candidates);
    }

      upload_inputs();
      cuda_check(cudaStreamSynchronize(stream), "initial input upload");
      calculate_memory_stats();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  void cleanup() noexcept {
    device_release(d_candidates_b);
    device_release(d_candidates_a);
    if (d_cub_temp != nullptr) {
      cudaFree(d_cub_temp);
    }
    device_release(d_offsets);
    device_release(d_keys_out);
    device_release(d_keys_in);
    device_release(d_all_distances);
    device_release(d_output_labels);
    device_release(d_output_distances);
    device_release(d_output_ids);
    device_release(d_labels);
    device_release(d_queries);
    device_release(d_x);
    for (auto &event : events) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    d_candidates_b = nullptr;
    d_candidates_a = nullptr;
    d_cub_temp = nullptr;
    d_offsets = nullptr;
    d_keys_out = nullptr;
    d_keys_in = nullptr;
    d_all_distances = nullptr;
    d_output_labels = nullptr;
    d_output_distances = nullptr;
    d_output_ids = nullptr;
    d_labels = nullptr;
    d_queries = nullptr;
    d_x = nullptr;
  }

  void calculate_memory_stats() {
    const std::size_t input_values =
        checked_product(n, d) + checked_product(q, d);
    memory_stats.input_bytes =
        checked_product(input_values, sizeof(float)) +
        checked_product(n, sizeof(std::int32_t));
    const std::size_t output_count = checked_product(q, k);
    memory_stats.output_bytes =
        checked_product(output_count, sizeof(std::uint32_t) + sizeof(float) +
                                           sizeof(std::int32_t));

    if (config.route == Route::MaterializedFullSort) {
      memory_stats.distance_buffer_bytes =
          checked_product(total, sizeof(float));
      memory_stats.key_buffer_bytes =
          checked_product(total, 2 * sizeof(std::uint64_t));
      memory_stats.cub_temp_bytes = cub_temp_bytes;
      memory_stats.auxiliary_bytes =
          memory_stats.distance_buffer_bytes + memory_stats.key_buffer_bytes +
          checked_product(q + 1, sizeof(int)) + cub_temp_bytes;
      memory_stats.dynamic_shared_bytes = 0;
    } else {
      const std::size_t candidates =
          checked_product(checked_product(q, tile_count), k);
      memory_stats.candidate_buffer_bytes =
          checked_product(candidates, 2 * sizeof(Candidate));
      memory_stats.auxiliary_bytes = memory_stats.candidate_buffer_bytes;
      memory_stats.dynamic_shared_bytes =
          std::max(config.tile_size * sizeof(Candidate),
                   2 * padded_k * sizeof(Candidate));
    }
    memory_stats.peak_allocated_bytes =
        memory_stats.input_bytes + memory_stats.output_bytes +
        memory_stats.auxiliary_bytes;
  }

  void upload_inputs() {
    cuda_check(cudaMemcpyAsync(d_x, host->x.data(),
                               host->x.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "copy X to device");
    cuda_check(cudaMemcpyAsync(d_queries, host->queries.data(),
                               host->queries.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "copy Q to device");
    cuda_check(cudaMemcpyAsync(d_labels, host->labels.data(),
                               host->labels.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "copy labels to device");
  }

  static double elapsed(cudaEvent_t start, cudaEvent_t end) {
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, start, end),
               "cudaEventElapsedTime");
    return milliseconds;
  }

  void launch_materialized_distance() {
    constexpr unsigned threads = 256;
    if (config.mapping == Mapping::ThreadPerVector) {
      const std::size_t blocks = ceil_div(total, std::size_t{threads});
      materialized_thread_distance_kernel<<<static_cast<unsigned>(blocks),
                                            threads, 0, stream>>>(
          d_x, d_queries, d_all_distances, n, d, q);
      check_launch("materialized thread-per-vector distance kernel");
    } else {
      const std::size_t total_threads = checked_product(total, 32);
      const std::size_t blocks = ceil_div(total_threads,
                                          static_cast<std::size_t>(threads));
      materialized_warp_distance_kernel<<<static_cast<unsigned>(blocks),
                                          threads, 0, stream>>>(
          d_x, d_queries, d_all_distances, n, d, q);
      check_launch("materialized warp-per-vector distance kernel");
    }
  }

  void launch_pack() {
    constexpr unsigned threads = 256;
    const std::size_t blocks = ceil_div(total, std::size_t{threads});
    pack_keys_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        d_all_distances, d_keys_in, n, total);
    check_launch("pack materialized distance keys");
  }

  void launch_full_sort() {
    cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                   d_cub_temp, cub_temp_bytes, d_keys_in, d_keys_out,
                   static_cast<int>(total), static_cast<int>(q), d_offsets,
                   d_offsets + 1, 0, 64, stream),
               "CUB segmented radix full sort");
  }

  void launch_fused_local() {
    const dim3 grid(static_cast<unsigned>(tile_count),
                    static_cast<unsigned>(q));
    const std::size_t shared_bytes = config.tile_size * sizeof(Candidate);
    if (config.mapping == Mapping::ThreadPerVector) {
      const unsigned threads = static_cast<unsigned>(config.tile_size);
      fused_thread_bitonic_kernel<<<grid, threads, shared_bytes, stream>>>(
          d_x, d_queries, d_candidates_a, n, d, q,
          static_cast<unsigned>(config.tile_size), static_cast<unsigned>(k),
          tile_count);
      check_launch("fused thread-per-vector local Bitonic kernel");
    } else {
      constexpr unsigned threads = 256;
      fused_warp_bitonic_kernel<<<grid, threads, shared_bytes, stream>>>(
          d_x, d_queries, d_candidates_a, n, d, q,
          static_cast<unsigned>(config.tile_size), static_cast<unsigned>(k),
          tile_count);
      check_launch("fused warp-per-vector local Bitonic kernel");
    }
  }

  Candidate *launch_merge_tree() {
    Candidate *input = d_candidates_a;
    Candidate *output = d_candidates_b;
    std::size_t lists = tile_count;
    while (lists > 1) {
      const std::size_t output_lists = ceil_div(lists, std::size_t{2});
      const unsigned threads = static_cast<unsigned>(2 * padded_k);
      const dim3 grid(static_cast<unsigned>(output_lists),
                      static_cast<unsigned>(q));
      merge_topk_kernel<<<grid, threads, threads * sizeof(Candidate), stream>>>(
          input, output, lists, output_lists, static_cast<unsigned>(k),
          static_cast<unsigned>(padded_k));
      check_launch("hierarchical Bitonic candidate merge kernel");
      std::swap(input, output);
      lists = output_lists;
    }
    return input;
  }

  void launch_gather_keys() {
    constexpr unsigned threads = 256;
    const std::size_t output_count = checked_product(q, k);
    const std::size_t blocks = ceil_div(output_count,
                                        static_cast<std::size_t>(threads));
    gather_keys_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        d_keys_out, d_labels, d_output_ids, d_output_distances,
        d_output_labels, n, k, q);
    check_launch("gather sorted materialized Top-k");
  }

  void launch_gather_candidates(const Candidate *final_candidates) {
    constexpr unsigned threads = 256;
    const std::size_t output_count = checked_product(q, k);
    const std::size_t blocks = ceil_div(output_count,
                                        static_cast<std::size_t>(threads));
    gather_candidates_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                               stream>>>(
        final_candidates, d_labels, d_output_ids, d_output_distances,
        d_output_labels, output_count);
    check_launch("gather fused Top-k candidates");
  }

  SearchResult run(bool copy_inputs, bool copy_outputs) {
    SearchResult result;
    result.memory = memory_stats;
    result.merge_levels = levels;
    const auto wall_start = std::chrono::steady_clock::now();

    if (copy_inputs) {
      cuda_check(cudaEventRecord(events[H2D_START], stream),
                 "record H2D start");
      upload_inputs();
      cuda_check(cudaEventRecord(events[H2D_END], stream), "record H2D end");
    }

    cuda_check(cudaEventRecord(events[DEVICE_START], stream),
               "record device start");
    Candidate *final_candidates = nullptr;
    if (config.route == Route::MaterializedFullSort) {
      result.times.local_selection_ms =
          std::numeric_limits<double>::quiet_NaN();
      result.times.merge_ms = std::numeric_limits<double>::quiet_NaN();

      cuda_check(cudaEventRecord(events[DISTANCE_START], stream),
                 "record distance start");
      launch_materialized_distance();
      cuda_check(cudaEventRecord(events[DISTANCE_END], stream),
                 "record distance end");

      cuda_check(cudaEventRecord(events[PACK_START], stream),
                 "record pack start");
      launch_pack();
      cuda_check(cudaEventRecord(events[PACK_END], stream),
                 "record pack end");

      cuda_check(cudaEventRecord(events[SELECT_START], stream),
                 "record sort start");
      launch_full_sort();
      cuda_check(cudaEventRecord(events[SELECT_END], stream),
                 "record sort end");
    } else {
      result.times.distance_ms = std::numeric_limits<double>::quiet_NaN();
      result.times.pack_ms = std::numeric_limits<double>::quiet_NaN();
      result.times.full_sort_ms = std::numeric_limits<double>::quiet_NaN();

      cuda_check(cudaEventRecord(events[SELECT_START], stream),
                 "record fused local-selection start");
      launch_fused_local();
      cuda_check(cudaEventRecord(events[SELECT_END], stream),
                 "record fused local-selection end");

      cuda_check(cudaEventRecord(events[MERGE_START], stream),
                 "record merge start");
      final_candidates = launch_merge_tree();
      cuda_check(cudaEventRecord(events[MERGE_END], stream),
                 "record merge end");
    }

    cuda_check(cudaEventRecord(events[GATHER_START], stream),
               "record gather start");
    if (config.route == Route::MaterializedFullSort) {
      launch_gather_keys();
    } else {
      launch_gather_candidates(final_candidates);
    }
    cuda_check(cudaEventRecord(events[GATHER_END], stream),
               "record gather end");
    cuda_check(cudaEventRecord(events[DEVICE_END], stream),
               "record device end");

    if (copy_outputs) {
      const std::size_t output_count = checked_product(q, k);
      cuda_check(cudaEventRecord(events[D2H_START], stream),
                 "record D2H start");
      cuda_check(cudaMemcpyAsync(h_output_ids.data(), d_output_ids,
                                 output_count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream),
                 "copy output IDs to host");
      cuda_check(cudaMemcpyAsync(h_output_distances.data(), d_output_distances,
                                 output_count * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream),
                 "copy output distances to host");
      cuda_check(cudaMemcpyAsync(h_output_labels.data(), d_output_labels,
                                 output_count * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, stream),
                 "copy output labels to host");
      cuda_check(cudaEventRecord(events[D2H_END], stream), "record D2H end");
      cuda_check(cudaEventSynchronize(events[D2H_END]),
                 "synchronize output transfer");
    } else {
      cuda_check(cudaEventSynchronize(events[DEVICE_END]),
                 "synchronize device search");
    }

    const auto wall_end = std::chrono::steady_clock::now();
    result.times.end_to_end_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    if (copy_inputs) {
      result.times.h2d_ms = elapsed(events[H2D_START], events[H2D_END]);
    }
    if (copy_outputs) {
      result.times.d2h_ms = elapsed(events[D2H_START], events[D2H_END]);
    }
    result.times.device_ms =
        elapsed(events[DEVICE_START], events[DEVICE_END]);
    result.times.gather_ms =
        elapsed(events[GATHER_START], events[GATHER_END]);
    if (copy_outputs) {
      // Host vectors are preallocated in the runner. Copying them into the
      // returned value happens after the measured end-to-end interval.
      result.ids = h_output_ids;
      result.distances = h_output_distances;
      result.labels = h_output_labels;
    }
    if (config.route == Route::MaterializedFullSort) {
      result.times.distance_ms =
          elapsed(events[DISTANCE_START], events[DISTANCE_END]);
      result.times.pack_ms = elapsed(events[PACK_START], events[PACK_END]);
      result.times.full_sort_ms =
          elapsed(events[SELECT_START], events[SELECT_END]);
    } else {
      result.times.local_selection_ms =
          elapsed(events[SELECT_START], events[SELECT_END]);
      result.times.merge_ms =
          elapsed(events[MERGE_START], events[MERGE_END]);
    }
    return result;
  }
};

Runner::Runner(const HostInput &input, SearchConfig config)
    : impl_(new Impl(input, config)) {}

Runner::~Runner() { delete impl_; }

Runner::Runner(Runner &&other) noexcept : impl_(other.impl_) {
  other.impl_ = nullptr;
}

Runner &Runner::operator=(Runner &&other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

SearchResult Runner::run(bool copy_inputs, bool copy_outputs) {
  if (impl_ == nullptr) {
    throw std::logic_error("run called on moved-from Runner");
  }
  return impl_->run(copy_inputs, copy_outputs);
}

const MemoryStats &Runner::memory() const {
  if (impl_ == nullptr) {
    throw std::logic_error("memory called on moved-from Runner");
  }
  return impl_->memory_stats;
}

std::size_t Runner::merge_levels() const {
  if (impl_ == nullptr) {
    throw std::logic_error("merge_levels called on moved-from Runner");
  }
  return impl_->levels;
}

std::vector<Candidate>
debug_merge_topk(const std::vector<Candidate> &lists,
                 std::size_t list_count, std::size_t k) {
  if (list_count == 0 || k == 0 ||
      lists.size() != checked_product(list_count, k)) {
    throw std::invalid_argument("invalid flattened candidate-list shape");
  }
  const std::size_t padded_k = next_power_of_two(k);
  if (padded_k > 128) {
    throw std::invalid_argument("debug merge supports k <= 128");
  }

  Candidate *d_a = nullptr;
  Candidate *d_b = nullptr;
  cudaStream_t stream = nullptr;
  try {
    device_allocate(&d_a, lists.size());
    device_allocate(&d_b, lists.size());
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create debug merge stream");
    cuda_check(cudaMemcpyAsync(d_a, lists.data(),
                               lists.size() * sizeof(Candidate),
                               cudaMemcpyHostToDevice, stream),
               "copy debug merge candidates");

    Candidate *input = d_a;
    Candidate *output = d_b;
    std::size_t current_lists = list_count;
    while (current_lists > 1) {
      const std::size_t output_lists =
          ceil_div(current_lists, std::size_t{2});
      const unsigned threads = static_cast<unsigned>(2 * padded_k);
      const dim3 grid(static_cast<unsigned>(output_lists), 1);
      merge_topk_kernel<<<grid, threads, threads * sizeof(Candidate), stream>>>(
          input, output, current_lists, output_lists,
          static_cast<unsigned>(k), static_cast<unsigned>(padded_k));
      check_launch("debug hierarchical merge kernel");
      std::swap(input, output);
      current_lists = output_lists;
    }

    std::vector<Candidate> result(k);
    cuda_check(cudaMemcpyAsync(result.data(), input, k * sizeof(Candidate),
                               cudaMemcpyDeviceToHost, stream),
               "copy debug merge result");
    cuda_check(cudaStreamSynchronize(stream), "synchronize debug merge");
    cudaStreamDestroy(stream);
    device_release(d_b);
    device_release(d_a);
    return result;
  } catch (...) {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    device_release(d_b);
    device_release(d_a);
    throw;
  }
}

} // namespace gpu_knn
