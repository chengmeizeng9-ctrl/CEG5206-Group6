#pragma once
// Shared internal CUDA kernels. Included inside each translation unit's
// anonymous namespace so launch wrappers retain internal linkage.

__device__ __forceinline__ Candidate invalid_candidate() {
  return Candidate{CUDART_INF_F, invalid_id};
}

__host__ __device__ __forceinline__ bool candidate_better(const Candidate &a,
                                                          const Candidate &b) {
  if (a.id == invalid_id || b.id == invalid_id) {
    return a.id != invalid_id && b.id == invalid_id;
  }
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
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

__device__ __forceinline__ float warp_distance(const float *query,
                                               const float *row, std::size_t d,
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

__global__ void
materialized_thread_distance_kernel(const float *x, const float *queries,
                                    float *distances, std::size_t n,
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

__global__ void materialized_warp_distance_kernel(const float *x,
                                                  const float *queries,
                                                  float *distances,
                                                  std::size_t n, std::size_t d,
                                                  std::size_t q) {
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
  const float sum =
      warp_distance(queries + query_id * d, x + sample_id * d, d, lane);
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

__global__ void
fused_thread_bitonic_kernel(const float *x, const float *queries,
                            Candidate *local_topk, std::size_t n, std::size_t d,
                            std::size_t q, unsigned tile_size, unsigned k,
                            std::size_t tile_count, std::size_t id_offset = 0) {
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
      candidate.distance =
          thread_distance(queries + query_id * d, x + sample_id * d, d);
      candidate.id = static_cast<std::uint32_t>(sample_id + id_offset);
    }
    shared[local] = candidate;
  }
  __syncthreads();
  bitonic_sort_shared(shared, tile_size);
  if (local < k) {
    const std::size_t output = (query_id * tile_count + tile_id) * k + local;
    local_topk[output] = shared[local];
  }
}

__global__ void fused_warp_bitonic_kernel(const float *x, const float *queries,
                                          Candidate *local_topk, std::size_t n,
                                          std::size_t d, std::size_t q,
                                          unsigned tile_size, unsigned k,
                                          std::size_t tile_count,
                                          std::size_t id_offset = 0) {
  extern __shared__ Candidate shared[];
  const std::size_t tile_id = blockIdx.x;
  const std::size_t query_id = blockIdx.y;
  if (tile_id >= tile_count || query_id >= q) {
    return;
  }
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x / 32;
  const unsigned warps_per_block = blockDim.x / 32;
  for (unsigned local = warp; local < tile_size; local += warps_per_block) {
    const std::size_t sample_id = tile_id * tile_size + local;
    Candidate candidate = invalid_candidate();
    if (sample_id < n) {
      const float sum =
          warp_distance(queries + query_id * d, x + sample_id * d, d, lane);
      if (lane == 0) {
        candidate.distance = sum;
        candidate.id = static_cast<std::uint32_t>(sample_id + id_offset);
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

__global__ void gather_candidates_kernel(const Candidate *candidates,
                                         const std::int32_t *labels,
                                         std::uint32_t *output_ids,
                                         float *output_distances,
                                         std::int32_t *output_labels,
                                         std::size_t total) {
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
