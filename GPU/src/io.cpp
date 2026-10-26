#include "gpu_knn.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace gpu_knn {

std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error("size multiplication overflow");
  }
  return a * b;
}

template <typename T>
static std::vector<T> load_exact(const std::string &path, std::size_t count) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary input element must be trivially copyable");
  const std::size_t bytes = checked_product(count, sizeof(T));
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  const auto end = stream.tellg();
  if (end < 0 || static_cast<std::uint64_t>(end) != bytes) {
    throw std::runtime_error("unexpected byte length for " + path);
  }
  std::vector<T> values(count);
  stream.seekg(0);
  if (bytes != 0) {
    stream.read(reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(bytes));
  }
  if (!stream) {
    throw std::runtime_error("failed to read " + path);
  }
  return values;
}

void validate_input(const HostInput &input) {
  const auto &s = input.shape;
  if (s.n == 0 || s.d == 0 || s.q == 0 || s.k == 0 || s.k > s.n) {
    throw std::invalid_argument("N, d, q and k must be positive and k <= N");
  }
  if (s.n > UINT32_MAX) {
    throw std::overflow_error("N exceeds uint32 ID representation");
  }
  const std::size_t nx = checked_product(s.n, s.d);
  const std::size_t nq = checked_product(s.q, s.d);
  if (input.x.size() != nx || input.queries.size() != nq ||
      input.labels.size() != s.n) {
    throw std::invalid_argument("input array length does not match shape");
  }
  for (const auto *array : {&input.x, &input.queries}) {
    for (const float value : *array) {
      if (!std::isfinite(value) || value < -1.0f || value > 1.0f) {
        throw std::invalid_argument(
            "input component must be finite and in [-1, 1]");
      }
    }
  }
}

HostInput load_input(const std::string &prefix, const Shape &shape) {
  HostInput input;
  input.shape = shape;
  input.x = load_exact<float>(prefix + ".X.f32",
                              checked_product(shape.n, shape.d));
  input.queries = load_exact<float>(prefix + ".Q.f32",
                                    checked_product(shape.q, shape.d));
  input.labels = load_exact<std::int32_t>(prefix + ".labels.i32", shape.n);
  validate_input(input);
  return input;
}

HostInput synthetic(std::size_t n, std::size_t d, std::size_t q,
                    std::size_t k, std::uint32_t seed) {
  HostInput input;
  input.shape = {n, d, q, k};
  input.x.resize(checked_product(n, d));
  input.queries.resize(checked_product(q, d));
  input.labels.resize(n);

  std::mt19937 rng(seed);
  for (auto *array : {&input.x, &input.queries}) {
    for (float &value : *array) {
      value = static_cast<float>(rng() >> 8) *
                  (1.0f / 16777216.0f) * 2.0f -
              1.0f;
    }
  }
  for (std::size_t id = 0; id < n; ++id) {
    input.labels[id] = static_cast<std::int32_t>(id % 10);
  }
  validate_input(input);
  return input;
}

const char *route_name(Route route) {
  switch (route) {
  case Route::MaterializedFullSort:
    return "R2_cuda_materialized_full_sort";
  case Route::FusedTiledBitonic:
    return "R3_cuda_fused_tiled_bitonic";
  }
  return "invalid_route";
}

const char *mapping_name(Mapping mapping) {
  switch (mapping) {
  case Mapping::ThreadPerVector:
    return "thread_per_vector";
  case Mapping::WarpPerVector:
    return "warp_per_vector";
  }
  return "invalid_mapping";
}

} // namespace gpu_knn
