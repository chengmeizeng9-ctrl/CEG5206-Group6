#include "optional_gpu_knn.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>

namespace {
size_t checks = 0, cases = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}
void selection(const gpu_knn::HostInput &input,
               const gpu_knn::SearchResult &result,
               const std::vector<float> &matrix) {
  std::vector<uint32_t> ids(input.shape.n);
  for (size_t qi = 0; qi < input.shape.q; ++qi) {
    std::iota(ids.begin(), ids.end(), 0u);
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
      float da = matrix[qi * input.shape.n + a],
            db = matrix[qi * input.shape.n + b];
      return da < db || (da == db && a < b);
    });
    for (size_t rank = 0; rank < input.shape.k; ++rank) {
      const size_t out = qi * input.shape.k + rank;
      float distance = matrix[qi * input.shape.n + ids[rank]];
      require(result.ids[out] == ids[rank] &&
                  std::memcmp(&result.distances[out], &distance,
                              sizeof(float)) == 0,
              "optional selection differs from independent full sort of its "
              "fixed distances");
    }
  }
}
void exercise(const gpu_knn::HostInput &input, size_t chunk = 64,
              size_t tile = 128) {
  ++cases;
  auto oracle = gpu_knn::make_fp64_oracle(input);
  for (auto mapping :
       {gpu_knn::Mapping::ThreadPerVector, gpu_knn::Mapping::WarpPerVector}) {
    gpu_knn::Runner baseline(
        input, {gpu_knn::Route::MaterializedFullSort, mapping, 0});
    auto expected = baseline.run(false, true);
    for (auto route : {gpu_knn::OptionalRoute::RadixSelect,
                       gpu_knn::OptionalRoute::GemmReference,
                       gpu_knn::OptionalRoute::StreamSingle,
                       gpu_knn::OptionalRoute::StreamDouble}) {
      gpu_knn::OptionalRunner runner(input, {route, mapping, tile, chunk});
      auto actual = runner.run(true, true);
      require(gpu_knn::verify_fp64(input, actual.output, oracle).passed(),
              "optional GPU numerical/structure check failed");
      if (route != gpu_knn::OptionalRoute::GemmReference)
        require(gpu_knn::exact_same_result(expected, actual.output),
                "optional direct-distance route differs from R2");
      if (runner.supports_resident())
        selection(input, actual.output, runner.materialized_distances());
      const auto repeated = runner.run(true, true);
      require(gpu_knn::exact_same_result(actual.output, repeated.output),
              "optional GPU route is nondeterministic");
      if (runner.supports_resident()) {
        runner.run(false, false);
        require(gpu_knn::exact_same_result(actual.output,
                                           runner.run(false, true).output),
                "resident/transfer-inclusive outputs differ");
      }
      const auto &memory = runner.memory();
      require(memory.device.peak_allocated_bytes ==
                  memory.device.input_bytes + memory.device.output_bytes +
                      memory.device.auxiliary_bytes,
              "inconsistent optional memory accounting");
    }
  }
}
void scans() {
  std::mt19937 rng(5206);
  for (size_t n : {1u, 31u, 255u, 256u, 257u, 4097u, 65537u}) {
    const size_t q = 3;
    std::vector<uint32_t> flags(n * q), expected(n * q);
    for (unsigned kind = 0; kind < 3; ++kind) {
      for (auto &flag : flags)
        flag = kind == 0 ? 0 : (kind == 1 ? 1 : rng() % 2);
      for (size_t qi = 0; qi < q; ++qi) {
        uint32_t sum = 0;
        for (size_t i = 0; i < n; ++i) {
          expected[qi * n + i] = sum;
          sum += flags[qi * n + i];
        }
      }
      require(gpu_knn::debug_exclusive_scan(flags, n, q) == expected,
              "hierarchical GPU scan differs from host exclusive sum");
    }
  }
}
} // namespace
int main() {
  try {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
    if (!devices) {
      std::cout << "SKIP no CUDA device available\n";
      return 77;
    }
    scans();
    exercise(gpu_knn::synthetic(1, 1, 1, 1), 1, 32);
    for (size_t d : {7u, 8u, 9u, 31u, 32u, 33u, 63u, 64u, 65u})
      exercise(gpu_knn::synthetic(257, d, 2, 10), 64, 128);
    exercise(gpu_knn::synthetic(33, 9, 2, 32), 1, 32);
    exercise(gpu_knn::synthetic(127, 33, 2, 127), 31, 128);
    exercise(gpu_knn::synthetic(128, 65, 1, 128), 129, 128);
    exercise(gpu_knn::synthetic(513, 128, 2, 5), 257, 256);
    exercise(gpu_knn::synthetic(66049, 1, 1, 10), 8192, 64);
    auto ties = gpu_knn::synthetic(129, 9, 2, 35);
    std::fill(ties.x.begin(), ties.x.end(), 0.0f);
    std::fill(ties.queries.begin(), ties.queries.end(), 0.0f);
    for (size_t id = 0; id < ties.shape.n; ++id)
      ties.labels[id] = int32_t(1000 - id);
    exercise(ties, 17, 64);
    auto cancellation = gpu_knn::synthetic(17, 128, 2, 5);
    std::fill(cancellation.x.begin(), cancellation.x.end(), 0.5f);
    std::fill(cancellation.queries.begin(), cancellation.queries.end(), 0.5f);
    cancellation.x[0] = std::nextafter(0.5f, 1.0f);
    exercise(cancellation, 7, 32);
    gpu_knn::OptionalRunner gemm(cancellation,
                                 {gpu_knn::OptionalRoute::GemmReference});
    require(gemm.run().near_zero_rechecks > 0,
            "GEMM cancellation correction was not exercised");
    auto small = gpu_knn::synthetic(257, 33, 2, 10),
         large = gpu_knn::synthetic(1025, 33, 2, 10);
    gpu_knn::OptionalConfig config{gpu_knn::OptionalRoute::StreamDouble,
                                   gpu_knn::Mapping::WarpPerVector, 128, 64};
    gpu_knn::OptionalRunner a(small, config), b(large, config);
    require(a.memory().device.peak_allocated_bytes ==
                b.memory().device.peak_allocated_bytes,
            "streaming device storage grows with the database despite a fixed "
            "chunk budget");
    require(b.memory().device.peak_allocated_bytes <
                b.memory().logical_input_bytes,
            "bounded streaming test did not exceed the allocated device-input "
            "budget");
    bool threw = false;
    try {
      a.run(false, false);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    require(threw, "streaming incorrectly advertises resident-only timing");
    threw = false;
    try {
      gpu_knn::debug_exclusive_scan({0, 2}, 2, 1);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    require(threw, "scan accepted flags outside 0/1");
    auto invalid = gpu_knn::synthetic(1, 1, 1, 1);
    invalid.shape.k = SIZE_MAX;
    threw = false;
    try {
      gpu_knn::OptionalRunner bad(invalid, {});
    } catch (const std::exception &) {
      threw = true;
    }
    require(threw, "invalid enormous k did not fail before padding/allocation");
    auto changed = gpu_knn::synthetic(33, 9, 1, 5);
    gpu_knn::OptionalRunner changing_stream(
        changed, {gpu_knn::OptionalRoute::StreamDouble,
                  gpu_knn::Mapping::WarpPerVector, 32, 7});
    changing_stream.run();
    std::fill(changed.queries.begin(), changed.queries.end(), 0.9f);
    gpu_knn::Runner changed_reference(changed,
                                      {gpu_knn::Route::MaterializedFullSort,
                                       gpu_knn::Mapping::WarpPerVector, 0});
    require(gpu_knn::exact_same_result(changing_stream.run().output,
                                       changed_reference.run(false, true)),
            "streaming did not refresh pinned query values on a "
            "transfer-inclusive call");
    std::cout << "PASS optional_gpu_cases=" << cases << " checks=" << checks
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
