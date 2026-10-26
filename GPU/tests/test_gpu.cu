#include "gpu_knn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t checks = 0;
std::size_t cases = 0;
std::size_t near_tie_id_mismatches = 0;

void require(bool condition, const std::string &message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool better(const gpu_knn::Candidate &a, const gpu_knn::Candidate &b) {
  if (a.id == gpu_knn::invalid_id || b.id == gpu_knn::invalid_id) {
    return a.id != gpu_knn::invalid_id && b.id == gpu_knn::invalid_id;
  }
  return a.distance < b.distance ||
         (a.distance == b.distance && a.id < b.id);
}

void exercise(const gpu_knn::HostInput &input, std::size_t tile,
              bool require_fp64_ids = true) {
  ++cases;
  const auto oracle = gpu_knn::make_fp64_oracle(input);
  for (const auto mapping : {gpu_knn::Mapping::ThreadPerVector,
                             gpu_knn::Mapping::WarpPerVector}) {
    gpu_knn::Runner materialized(
        input, {gpu_knn::Route::MaterializedFullSort, mapping, 0});
    gpu_knn::Runner fused(
        input, {gpu_knn::Route::FusedTiledBitonic, mapping, tile});

    const auto materialized_result = materialized.run(false, true);
    const auto fused_result = fused.run(false, true);
    const auto materialized_check =
        gpu_knn::verify_fp64(input, materialized_result, oracle);
    const auto fused_check = gpu_knn::verify_fp64(input, fused_result, oracle);
    require(materialized_check.passed(),
            "materialized route failed FP64/structure verification");
    require(fused_check.passed(),
            "fused route failed FP64/structure verification");
    if (require_fp64_ids) {
      require(materialized_check.fp64_id_agreement &&
                  fused_check.fp64_id_agreement,
              "FP64 Top-k IDs differ on a non-boundary fixture");
    } else {
      near_tie_id_mismatches += materialized_check.id_mismatches +
                                fused_check.id_mismatches;
    }
    require(gpu_knn::exact_same_result(materialized_result, fused_result),
            "fused result differs from same-mapping materialized full sort");

    const auto materialized_repeat = materialized.run(false, true);
    const auto fused_repeat = fused.run(false, true);
    require(gpu_knn::exact_same_result(materialized_result,
                                       materialized_repeat),
            "materialized route is not deterministic");
    require(gpu_knn::exact_same_result(fused_result, fused_repeat),
            "fused route is not deterministic");
  }
}

void test_merge() {
  std::mt19937 rng(5206);
  for (const std::size_t list_count : {std::size_t{1}, std::size_t{2},
                                       std::size_t{3}, std::size_t{5},
                                       std::size_t{9}, std::size_t{31},
                                       std::size_t{32}, std::size_t{33}}) {
    for (const std::size_t k : {std::size_t{1}, std::size_t{5},
                                std::size_t{10}, std::size_t{31},
                                std::size_t{32}, std::size_t{127}}) {
      std::vector<gpu_knn::Candidate> lists(list_count * k);
      std::uint32_t id = 0;
      for (std::size_t list = 0; list < list_count; ++list) {
        for (std::size_t rank = 0; rank < k; ++rank) {
          // Deliberately create many equal distances so ID tie handling is
          // exercised at every merge level.
          lists[list * k + rank] =
              {static_cast<float>((rng() % 11) / 3), id++};
        }
        std::sort(lists.begin() + static_cast<std::ptrdiff_t>(list * k),
                  lists.begin() + static_cast<std::ptrdiff_t>((list + 1) * k),
                  better);
      }
      auto expected = lists;
      std::sort(expected.begin(), expected.end(), better);
      expected.resize(k);
      const auto actual =
          gpu_knn::debug_merge_topk(lists, list_count, k);
      require(actual.size() == k, "GPU merge returned wrong length");
      for (std::size_t rank = 0; rank < k; ++rank) {
        require(actual[rank].id == expected[rank].id &&
                    actual[rank].distance == expected[rank].distance,
                "hierarchical GPU merge differs from full sort");
      }
    }
  }
}

void test_ties() {
  gpu_knn::HostInput zero;
  zero.shape = {40, 9, 2, 35};
  zero.x.assign(zero.shape.n * zero.shape.d, 0.0f);
  zero.queries.assign(zero.shape.q * zero.shape.d, 0.0f);
  zero.labels.resize(zero.shape.n);
  for (std::size_t id = 0; id < zero.shape.n; ++id) {
    zero.labels[id] = static_cast<std::int32_t>(100 - id);
  }
  gpu_knn::validate_input(zero);
  exercise(zero, 64);

  gpu_knn::Runner runner(
      zero, {gpu_knn::Route::FusedTiledBitonic,
             gpu_knn::Mapping::WarpPerVector, 64});
  const auto result = runner.run(false, true);
  for (std::size_t query = 0; query < zero.shape.q; ++query) {
    for (std::size_t rank = 0; rank < zero.shape.k; ++rank) {
      require(result.ids[query * zero.shape.k + rank] == rank,
              "equal-distance results are not ordered by ID");
    }
  }

  gpu_knn::HostInput symmetric;
  symmetric.shape = {5, 2, 2, 3};
  symmetric.x = {1, 0, -1, 0, 0, 1, 0, -1, 0, 0};
  symmetric.queries = {0, 0, 1, 0};
  symmetric.labels = {11, 22, 33, 44, 55};
  gpu_knn::validate_input(symmetric);
  exercise(symmetric, 32);
}

void test_invalid_config() {
  auto input = gpu_knn::synthetic(33, 9, 2, 10);
  bool threw = false;
  try {
    gpu_knn::Runner bad(
        input, {gpu_knn::Route::FusedTiledBitonic,
                gpu_knn::Mapping::ThreadPerVector, 48});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "non-power-of-two tile was accepted");

  input.shape.k = 33;
  threw = false;
  try {
    gpu_knn::Runner bad(
        input, {gpu_knn::Route::FusedTiledBitonic,
                gpu_knn::Mapping::ThreadPerVector, 32});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "k larger than tile was accepted");
}

void test_boundaries() {
  for (const std::size_t tile : {32u, 64u, 128u, 256u}) {
    for (const std::size_t n : {tile - 1, tile, tile + 1,
                                2 * tile - 1, 2 * tile, 2 * tile + 1}) {
      exercise(gpu_knn::synthetic(n, 33, 2, 10), tile);
    }
  }
  for (const std::size_t d : {31u, 32u, 33u, 63u, 64u, 65u}) {
    exercise(gpu_knn::synthetic(129, d, 2, 5), 128);
  }
  for (const std::size_t n : {7u, 32u, 33u, 127u, 128u}) {
    exercise(gpu_knn::synthetic(n, 9, 2, n), 128);
  }
}

void test_near_ties() {
  auto near = gpu_knn::synthetic(8, 9, 2, 4);
  std::fill(near.x.begin(), near.x.end(), 0.5f);
  std::fill(near.queries.begin(), near.queries.end(), 0.0f);
  near.x[0] = std::nextafter(0.5f, 1.0f);
  // FP32 can round distinct FP64 distances to an exact tie. Keep FP64 ID
  // differences visible while requiring deterministic, bitwise R2/R3 equality.
  exercise(near, 32, false);
}

} // namespace

int main() {
  try {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    if (devices == 0) {
      std::cout << "SKIP no CUDA device available\n";
      return 77;
    }

    test_merge();
    exercise(gpu_knn::synthetic(1, 1, 1, 1), 32);
    exercise(gpu_knn::synthetic(257, 9, 3, 10), 32);
    exercise(gpu_knn::synthetic(257, 9, 3, 10), 128);
    exercise(gpu_knn::synthetic(129, 33, 2, 5), 64);
    exercise(gpu_knn::synthetic(65, 65, 1, 32), 32);
    exercise(gpu_knn::synthetic(513, 128, 4, 10), 256);
    test_ties();
    test_invalid_config();
    test_boundaries();
    test_near_ties();

    std::cout << "PASS gpu_cases=" << cases << " checks=" << checks
              << " near_tie_FP64_ID_mismatches=" << near_tie_id_mismatches
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
