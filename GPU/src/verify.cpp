#include "gpu_knn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace gpu_knn {

Fp64Oracle make_fp64_oracle(const HostInput &input) {
  validate_input(input);
  const auto &s = input.shape;
  Fp64Oracle oracle;
  oracle.shape = s;
  const std::size_t total = checked_product(s.q, s.n);
  oracle.distances.resize(total);
  oracle.order.resize(total);

  for (std::size_t qi = 0; qi < s.q; ++qi) {
    const float *query = input.queries.data() + qi * s.d;
    for (std::size_t id = 0; id < s.n; ++id) {
      const float *row = input.x.data() + id * s.d;
      double sum = 0.0;
      for (std::size_t j = 0; j < s.d; ++j) {
        const double delta = static_cast<double>(query[j]) -
                             static_cast<double>(row[j]);
        sum += delta * delta;
      }
      const std::size_t index = qi * s.n + id;
      oracle.distances[index] = sum;
      oracle.order[index] = static_cast<std::uint32_t>(id);
    }
    auto begin = oracle.order.begin() + static_cast<std::ptrdiff_t>(qi * s.n);
    std::sort(begin, begin + static_cast<std::ptrdiff_t>(s.n),
              [&](std::uint32_t a, std::uint32_t b) {
                const double da = oracle.distances[qi * s.n + a];
                const double db = oracle.distances[qi * s.n + b];
                return da < db || (da == db && a < b);
              });
  }
  return oracle;
}

static bool candidate_better(float da, std::uint32_t ia, float db,
                             std::uint32_t ib) {
  return da < db || (da == db && ia < ib);
}

Verification verify_fp64(const HostInput &input, const SearchResult &result,
                         const Fp64Oracle &oracle) {
  validate_input(input);
  const auto &s = input.shape;
  Verification verification;
  const std::size_t expected = checked_product(s.q, s.k);
  if (result.ids.size() != expected || result.distances.size() != expected ||
      result.labels.size() != expected || oracle.shape.n != s.n ||
      oracle.shape.d != s.d || oracle.shape.q != s.q ||
      oracle.distances.size() != checked_product(s.q, s.n) ||
      oracle.order.size() != checked_product(s.q, s.n)) {
    verification.structure = false;
    return verification;
  }

  constexpr double atol = 1e-5;
  constexpr double rtol = 1e-4;
  for (std::size_t qi = 0; qi < s.q; ++qi) {
    std::unordered_set<std::uint32_t> seen;
    seen.reserve(s.k * 2 + 1);
    for (std::size_t rank = 0; rank < s.k; ++rank) {
      const std::size_t index = qi * s.k + rank;
      const std::uint32_t id = result.ids[index];
      const float distance = result.distances[index];
      if (id >= s.n || !seen.insert(id).second || !std::isfinite(distance) ||
          distance < 0.0f || (distance == 0.0f && std::signbit(distance))) {
        verification.structure = false;
        continue;
      }
      verification.labels &= result.labels[index] == input.labels[id];
      if (rank > 0) {
        const std::size_t previous = index - 1;
        verification.sorted &=
            !candidate_better(distance, id, result.distances[previous],
                              result.ids[previous]);
      }

      const double reference = oracle.distances[qi * s.n + id];
      const double error =
          std::abs(static_cast<double>(distance) - reference);
      verification.max_abs_error =
          std::max(verification.max_abs_error, error);
      const double denominator =
          std::max(std::abs(reference), std::numeric_limits<double>::min());
      verification.max_rel_error =
          std::max(verification.max_rel_error, error / denominator);
      verification.distance_tolerance &=
          error <= atol + rtol * std::abs(reference);

      if (id != oracle.order[qi * s.n + rank]) {
        verification.fp64_id_agreement = false;
        ++verification.id_mismatches;
      }
    }
  }
  return verification;
}

bool exact_same_result(const SearchResult &a, const SearchResult &b) {
  if (a.ids != b.ids || a.labels != b.labels ||
      a.distances.size() != b.distances.size()) {
    return false;
  }
  if (a.distances.empty()) {
    return true;
  }
  return std::memcmp(a.distances.data(), b.distances.data(),
                     a.distances.size() * sizeof(float)) == 0;
}

} // namespace gpu_knn
