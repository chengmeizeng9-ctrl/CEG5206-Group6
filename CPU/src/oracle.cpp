#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
namespace knn {
Oracle oracle(const Input &i) {
  validate(i);
  floating_environment();
  Oracle o{{}, {}, i.n, i.q};
  o.distances.resize(checked_product(i.n, i.q));
  o.order.resize(o.distances.size());
  for (size_t q = 0; q < i.q; ++q) {
    for (size_t n = 0; n < i.n; ++n) {
      double sum = 0;
      for (size_t d = 0; d < i.d; ++d) {
        double delta =
            double(i.queries[q * i.d + d]) - double(i.x[n * i.d + d]);
        sum += delta * delta;
      }
      o.distances[q * i.n + n] = sum;
      o.order[q * i.n + n] = n;
    }
    auto begin = o.order.begin() + q * i.n;
    std::sort(begin, begin + i.n, [&](uint32_t a, uint32_t b) {
      double da = o.distances[q * i.n + a], db = o.distances[q * i.n + b];
      return da < db || (da == db && a < b);
    });
  }
  return o;
}
Verification verify(const Input &i, Config c, const Result &r,
                    const Oracle &o) {
  Verification v;
  size_t count = checked_product(i.q, c.k);
  if (r.ids.size() != count || r.distances.size() != count ||
      r.labels.size() != count || o.n != i.n || o.q != i.q ||
      o.distances.size() != i.q * i.n || o.order.size() != i.q * i.n) {
    v.structure = false;
    return v;
  }
  for (size_t q = 0; q < i.q; ++q) {
    std::unordered_set<uint32_t> seen;
    std::vector<Candidate> full;
    full.reserve(i.n);
    for (size_t id = 0; id < i.n; ++id) {
      float dist = c.route == Route::Avx2TopKHeap
                       ? simd_distance(i.queries.data() + q * i.d,
                                       i.x.data() + id * i.d, i.d)
                       : scalar_distance(i.queries.data() + q * i.d,
                                         i.x.data() + id * i.d, i.d);
      double reference = o.distances[q * i.n + id];
      double error = std::abs(double(dist) - reference);
      v.max_abs_error = std::max(v.max_abs_error, error);
      v.distance_tolerance &= error <= 1e-5 + 1e-4 * std::abs(reference);
      full.push_back({dist, uint32_t(id)});
    }
    std::sort(full.begin(), full.end(), Better{});
    for (size_t rank = 0; rank < c.k; ++rank) {
      size_t idx = q * c.k + rank;
      auto id = r.ids[idx];
      if (id >= i.n || !seen.insert(id).second ||
          !std::isfinite(r.distances[idx]) || r.distances[idx] < 0 ||
          (r.distances[idx] == 0 && std::signbit(r.distances[idx]))) {
        v.structure = false;
        continue;
      }
      v.labels &= r.labels[idx] == i.labels[id];
      if (rank)
        v.sorted &= !Better{}(Candidate{r.distances[idx], id},
                              Candidate{r.distances[idx - 1], r.ids[idx - 1]});
      double ref = o.distances[q * i.n + id],
             err = std::abs(double(r.distances[idx]) - ref);
      v.max_abs_error = std::max(v.max_abs_error, err);
      v.distance_tolerance &= err <= 1e-5 + 1e-4 * std::abs(ref);
      v.selection &=
          id == full[rank].id && r.distances[idx] == full[rank].distance;
      if (id != o.order[q * i.n + rank]) {
        v.ids = false;
        ++v.id_mismatches;
      }
    }
  }
  return v;
}
} // namespace knn
