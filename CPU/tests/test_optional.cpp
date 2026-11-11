#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
size_t checks = 0, cases = 0;
void require(bool v, const char *s) {
  ++checks;
  if (!v)
    throw std::runtime_error(s);
}
bool same(const knn::Result &a, const knn::Result &b) {
  return a.ids == b.ids && a.labels == b.labels &&
         a.distances.size() == b.distances.size() &&
         std::memcmp(a.distances.data(), b.distances.data(),
                     a.distances.size() * sizeof(float)) == 0;
}
void exercise(const knn::Input &input, size_t k, size_t tile) {
  ++cases;
  const auto oracle = knn::oracle(input);
  for (auto route :
       {knn::Route::Avx2TiledBitonic, knn::Route::ScalarEarlyAbandon}) {
    auto reference =
        knn::search(input, {k, route == knn::Route::Avx2TiledBitonic
                                   ? knn::Route::Avx2TopKHeap
                                   : knn::Route::ScalarFullSort});
    knn::Workspace w;
    knn::Result actual;
    knn::Config cfg{k, route, tile};
    auto validated = knn::validate(input);
    knn::prepare(validated, cfg, w, actual);
    knn::Diagnostics diag;
    knn::search(validated, cfg, w, actual, &diag);
    require(same(actual, reference),
            "optional CPU route differs from its arithmetic-matched baseline");
    if (route == knn::Route::ScalarEarlyAbandon)
      require(same(actual, knn::search(input, {k, knn::Route::ScalarTopKHeap})),
              "early abandonment differs from its full-evaluation scalar heap "
              "control");
    require(knn::verify(input, cfg, actual, oracle).passed(),
            "optional CPU oracle/selection check failed");
    require(diag.candidates == input.n * input.q,
            "wrong examined-candidate count");
    require(diag.dimensions <= input.n * input.q * input.d,
            "dimension counter overflow");
    auto saved = actual;
    knn::search(validated, cfg, w, actual);
    require(same(actual, saved), "optional CPU route is nondeterministic");
  }
}
} // namespace
int main() {
  try {
    for (size_t tile : {32u, 64u, 128u, 256u})
      for (size_t n : {size_t(1), tile - 1, tile, tile + 1, 2 * tile + 1})
        for (size_t d : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 65u}) {
          auto input = knn::synthetic(n, d, 2);
          exercise(input, 1, tile);
          exercise(input, std::min(size_t(10), n), tile);
          exercise(input, n, tile); // Also exercises k larger than one tile.
        }
    auto ties = knn::synthetic(129, 9, 2);
    std::fill(ties.x.begin(), ties.x.end(), 0.5f);
    std::fill(ties.queries.begin(), ties.queries.end(), 0.0f);
    exercise(ties, 35, 32);
    ties.x[0] = std::nextafter(0.5f, 1.0f);
    exercise(ties, 4, 64);
    auto prunable = knn::synthetic(65, 64, 1);
    std::fill(prunable.x.begin(), prunable.x.end(), 1.0f);
    std::fill_n(prunable.x.begin(), 64, 0.0f);
    std::fill(prunable.queries.begin(), prunable.queries.end(), 0.0f);
    knn::Workspace w;
    knn::Result out;
    knn::Diagnostics diag;
    auto token = knn::validate(prunable);
    knn::Config early{1, knn::Route::ScalarEarlyAbandon};
    knn::prepare(token, early, w, out);
    knn::search(token, early, w, out, &diag);
    require(diag.abandoned_candidates == 64, "early abandonment never pruned");
    require(diag.dimensions == 128,
            "early abandonment did not stop after the first exceeding term");
    require(out.ids[0] == 0, "early abandonment selected the wrong neighbour");
    bool threw = false;
    try {
      knn::search(prunable, {1, knn::Route::Avx2TiledBitonic, 48});
    } catch (const knn::Error &) {
      threw = true;
    }
    require(threw, "invalid CPU tile accepted");
    std::cout << "PASS optional_cpu_cases=" << cases << " checks=" << checks
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
