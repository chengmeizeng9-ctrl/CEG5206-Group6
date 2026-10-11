#include "knn.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <immintrin.h>
#include <random>
namespace knn {
size_t checked_product(size_t a, size_t b) {
  if (b && a > SIZE_MAX / b)
    throw Error(ErrorCode::SIZE_OVERFLOW, "size multiplication overflow");
  return a * b;
}
void floating_environment() {
  if (std::fesetround(FE_TONEAREST))
    throw Error(ErrorCode::EXECUTION_FAILURE, "rounding mode");
  _mm_setcsr(_mm_getcsr() & ~(unsigned(1 << 15) | unsigned(1 << 6)));
}
Validated validate(const Input &i) {
  if (!i.n || !i.d || !i.q)
    throw Error(ErrorCode::INVALID_ARGUMENT, "N,d,q must be positive");
  if (i.n > UINT32_MAX || i.d > INT32_MAX)
    throw Error(ErrorCode::SIZE_OVERFLOW, "N or d exceeds representation");
  auto nx = checked_product(i.n, i.d), nq = checked_product(i.q, i.d);
  checked_product(nx, sizeof(float));
  checked_product(nq, sizeof(float));
  if (i.x.size() != nx || i.queries.size() != nq || i.labels.size() != i.n)
    throw Error(ErrorCode::INVALID_SHAPE, "X/Q/labels length");
  for (const auto *v : {&i.x, &i.queries})
    for (float f : *v)
      if (!std::isfinite(f) || f < -1 || f > 1)
        throw Error(ErrorCode::INVALID_VALUE,
                    "component outside finite [-1,1]");
  return Validated(i);
}
static void config(const Input &i, Config c) {
  if (!c.k || c.k > i.n || c.route < 0 || c.route > 7)
    throw Error(ErrorCode::INVALID_ARGUMENT, "k or route");
  if (c.route && c.t != 32 && c.t != 64 && c.t != 128 && c.t != 256)
    throw Error(ErrorCode::INVALID_ARGUMENT, "T");
  checked_product(checked_product(i.q, c.k), sizeof(float));
  if ((c.route == 2 || c.route == 4 || c.route == 6) &&
      !__builtin_cpu_supports("avx2"))
    throw Error(ErrorCode::UNSUPPORTED_CONFIGURATION, "AVX2 unavailable");
}
size_t Workspace::bytes() const {
  return (all.capacity() + heap.capacity() + tile.capacity() +
          merged.capacity()) *
         sizeof(Candidate);
}
void prepare(const Validated &v, Config c, Workspace &w, Result &r) {
  const auto &i = v.get();
  config(i, c);
  try {
    size_t out = checked_product(i.q, c.k);
    r.ids.resize(out);
    r.distances.resize(out);
    r.labels.resize(out);
    w.all.resize(c.route <= 2 ? i.n : 0);
    w.heap.clear();
    w.heap.reserve(c.route >= 3 ? c.k : 0);
    w.tile.resize(c.route == 5 || c.route == 6 ? c.t : 0);
    w.merged.resize(c.route == 5 || c.route == 6 ? c.k : 0);
  } catch (const std::bad_alloc &) {
    throw Error(ErrorCode::RESOURCE_EXHAUSTED, "prepare allocation");
  } catch (const std::length_error &) {
    throw Error(ErrorCode::RESOURCE_EXHAUSTED, "prepare capacity");
  }
}
// Our replaceable CPU selection implementation supports shared integration.
void bitonic(Candidate *a, size_t n) {
  if (!n || (n & (n - 1)))
    throw Error(ErrorCode::INVALID_ARGUMENT,
                "Bitonic capacity must be power of two");
  Better better;
  for (size_t width = 2; width <= n; width *= 2) {
    for (size_t stride = width / 2; stride; stride /= 2)
      for (size_t i = 0; i < n; ++i) {
        size_t j = i ^ stride;
        if (j > i) {
          bool asc = (i & width) == 0;
          if (asc ? better(a[j], a[i]) : better(a[i], a[j]))
            std::swap(a[i], a[j]);
        }
      }
    if (width == n)
      break;
  }
}
size_t merge_topk(const Candidate *a, size_t na, const Candidate *b, size_t nb,
                  Candidate *out, size_t k) {
  size_t x = 0, y = 0, z = 0;
  Better better;
  while (z < k && (x < na || y < nb)) {
    out[z++] = (y == nb || (x < na && better(a[x], b[y]))) ? a[x++] : b[y++];
  }
  return z;
}
void search(const Validated &v, Config c, Workspace &w, Result &r,
            Diagnostics *diag) {
  const auto &i = v.get();
  config(i, c);
  size_t out = checked_product(i.q, c.k);
  if (r.ids.size() != out || r.distances.size() != out ||
      r.labels.size() != out || (c.route <= 2 && w.all.size() < i.n) ||
      (c.route >= 3 && w.heap.capacity() < c.k) ||
      ((c.route == 5 || c.route == 6) &&
       (w.tile.size() < c.t || w.merged.size() < c.k)))
    throw Error(ErrorCode::INVALID_SHAPE,
                "prepare workspace/output before search");
  floating_environment();
  if (diag)
    *diag = {};
  Better better;
  bool simd = c.route == 2 || c.route == 4 || c.route == 6;
  for (size_t qi = 0; qi < i.q; ++qi) {
    w.heap.clear();
    size_t kept = 0;
    const float *q = i.queries.data() + qi * i.d;
    size_t tile_size = c.route == 0 ? i.n : c.t;
    for (size_t base = 0; base < i.n;) {
      size_t count = std::min(tile_size, i.n - base);
      for (size_t j = 0; j < count; ++j) {
        size_t id = base + j;
        float distance = 0;
        size_t used = i.d;
        bool skipped = false;
        const float *x = i.x.data() + id * i.d;
        if (c.route == 7 && w.heap.size() == c.k)
          skipped = abandoned_distance(q, x, i.d, w.heap.front().distance,
                                       distance, used);
        else
          distance =
              simd ? simd_distance(q, x, i.d) : scalar_distance(q, x, i.d);
        if (diag) {
          ++diag->candidates;
          diag->dimensions += used;
          diag->abandoned += skipped;
        }
        if (skipped)
          continue;
        Candidate cand{distance, static_cast<uint32_t>(id)};
        if (c.route <= 2)
          w.all[id] = cand;
        else if (c.route == 5 || c.route == 6)
          w.tile[j] = cand;
        else if (w.heap.size() < c.k) {
          w.heap.push_back(cand);
          std::push_heap(w.heap.begin(), w.heap.end(), better);
        } else if (better(cand, w.heap.front())) {
          std::pop_heap(w.heap.begin(), w.heap.end(), better);
          w.heap.back() = cand;
          std::push_heap(w.heap.begin(), w.heap.end(), better);
        }
      }
      if (c.route == 5 || c.route == 6) {
        std::fill(w.tile.begin() + count, w.tile.begin() + c.t,
                  Candidate{INFINITY, invalid_id});
        bitonic(w.tile.data(), c.t);
        size_t local = std::min(c.k, count);
        size_t total = merge_topk(w.heap.data(), kept, w.tile.data(), local,
                                  w.merged.data(), c.k);
        w.heap.assign(w.merged.begin(), w.merged.begin() + total);
        kept = total;
      }
      base += count;
    }
    const Candidate *selected;
    if (c.route <= 2) {
      std::sort(w.all.begin(), w.all.begin() + i.n, better);
      selected = w.all.data();
    } else {
      if (c.route != 5 && c.route != 6)
        std::sort(w.heap.begin(), w.heap.end(), better);
      selected = w.heap.data();
    }
    for (size_t rank = 0; rank < c.k; ++rank) {
      size_t idx = qi * c.k + rank;
      auto cand = selected[rank];
      r.ids[idx] = cand.id;
      r.distances[idx] = cand.distance;
      r.labels[idx] = i.labels[cand.id];
    }
  }
}
Result search(const Input &i, Config c) {
  auto v = validate(i);
  Workspace w;
  Result r;
  prepare(v, c, w, r);
  search(v, c, w, r);
  return r;
}
std::vector<Candidate> select_fixed(std::vector<Candidate> a, size_t k,
                                    int route, size_t t) {
  if (!k || k > a.size() || route < 0 || route > 2)
    throw Error(ErrorCode::INVALID_ARGUMENT, "fixed selection");
  Better b;
  std::vector<Candidate> out;
  if (route == 0) {
    std::sort(a.begin(), a.end(), b);
    a.resize(k);
    return a;
  }
  if (route == 1) {
    out.reserve(k);
    for (auto c : a) {
      if (out.size() < k) {
        out.push_back(c);
        std::push_heap(out.begin(), out.end(), b);
      } else if (b(c, out.front())) {
        std::pop_heap(out.begin(), out.end(), b);
        out.back() = c;
        std::push_heap(out.begin(), out.end(), b);
      }
    }
    std::sort(out.begin(), out.end(), b);
    return out;
  }
  if (t != 32 && t != 64 && t != 128 && t != 256)
    throw Error(ErrorCode::INVALID_ARGUMENT, "fixed T");
  std::vector<Candidate> tile(t), merged(k);
  for (size_t base = 0; base < a.size(); base += t) {
    size_t n = std::min(t, a.size() - base);
    std::copy_n(a.begin() + base, n, tile.begin());
    std::fill(tile.begin() + n, tile.end(), Candidate{INFINITY, invalid_id});
    bitonic(tile.data(), t);
    auto count = merge_topk(out.data(), out.size(), tile.data(), std::min(k, n),
                            merged.data(), k);
    out.assign(merged.begin(), merged.begin() + count);
  }
  return out;
}
Input synthetic(size_t n, size_t d, size_t q, uint32_t seed) {
  Input i{n, d, q, {}, {}, {}};
  if (!n || !d || !q || n > UINT32_MAX || d > INT32_MAX)
    throw Error(ErrorCode::INVALID_ARGUMENT, "synthetic shape");
  i.x.resize(checked_product(n, d));
  i.queries.resize(checked_product(q, d));
  i.labels.resize(n);
  std::mt19937 rng(seed);
  for (auto *v : {&i.x, &i.queries})
    for (float &f : *v)
      f = float(rng() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
  for (size_t j = 0; j < n; ++j)
    i.labels[j] = j % 10;
  return i;
}
const char *route_name(int r) {
  static const char *names[] = {"R0_scalar_sort",  "R1_tiled_scalar_sort",
                                "R2_avx2_sort",    "R3_scalar_heap",
                                "R4_avx2_heap",    "R5_scalar_bitonic",
                                "R6_avx2_bitonic", "R7_early_heap"};
  return r >= 0 && r < 8 ? names[r] : "invalid";
}
} // namespace knn
