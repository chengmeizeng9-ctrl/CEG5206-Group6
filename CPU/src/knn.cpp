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
static void check_config(const Input &i, Config c) {
  if (!c.k || c.k > i.n)
    throw Error(ErrorCode::INVALID_ARGUMENT, "k");
  if (c.route != Route::ScalarFullSort && c.route != Route::Avx2TopKHeap)
    throw Error(ErrorCode::INVALID_ARGUMENT, "route");
  if (c.route == Route::Avx2TopKHeap && !__builtin_cpu_supports("avx2"))
    throw Error(ErrorCode::UNSUPPORTED_CONFIGURATION, "AVX2 unavailable");
  checked_product(checked_product(i.q, c.k), sizeof(float));
}
size_t Workspace::bytes() const {
  return (all.capacity() + heap.capacity()) * sizeof(Candidate);
}
void prepare(const Validated &v, Config c, Workspace &w, Result &r) {
  const auto &i = v.get();
  check_config(i, c);
  try {
    size_t out = checked_product(i.q, c.k);
    r.ids.resize(out);
    r.distances.resize(out);
    r.labels.resize(out);
    if (c.route == Route::ScalarFullSort) {
      w.all.resize(i.n);
      w.heap.clear();
    } else {
      w.all.clear();
      w.heap.clear();
      w.heap.reserve(c.k);
    }
  } catch (const std::bad_alloc &) {
    throw Error(ErrorCode::RESOURCE_EXHAUSTED, "prepare allocation");
  } catch (const std::length_error &) {
    throw Error(ErrorCode::RESOURCE_EXHAUSTED, "prepare capacity");
  }
}
size_t merge_topk(const Candidate *a, size_t na, const Candidate *b, size_t nb,
                  Candidate *out, size_t k) {
  if ((na && !a) || (nb && !b) || (k && !out))
    throw Error(ErrorCode::INVALID_ARGUMENT, "null candidate merge buffer");
  size_t x = 0, y = 0, z = 0;
  Better better;
  while (z < k && (x < na || y < nb))
    out[z++] = (y == nb || (x < na && better(a[x], b[y]))) ? a[x++] : b[y++];
  return z;
}
static void heap_insert(std::vector<Candidate> &heap, const Candidate &candidate,
                        size_t k) {
  Better better;
  if (heap.size() < k) {
    heap.push_back(candidate);
    std::push_heap(heap.begin(), heap.end(), better);
  } else if (better(candidate, heap.front())) {
    std::pop_heap(heap.begin(), heap.end(), better);
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end(), better);
  }
}
std::vector<Candidate> select_topk_heap(const std::vector<Candidate> &candidates,
                                       size_t k) {
  if (k > candidates.size())
    throw Error(ErrorCode::INVALID_ARGUMENT, "heap k exceeds candidate count");
  if (!k)
    return {};
  std::vector<Candidate> heap;
  heap.reserve(k);
  for (const auto &candidate : candidates)
    heap_insert(heap, candidate, k);
  std::sort(heap.begin(), heap.end(), Better{});
  return heap;
}
void search(const Validated &v, Config c, Workspace &w, Result &r,
            Diagnostics *diag) {
  const auto &i = v.get();
  check_config(i, c);
  size_t out = checked_product(i.q, c.k);
  if (r.ids.size() != out || r.distances.size() != out ||
      r.labels.size() != out ||
      (c.route == Route::ScalarFullSort && w.all.size() < i.n) ||
      (c.route == Route::Avx2TopKHeap && w.heap.capacity() < c.k))
    throw Error(ErrorCode::INVALID_SHAPE,
                "prepare workspace/output before search");
  floating_environment();
  if (diag)
    *diag = {};
  Better better;
  for (size_t qi = 0; qi < i.q; ++qi) {
    w.heap.clear();
    const float *q = i.queries.data() + qi * i.d;
    for (size_t id = 0; id < i.n; ++id) {
      float distance = c.route == Route::Avx2TopKHeap
                           ? simd_distance(q, i.x.data() + id * i.d, i.d)
                           : scalar_distance(q, i.x.data() + id * i.d, i.d);
      if (diag) {
        ++diag->candidates;
        diag->dimensions += i.d;
      }
      Candidate candidate{distance, static_cast<uint32_t>(id)};
      if (c.route == Route::ScalarFullSort)
        w.all[id] = candidate;
      else
        heap_insert(w.heap, candidate, c.k);
    }
    const Candidate *selected;
    if (c.route == Route::ScalarFullSort) {
      std::sort(w.all.begin(), w.all.begin() + i.n, better);
      selected = w.all.data();
    } else {
      std::sort(w.heap.begin(), w.heap.end(), better);
      selected = w.heap.data();
    }
    for (size_t rank = 0; rank < c.k; ++rank) {
      size_t idx = qi * c.k + rank;
      const auto candidate = selected[rank];
      r.ids[idx] = candidate.id;
      r.distances[idx] = candidate.distance;
      r.labels[idx] = i.labels[candidate.id];
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
const char *route_name(Route route) {
  switch (route) {
  case Route::ScalarFullSort:
    return "R0_scalar_full_sort";
  case Route::Avx2TopKHeap:
    return "R1_avx2_topk_heap";
  }
  return "invalid";
}
} // namespace knn
