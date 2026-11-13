#include "knn.hpp"
#include <immintrin.h>
namespace knn {
// Eight lane partial sums, lane-order scalar reduction, then scalar remainder.
float simd_distance(const float *a, const float *b, size_t d) {
  __m256 s = _mm256_setzero_ps();
  size_t j = 0;
  for (; d - j >= 8; j += 8) {
    auto v = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
    s = _mm256_add_ps(s, _mm256_mul_ps(v, v));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, s);
  float out = 0;
  for (float v : lanes)
    out += v;
  for (; j < d; ++j) {
    float v = a[j] - b[j];
    out += v * v;
  }
  return out == 0 ? 0 : out;
}
} // namespace knn
