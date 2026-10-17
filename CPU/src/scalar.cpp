#include "knn.hpp"
namespace knn {
float scalar_distance(const float *a, const float *b, size_t d) {
  float s = 0;
  for (size_t j = 0; j < d; ++j) {
    float v = a[j] - b[j];
    float p = v * v;
    s = s + p;
  }
  return s == 0 ? 0 : s;
}
// Non-negative additions in the same FP32 order give a monotone partial sum.
// Equality must continue to preserve the distance/ID tie rule.
bool abandoned_distance(const float *a, const float *b, size_t d, float tau,
                        float &out, size_t &used) {
  float s = 0;
  used = 0;
  for (size_t j = 0; j < d; ++j) {
    float v = a[j] - b[j];
    s = s + v * v;
    ++used;
    if (s > tau)
      return true;
  }
  out = s == 0 ? 0 : s;
  return false;
}
} // namespace knn
