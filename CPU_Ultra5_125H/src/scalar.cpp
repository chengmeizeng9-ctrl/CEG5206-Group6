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
} // namespace knn
