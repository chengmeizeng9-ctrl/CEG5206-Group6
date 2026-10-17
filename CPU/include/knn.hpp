#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace knn {
constexpr uint32_t invalid_id = UINT32_MAX;
struct Candidate {
  float distance;
  uint32_t id;
};
struct Better {
  bool operator()(const Candidate &a, const Candidate &b) const noexcept {
    if (a.id == invalid_id || b.id == invalid_id)
      return a.id != invalid_id && b.id == invalid_id;
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  }
};
enum class ErrorCode {
  INVALID_ARGUMENT,
  INVALID_SHAPE,
  INVALID_VALUE,
  SIZE_OVERFLOW,
  RESOURCE_EXHAUSTED,
  UNSUPPORTED_CONFIGURATION,
  EXECUTION_FAILURE
};
struct Error : std::runtime_error {
  ErrorCode code;
  Error(ErrorCode c, const std::string &s) : runtime_error(s), code(c) {}
};
struct Input {
  size_t n, d, q;
  std::vector<float> x, queries;
  std::vector<int32_t> labels;
};
// Borrows immutable input: do not modify or destroy Input while using this
// token.
class Validated {
  const Input *input_;
  explicit Validated(const Input &i) : input_(&i) {}
  friend Validated validate(const Input &);

public:
  const Input &get() const { return *input_; }
};
struct Config {
  size_t k, t = 128;
  int route = 0;
};
struct Result {
  std::vector<uint32_t> ids;
  std::vector<float> distances;
  std::vector<int32_t> labels;
};
struct Workspace {
  std::vector<Candidate> all, heap, tile, merged;
  size_t bytes() const;
};
struct Diagnostics {
  uint64_t dimensions = 0, candidates = 0, abandoned = 0;
};
struct Oracle {
  std::vector<double> distances;
  std::vector<uint32_t> order;
  size_t n, q;
};
struct Verification {
  bool structure = true, labels = true, sorted = true,
       distance_tolerance = true, ids = true, selection = true;
  size_t id_mismatches = 0;
  double max_abs_error = 0;
  bool passed() const {
    return structure && labels && sorted && distance_tolerance && selection;
  }
};
size_t checked_product(size_t, size_t);
void floating_environment();
Validated validate(const Input &);
void prepare(const Validated &, Config, Workspace &, Result &);
// Prepared search excludes full input validation; caller owns buffers and
// lifetime.
void search(const Validated &, Config, Workspace &, Result &,
            Diagnostics * = nullptr);
Result search(const Input &, Config);
float scalar_distance(const float *, const float *, size_t);
bool abandoned_distance(const float *, const float *, size_t, float, float &,
                        size_t &);
float simd_distance(const float *, const float *, size_t);
void bitonic(Candidate *, size_t);
size_t merge_topk(const Candidate *, size_t, const Candidate *, size_t,
                  Candidate *, size_t);
std::vector<Candidate> select_fixed(std::vector<Candidate>, size_t, int,
                                    size_t = 128);
Oracle oracle(const Input &);
Verification verify(const Input &, Config, const Result &, const Oracle &);
Input synthetic(size_t, size_t, size_t, uint32_t = 5206);
const char *route_name(int);
} // namespace knn
