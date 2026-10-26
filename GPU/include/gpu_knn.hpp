#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_knn {

constexpr std::uint32_t invalid_id = UINT32_MAX;

struct Candidate {
  float distance = 0.0f;
  std::uint32_t id = invalid_id;
};

struct Shape {
  std::size_t n = 0;
  std::size_t d = 0;
  std::size_t q = 0;
  std::size_t k = 0;
};

struct HostInput {
  Shape shape;
  std::vector<float> x;
  std::vector<float> queries;
  std::vector<std::int32_t> labels;
};

enum class Route {
  MaterializedFullSort,
  FusedTiledBitonic,
};

enum class Mapping {
  ThreadPerVector,
  WarpPerVector,
};

struct SearchConfig {
  Route route = Route::MaterializedFullSort;
  Mapping mapping = Mapping::WarpPerVector;
  std::size_t tile_size = 128;
};

struct StageTimes {
  double h2d_ms = 0.0;
  double distance_ms = 0.0;
  double pack_ms = 0.0;
  double local_selection_ms = 0.0;
  double full_sort_ms = 0.0;
  double merge_ms = 0.0;
  double gather_ms = 0.0;
  double d2h_ms = 0.0;
  double device_ms = 0.0;
  double end_to_end_ms = 0.0;
};

struct MemoryStats {
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t peak_allocated_bytes = 0;
  std::size_t dynamic_shared_bytes = 0;
  std::size_t cub_temp_bytes = 0;
  std::size_t distance_buffer_bytes = 0;
  std::size_t key_buffer_bytes = 0;
  std::size_t candidate_buffer_bytes = 0;
};

struct SearchResult {
  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
  std::vector<std::int32_t> labels;
  StageTimes times;
  MemoryStats memory;
  std::size_t merge_levels = 0;
};

struct Fp64Oracle {
  Shape shape;
  std::vector<double> distances;
  std::vector<std::uint32_t> order;
};

struct Verification {
  bool structure = true;
  bool labels = true;
  bool sorted = true;
  bool distance_tolerance = true;
  bool fp64_id_agreement = true;
  std::size_t id_mismatches = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;

  bool passed() const {
    return structure && labels && sorted && distance_tolerance;
  }
};

std::size_t checked_product(std::size_t a, std::size_t b);
HostInput load_input(const std::string &prefix, const Shape &shape);
HostInput synthetic(std::size_t n, std::size_t d, std::size_t q,
                    std::size_t k, std::uint32_t seed = 5206);
void validate_input(const HostInput &input);
Fp64Oracle make_fp64_oracle(const HostInput &input);
Verification verify_fp64(const HostInput &input, const SearchResult &result,
                         const Fp64Oracle &oracle);
bool exact_same_result(const SearchResult &a, const SearchResult &b);

const char *route_name(Route route);
const char *mapping_name(Mapping mapping);

class Runner {
public:
  Runner(const HostInput &input, SearchConfig config);
  ~Runner();

  Runner(const Runner &) = delete;
  Runner &operator=(const Runner &) = delete;
  Runner(Runner &&) noexcept;
  Runner &operator=(Runner &&) noexcept;

  // copy_inputs=true measures H2D of X, Q and labels for transfer-inclusive
  // timing. copy_outputs=false leaves final Top-k on the device and is used for
  // steady-state GPU-resident timing.
  SearchResult run(bool copy_inputs, bool copy_outputs);
  const MemoryStats &memory() const;
  std::size_t merge_levels() const;

private:
  struct Impl;
  Impl *impl_ = nullptr;
};

// Directly exercises the hierarchical GPU merge. `lists` contains
// list_count sorted lists of exactly k candidates each.
std::vector<Candidate>
debug_merge_topk(const std::vector<Candidate> &lists,
                 std::size_t list_count, std::size_t k);

} // namespace gpu_knn
