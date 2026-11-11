#pragma once
#include "gpu_knn.hpp"
#include <memory>

namespace gpu_knn {
enum class OptionalRoute {
  RadixSelect,
  GemmReference,
  StreamSingle,
  StreamDouble
};
struct OptionalConfig {
  OptionalRoute route = OptionalRoute::RadixSelect;
  Mapping mapping = Mapping::WarpPerVector;
  std::size_t tile_size = 128;
  std::size_t chunk_size = 32768;
};
struct OptionalTimes {
  double norm_x_ms = 0, norm_q_ms = 0, gemm_ms = 0, reconstruct_ms = 0;
  double radix_ms = 0, scan_ms = 0, compact_ms = 0;
  double pipeline_ms = 0, staging_ms = 0, chunk_selection_ms = 0;
  double compute_service_ms = 0, estimated_overlap_ratio = 0;
};
struct OptionalMemory {
  MemoryStats device;
  std::size_t logical_input_bytes = 0, pinned_host_bytes = 0,
              cublas_workspace_bytes = 0;
  std::size_t chunk_size = 0, buffers = 0;
  double setup_ms = 0, database_norm_setup_ms = 0;
  int cublas_version = 0;
};
struct OptionalResult {
  SearchResult output;
  OptionalTimes times;
  std::uint64_t near_zero_rechecks = 0;
};
const char *route_name(OptionalRoute route);
class OptionalRunner {
public:
  OptionalRunner(const HostInput &, OptionalConfig);
  ~OptionalRunner();
  OptionalRunner(const OptionalRunner &) = delete;
  OptionalRunner &operator=(const OptionalRunner &) = delete;
  OptionalRunner(OptionalRunner &&) noexcept;
  OptionalRunner &operator=(OptionalRunner &&) noexcept;
  // Streaming always transfers chunks and returns outputs, so it only has
  // end-to-end timing. Calling it in resident mode raises invalid_argument.
  OptionalResult run(bool copy_inputs = true, bool copy_outputs = true);
  bool supports_resident() const;
  const OptionalMemory &memory() const;
  // Outside timed intervals: independent fixed-distance selection validation.
  std::vector<float> materialized_distances() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Self-written hierarchical Blelloch scan, independently testable. Values
// are 0/1 flags; each query segment starts its exclusive sum from zero.
std::vector<std::uint32_t>
debug_exclusive_scan(const std::vector<std::uint32_t> &,
                     std::size_t segment_length, std::size_t segments);
} // namespace gpu_knn
