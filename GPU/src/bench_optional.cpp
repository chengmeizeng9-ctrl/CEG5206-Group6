#include "knn.hpp"
#include "optional_gpu_knn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sched.h>
#include <set>
#include <sstream>

namespace fs = std::filesystem;
using Record = std::map<std::string, std::string>;
using Clock = std::chrono::steady_clock;
namespace {
std::string number(double value) {
  if (!std::isfinite(value))
    return "";
  std::ostringstream s;
  s << std::setprecision(12) << value;
  return s.str();
}
std::vector<std::string> split(const std::string &text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ','))
    out.push_back(part);
  return out;
}
size_t positive(const std::string &text) {
  size_t pos = 0;
  if (text.empty() || text[0] == '-')
    throw std::invalid_argument("expected positive integer");
  auto n = std::stoull(text, &pos);
  if (pos != text.size() || !n || n > size_t(INT32_MAX))
    throw std::invalid_argument("invalid positive integer");
  return size_t(n);
}
class Csv {
  std::ofstream file;
  std::vector<std::string> fields;

public:
  Csv(const fs::path &path, std::vector<std::string> columns)
      : file(path), fields(std::move(columns)) {
    if (!file)
      throw std::runtime_error("cannot write " + path.string());
    for (size_t i = 0; i < fields.size(); ++i)
      file << (i ? "," : "") << fields[i];
    file << '\n';
  }
  void write(const Record &row) {
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i)
        file << ',';
      auto it = row.find(fields[i]);
      if (it != row.end())
        file << it->second;
    }
    file << '\n';
    file.flush();
    if (!file)
      throw std::runtime_error("CSV write failed");
  }
};
struct Arguments {
  fs::path inputs, output;
  size_t tile = 128, chunk = 32768;
  int warmups = 10, repeats = 20;
  gpu_knn::Mapping mapping = gpu_knn::Mapping::WarpPerVector;
  std::set<std::string> cases;
  bool stream_only = false, validate_only = false;
};
Arguments arguments(int argc, char **argv) {
  Arguments a;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto value = [&]() {
      if (++i >= argc)
        throw std::invalid_argument("missing value");
      return std::string(argv[i]);
    };
    if (key == "--inputs")
      a.inputs = value();
    else if (key == "--output")
      a.output = value();
    else if (key == "--tile")
      a.tile = positive(value());
    else if (key == "--chunk-size")
      a.chunk = positive(value());
    else if (key == "--warmups")
      a.warmups = int(positive(value()));
    else if (key == "--repeats")
      a.repeats = int(positive(value()));
    else if (key == "--mapping") {
      auto m = value();
      if (m != "thread" && m != "warp")
        throw std::invalid_argument("mapping must be thread/warp");
      a.mapping = m == "thread" ? gpu_knn::Mapping::ThreadPerVector
                                : gpu_knn::Mapping::WarpPerVector;
    } else if (key == "--cases") {
      for (auto c : split(value()))
        a.cases.insert(c);
    } else if (key == "--stream-only")
      a.stream_only = true;
    else if (key == "--validate-only")
      a.validate_only = true;
    else if (key == "--smoke") {
      a.cases = {"0"};
      a.warmups = 1;
      a.repeats = 3;
    } else if (key == "--help") {
      std::cout << "knn_optional_bench --inputs CPU_RESULTS --output NEW_DIR "
                   "[--mapping thread|warp] "
                   "[--tile 128] [--chunk-size 32768] [--warmups 10] "
                   "[--repeats 20] [--cases 0,1] [--smoke] [--stream-only] "
                   "[--validate-only]\n";
      std::exit(0);
    } else
      throw std::invalid_argument("unknown argument " + key);
  }
  if (a.inputs.empty() || a.output.empty())
    throw std::invalid_argument("--inputs and --output are required");
  if (a.tile < 32 || a.tile > 256 || (a.tile & (a.tile - 1)))
    throw std::invalid_argument("tile must be 32,64,128,256");
  if (fs::exists(a.output) && !fs::is_empty(a.output))
    throw std::runtime_error("output directory must be new or empty");
  return a;
}
bool same(const knn::Result &a, const knn::Result &b) {
  return a.ids == b.ids && a.labels == b.labels &&
         a.distances.size() == b.distances.size() &&
         std::memcmp(a.distances.data(), b.distances.data(),
                     a.distances.size() * sizeof(float)) == 0;
}
bool fixed_selection(const gpu_knn::HostInput &input,
                     const gpu_knn::SearchResult &result,
                     const std::vector<float> &distances) {
  std::vector<uint32_t> ids(input.shape.n);
  for (size_t qi = 0; qi < input.shape.q; ++qi) {
    std::iota(ids.begin(), ids.end(), 0u);
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
      float da = distances[qi * input.shape.n + a],
            db = distances[qi * input.shape.n + b];
      return da < db || (da == db && a < b);
    });
    for (size_t rank = 0; rank < input.shape.k; ++rank) {
      size_t out = qi * input.shape.k + rank;
      float value = distances[qi * input.shape.n + ids[rank]];
      if (result.ids[out] != ids[rank] ||
          std::memcmp(&value, &result.distances[out], sizeof(float)))
        return false;
    }
  }
  return true;
}
void gpu_times(Record &row, const gpu_knn::SearchResult &r,
               const std::string &mode) {
  const auto &t = r.times;
  row["mode"] = mode;
  row["complete_ms"] =
      number(mode == "resident" ? t.device_ms : t.end_to_end_ms);
  row["device_ms"] = number(t.device_ms);
  row["h2d_ms"] = mode == "resident" ? "" : number(t.h2d_ms);
  row["d2h_ms"] = mode == "resident" ? "" : number(t.d2h_ms);
  row["distance_ms"] = number(t.distance_ms);
  row["pack_ms"] = number(t.pack_ms);
  row["full_sort_ms"] = number(t.full_sort_ms);
  row["local_selection_ms"] = number(t.local_selection_ms);
  row["merge_ms"] = number(t.merge_ms);
  row["gather_ms"] = number(t.gather_ms);
}
void memory_fields(Record &row, const gpu_knn::MemoryStats &m) {
  row["input_device_bytes"] = std::to_string(m.input_bytes);
  row["output_device_bytes"] = std::to_string(m.output_bytes);
  row["aux_bytes"] = std::to_string(m.auxiliary_bytes);
  row["peak_device_bytes"] = std::to_string(m.peak_allocated_bytes);
  row["dynamic_shared_bytes"] = std::to_string(m.dynamic_shared_bytes);
  row["cub_temp_bytes"] = std::to_string(m.cub_temp_bytes);
}
void optional_times(Record &row, const gpu_knn::OptionalResult &r) {
  const auto &t = r.times;
  row["norm_x_ms"] = number(t.norm_x_ms);
  row["norm_q_ms"] = number(t.norm_q_ms);
  row["gemm_ms"] = number(t.gemm_ms);
  row["reconstruct_ms"] = number(t.reconstruct_ms);
  row["radix_ms"] = number(t.radix_ms);
  row["scan_ms"] = number(t.scan_ms);
  row["compact_ms"] = number(t.compact_ms);
  row["pipeline_ms"] = number(t.pipeline_ms);
  row["staging_ms"] = number(t.staging_ms);
  row["chunk_selection_ms"] = number(t.chunk_selection_ms);
  row["compute_service_ms"] = number(t.compute_service_ms);
  row["estimated_overlap_ratio"] = number(t.estimated_overlap_ratio);
  if (row["mode"] != "resident")
    row["near_zero_rechecks"] = std::to_string(r.near_zero_rechecks);
}
struct GpuEntry {
  std::unique_ptr<gpu_knn::Runner> control;
  std::unique_ptr<gpu_knn::OptionalRunner> optional;
  Record row;
  bool resident() const { return control || optional->supports_resident(); }
  gpu_knn::OptionalResult run(bool copy) {
    if (optional)
      return optional->run(copy, copy);
    gpu_knn::OptionalResult result;
    result.output = control->run(copy, copy);
    return result;
  }
};
} // namespace
int main(int argc, char **argv) {
  try {
    auto a = arguments(argc, argv);
    fs::create_directories(a.output);
    std::ifstream preview(a.inputs / "inputs.csv");
    std::string preview_line;
    if (!std::getline(preview, preview_line) ||
        preview_line != "case,N,d,q,k,prefix")
      throw std::runtime_error("invalid CPU input manifest");
    std::set<std::string> available_cases;
    std::vector<std::string> selected_cases;
    while (std::getline(preview, preview_line)) {
      if (preview_line.empty())
        continue;
      auto fields = split(preview_line);
      if (fields.size() != 6)
        throw std::runtime_error("malformed input manifest");
      if (!available_cases.insert(fields[0]).second)
        throw std::runtime_error("duplicate manifest case");
      if (a.cases.empty() || a.cases.count(fields[0]))
        selected_cases.push_back(fields[0]);
    }
    for (const auto &requested : a.cases)
      if (!available_cases.count(requested))
        throw std::invalid_argument("unknown case " + requested);
    if (selected_cases.empty())
      throw std::runtime_error("no input cases selected");
    cpu_set_t allowed, chosen;
    CPU_ZERO(&allowed);
    CPU_ZERO(&chosen);
    if (sched_getaffinity(0, sizeof allowed, &allowed))
      throw std::runtime_error("cannot read CPU affinity");
    int cpu = CPU_ISSET(1, &allowed) ? 1 : -1;
    for (int id = 0; cpu < 0 && id < CPU_SETSIZE; ++id)
      if (CPU_ISSET(id, &allowed))
        cpu = id;
    if (cpu < 0)
      throw std::runtime_error("no CPU available");
    CPU_SET(cpu, &chosen);
    if (sched_setaffinity(0, sizeof chosen, &chosen))
      throw std::runtime_error("cannot pin optional benchmark thread");
    int device = 0;
    cudaDeviceProp gpu{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&gpu, device) != cudaSuccess)
      throw std::runtime_error("CUDA device unavailable");
    std::ofstream meta(a.output / "run.txt");
    meta << "cpu=" << cpu << "\nthreads=1\ngpu=" << gpu.name
         << "\nwarmups=" << a.warmups << "\nrepeats=" << a.repeats
         << "\nmapping=" << gpu_knn::mapping_name(a.mapping)
         << "\ntile=" << a.tile << "\nchunk_size=" << a.chunk
         << "\nstream_only=" << a.stream_only
         << "\nvalidation_only=" << a.validate_only
         << "\ngemm_math=pedantic_fp32_with_near_zero_direct_rechecks"
         << "\nstreaming_timing=end_to_end_only\nstreaming_overlap=service_"
            "time_estimate_not_a_profiler_trace\n";
    meta << "cases=";
    for (size_t i = 0; i < selected_cases.size(); ++i)
      meta << (i ? "," : "") << selected_cases[i];
    meta << '\n';
    meta.flush();
    std::vector<std::string> identity = {
        "case",  "N",       "d",         "q",          "k",      "backend",
        "route", "mapping", "tile_size", "chunk_size", "buffers"};
    auto raw_fields = identity;
    for (auto f : {"mode",
                   "round",
                   "complete_ms",
                   "device_ms",
                   "h2d_ms",
                   "d2h_ms",
                   "distance_ms",
                   "pack_ms",
                   "full_sort_ms",
                   "local_selection_ms",
                   "merge_ms",
                   "gather_ms",
                   "norm_x_ms",
                   "norm_q_ms",
                   "gemm_ms",
                   "reconstruct_ms",
                   "radix_ms",
                   "scan_ms",
                   "compact_ms",
                   "pipeline_ms",
                   "staging_ms",
                   "chunk_selection_ms",
                   "compute_service_ms",
                   "estimated_overlap_ratio",
                   "aux_bytes",
                   "peak_device_bytes",
                   "pinned_host_bytes",
                   "setup_ms",
                   "dimensions_per_candidate",
                   "abandoned_candidates",
                   "near_zero_rechecks"})
      raw_fields.emplace_back(f);
    auto correct_fields = identity;
    for (auto f : {"structure", "labels", "sorted", "distance_tolerance",
                   "FP64_ID_agreement", "ID_mismatches", "max_abs_error",
                   "max_rel_error", "deterministic", "fixed_selection_match",
                   "exact_reference_match", "reference_kind", "passed"})
      correct_fields.emplace_back(f);
    auto mem_fields = identity;
    for (auto f :
         {"input_device_bytes", "output_device_bytes", "aux_bytes",
          "peak_device_bytes", "dynamic_shared_bytes", "cub_temp_bytes",
          "logical_input_bytes", "pinned_host_bytes", "cublas_workspace_bytes",
          "cublas_version", "setup_ms", "database_norm_setup_ms",
          "dimensions_per_candidate", "abandoned_candidates"})
      mem_fields.emplace_back(f);
    Csv raw(a.output / "raw.csv", raw_fields),
        correct(a.output / "correctness.csv", correct_fields),
        memory(a.output / "memory.csv", mem_fields);
    std::ifstream manifest(a.inputs / "inputs.csv");
    std::string line;
    if (!std::getline(manifest, line) || line != "case,N,d,q,k,prefix")
      throw std::runtime_error("invalid CPU input manifest");
    size_t processed = 0;
    while (std::getline(manifest, line)) {
      if (line.empty())
        continue;
      auto f = split(line);
      if (f.size() != 6)
        throw std::runtime_error("malformed input manifest");
      if (!a.cases.empty() && !a.cases.count(f[0]))
        continue;
      ++processed;
      gpu_knn::Shape shape{positive(f[1]), positive(f[2]), positive(f[3]),
                           positive(f[4])};
      auto input = gpu_knn::load_input((a.inputs / f[5]).string(), shape);
      auto oracle = gpu_knn::make_fp64_oracle(input);
      Record common{
          {"case", f[0]}, {"N", f[1]}, {"d", f[2]}, {"q", f[3]}, {"k", f[4]}};
      std::cout << "Optional case " << f[0] << " N=" << shape.n
                << " d=" << shape.d << " q=" << shape.q << " k=" << shape.k
                << std::endl;
      if (!a.stream_only) {
        knn::Input cpu{shape.n, shape.d,       shape.q,
                       input.x, input.queries, input.labels};
        knn::Oracle cpu_oracle{oracle.distances, oracle.order, shape.n,
                               shape.q};
        auto token = knn::validate(cpu);
        auto scalar = knn::search(cpu, {shape.k, knn::Route::ScalarFullSort});
        auto simd = knn::search(cpu, {shape.k, knn::Route::Avx2TopKHeap});
        struct CpuEntry {
          knn::Config config;
          knn::Workspace workspace;
          knn::Result out;
          Record row;
        };
        std::vector<CpuEntry> entries;
        for (auto route :
             {knn::Route::ScalarFullSort, knn::Route::Avx2TopKHeap,
              knn::Route::ScalarTopKHeap, knn::Route::Avx2TiledBitonic,
              knn::Route::ScalarEarlyAbandon}) {
          CpuEntry entry;
          entry.config = {shape.k, route, a.tile};
          entry.row = common;
          entry.row["backend"] = "CPU";
          entry.row["route"] = knn::route_name(route);
          entry.row["mapping"] = (route == knn::Route::Avx2TopKHeap ||
                                  route == knn::Route::Avx2TiledBitonic)
                                     ? "avx2"
                                     : "scalar";
          if (route == knn::Route::Avx2TiledBitonic)
            entry.row["tile_size"] = std::to_string(a.tile);
          knn::prepare(token, entry.config, entry.workspace, entry.out);
          knn::Diagnostics diag;
          knn::search(token, entry.config, entry.workspace, entry.out, &diag);
          const auto saved = entry.out;
          const auto v = knn::verify(cpu, entry.config, entry.out, cpu_oracle);
          knn::search(token, entry.config, entry.workspace, entry.out);
          const bool deterministic = same(saved, entry.out);
          const bool exact =
              same(entry.out, (route == knn::Route::Avx2TopKHeap ||
                               route == knn::Route::Avx2TiledBitonic)
                                  ? simd
                                  : scalar);
          Record row = entry.row;
          row["structure"] = std::to_string(v.structure);
          row["labels"] = std::to_string(v.labels);
          row["sorted"] = std::to_string(v.sorted);
          row["distance_tolerance"] = std::to_string(v.distance_tolerance);
          row["FP64_ID_agreement"] = std::to_string(v.ids);
          row["ID_mismatches"] = std::to_string(v.id_mismatches);
          row["max_abs_error"] = number(v.max_abs_error);
          row["deterministic"] = std::to_string(deterministic);
          row["fixed_selection_match"] = std::to_string(v.selection);
          row["exact_reference_match"] = std::to_string(exact);
          row["reference_kind"] = "arithmetic_matched_cpu_full_selection";
          const bool passed = v.passed() && deterministic && exact;
          row["passed"] = std::to_string(passed);
          correct.write(row);
          if (!passed)
            throw std::runtime_error("optional CPU correctness failed");
          entry.row["dimensions_per_candidate"] =
              number(double(diag.dimensions) / double(diag.candidates));
          entry.row["abandoned_candidates"] =
              std::to_string(diag.abandoned_candidates);
          entry.row["aux_bytes"] = std::to_string(entry.workspace.bytes());
          memory.write(entry.row);
          entries.push_back(std::move(entry));
        }
        for (int warmup = 0; !a.validate_only && warmup < a.warmups; ++warmup)
          for (auto &entry : entries)
            knn::search(token, entry.config, entry.workspace, entry.out);
        for (int round = 0; !a.validate_only && round < a.repeats; ++round)
          for (size_t i = 0; i < entries.size(); ++i) {
            auto &entry = entries[(size_t(round) + i) % entries.size()];
            auto start = Clock::now();
            knn::search(token, entry.config, entry.workspace, entry.out);
            double ms =
                std::chrono::duration<double, std::milli>(Clock::now() - start)
                    .count();
            auto row = entry.row;
            row["mode"] = "host_complete";
            row["round"] = std::to_string(round);
            row["complete_ms"] = number(ms);
            raw.write(row);
          }
      }
      std::unique_ptr<gpu_knn::Runner> direct;
      gpu_knn::SearchResult expected;
      std::vector<GpuEntry> gpu_entries;
      if (!a.stream_only) {
        direct = std::make_unique<gpu_knn::Runner>(
            input, gpu_knn::SearchConfig{gpu_knn::Route::MaterializedFullSort,
                                         a.mapping, 0});
        expected = direct->run(false, true);
        for (auto route : {gpu_knn::Route::MaterializedFullSort,
                           gpu_knn::Route::FusedTiledBitonic}) {
          GpuEntry entry;
          entry.control = std::make_unique<gpu_knn::Runner>(
              input, gpu_knn::SearchConfig{route, a.mapping, a.tile});
          auto &baseline = *entry.control;
          auto first = baseline.run(false, true),
               second = baseline.run(false, true);
          auto v = gpu_knn::verify_fp64(input, first, oracle);
          bool exact = gpu_knn::exact_same_result(first, expected);
          Record row = common;
          row["backend"] = "CUDA";
          row["route"] = gpu_knn::route_name(route);
          row["mapping"] = gpu_knn::mapping_name(a.mapping);
          if (route == gpu_knn::Route::FusedTiledBitonic)
            row["tile_size"] = std::to_string(a.tile);
          row["structure"] = std::to_string(v.structure);
          row["labels"] = std::to_string(v.labels);
          row["sorted"] = std::to_string(v.sorted);
          row["distance_tolerance"] = std::to_string(v.distance_tolerance);
          row["FP64_ID_agreement"] = std::to_string(v.fp64_id_agreement);
          row["ID_mismatches"] = std::to_string(v.id_mismatches);
          row["max_abs_error"] = number(v.max_abs_error);
          row["max_rel_error"] = number(v.max_rel_error);
          bool deterministic = gpu_knn::exact_same_result(first, second),
               passed = v.passed() && exact && deterministic;
          row["deterministic"] = std::to_string(deterministic);
          row["exact_reference_match"] = std::to_string(exact);
          row["reference_kind"] = "materialized_same_mapping";
          row["passed"] = std::to_string(passed);
          correct.write(row);
          if (!passed)
            throw std::runtime_error("optional GPU control correctness failed");
          memory_fields(row, baseline.memory());
          memory.write(row);
          entry.row = std::move(row);
          gpu_entries.push_back(std::move(entry));
        }
      }
      gpu_knn::SearchResult single_reference;
      for (auto route : {gpu_knn::OptionalRoute::RadixSelect,
                         gpu_knn::OptionalRoute::GemmReference,
                         gpu_knn::OptionalRoute::StreamSingle,
                         gpu_knn::OptionalRoute::StreamDouble}) {
        const bool streaming = route == gpu_knn::OptionalRoute::StreamSingle ||
                               route == gpu_knn::OptionalRoute::StreamDouble;
        if (a.stream_only && !streaming)
          continue;
        GpuEntry entry;
        entry.optional = std::make_unique<gpu_knn::OptionalRunner>(
            input, gpu_knn::OptionalConfig{route, a.mapping, a.tile, a.chunk});
        auto &runner = *entry.optional;
        auto first = runner.run(true, true), second = runner.run(true, true);
        auto v = gpu_knn::verify_fp64(input, first.output, oracle);
        bool deterministic =
            gpu_knn::exact_same_result(first.output, second.output);
        bool fixed = true, exact = true;
        if (runner.supports_resident())
          fixed = fixed_selection(input, first.output,
                                  runner.materialized_distances());
        if (route == gpu_knn::OptionalRoute::StreamSingle)
          single_reference = first.output;
        if (route == gpu_knn::OptionalRoute::GemmReference)
          exact = true;
        else if (direct)
          exact = gpu_knn::exact_same_result(first.output, expected);
        else
          exact = gpu_knn::exact_same_result(first.output, single_reference);
        Record row = common;
        row["backend"] = "CUDA";
        row["route"] = gpu_knn::route_name(route);
        row["mapping"] = route == gpu_knn::OptionalRoute::GemmReference
                             ? "cublas_fp32"
                             : gpu_knn::mapping_name(a.mapping);
        if (streaming) {
          row["tile_size"] = std::to_string(a.tile);
          row["chunk_size"] = std::to_string(runner.memory().chunk_size);
          row["buffers"] = std::to_string(runner.memory().buffers);
        }
        row["structure"] = std::to_string(v.structure);
        row["labels"] = std::to_string(v.labels);
        row["sorted"] = std::to_string(v.sorted);
        row["distance_tolerance"] = std::to_string(v.distance_tolerance);
        row["FP64_ID_agreement"] = std::to_string(v.fp64_id_agreement);
        row["ID_mismatches"] = std::to_string(v.id_mismatches);
        row["max_abs_error"] = number(v.max_abs_error);
        row["max_rel_error"] = number(v.max_rel_error);
        row["deterministic"] = std::to_string(deterministic);
        row["fixed_selection_match"] =
            runner.supports_resident() ? std::to_string(fixed) : "";
        row["exact_reference_match"] =
            route == gpu_knn::OptionalRoute::GemmReference
                ? ""
                : std::to_string(exact);
        row["reference_kind"] = route == gpu_knn::OptionalRoute::GemmReference
                                    ? "own_GEMM_distances_full_sort"
                                    : (direct ? "materialized_same_mapping"
                                              : "stream_single_plus_FP64");
        bool passed = v.passed() && deterministic && fixed && exact;
        row["passed"] = std::to_string(passed);
        correct.write(row);
        if (!passed)
          throw std::runtime_error("optional GPU correctness failed in " +
                                   row["route"]);
        const auto &stats = runner.memory();
        memory_fields(row, stats.device);
        row["logical_input_bytes"] = std::to_string(stats.logical_input_bytes);
        row["pinned_host_bytes"] = std::to_string(stats.pinned_host_bytes);
        row["cublas_workspace_bytes"] =
            std::to_string(stats.cublas_workspace_bytes);
        row["setup_ms"] = number(stats.setup_ms);
        row["cublas_version"] = std::to_string(stats.cublas_version);
        row["database_norm_setup_ms"] = number(stats.database_norm_setup_ms);
        memory.write(row);
        entry.row = std::move(row);
        gpu_entries.push_back(std::move(entry));
      }
      direct.reset(); // Release the extra correctness-only full-N reference.
      for (int warmup = 0; !a.validate_only && warmup < a.warmups; ++warmup)
        for (auto &entry : gpu_entries) {
          if (entry.resident())
            entry.run(false);
          entry.run(true);
        }
      // Rotate route order and alternate timing boundaries, as in the core
      // benchmark, rather than always measuring one route in a cold phase.
      for (int round = 0; !a.validate_only && round < a.repeats; ++round)
        for (size_t offset = 0; offset < gpu_entries.size(); ++offset) {
          auto &entry =
              gpu_entries[(size_t(round) + offset) % gpu_entries.size()];
          for (int m = 0; m < (entry.resident() ? 2 : 1); ++m) {
            bool resident = entry.resident() && ((m == 0) != (round % 2 != 0));
            auto result = entry.run(!resident);
            auto record = entry.row;
            gpu_times(record, result.output,
                      resident ? "resident" : "end_to_end");
            if (entry.optional)
              optional_times(record, result);
            record["round"] = std::to_string(round);
            raw.write(record);
          }
        }
    }
    if (!processed)
      throw std::runtime_error("no input cases selected");
    std::cout << (a.validate_only ? "Optional validation complete: "
                                  : "Optional experiment complete: ")
              << a.output << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
