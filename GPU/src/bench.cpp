#include "gpu_knn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using gpu_knn::Mapping;
using gpu_knn::Route;

namespace {

struct Arguments {
  fs::path inputs;
  fs::path output;
  std::set<std::string> cases;
  std::vector<Route> routes{Route::MaterializedFullSort,
                            Route::FusedTiledBitonic};
  std::vector<Mapping> mappings{Mapping::WarpPerVector};
  std::vector<std::size_t> tiles{128};
  std::vector<std::string> modes{"resident", "end_to_end"};
  int warmups = 10;
  int repeats = 20;
};

struct ManifestRow {
  std::string case_id;
  gpu_knn::Shape shape;
  std::string prefix;
};

struct RunEntry {
  gpu_knn::SearchConfig config;
  std::unique_ptr<gpu_knn::Runner> runner;
  gpu_knn::SearchResult first;
  gpu_knn::Verification verification;
  bool deterministic = false;
  std::optional<bool> materialized_match;
};

std::vector<std::string> split(const std::string &text, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, delimiter)) {
    fields.push_back(field);
  }
  return fields;
}

std::size_t parse_size(const std::string &text, const char *name) {
  std::size_t position = 0;
  const unsigned long long value = std::stoull(text, &position);
  if (position != text.size() ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

int parse_positive_int(const std::string &text, const char *name) {
  const std::size_t value = parse_size(text, name);
  if (value == 0 || value > static_cast<std::size_t>(INT_MAX)) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<int>(value);
}

std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

void usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " --inputs CPU_RESULTS --output GPU_RESULTS [options]\n"
         "Options:\n"
         "  --cases 0,1,2              default: all manifest cases\n"
         "  --routes materialized,fused default: both\n"
         "  --mappings thread,warp      default: warp\n"
         "  --tiles 32,64,128,256       default: 128 (fused only)\n"
         "  --modes resident,end_to_end default: both\n"
         "  --warmups 10                default: 10\n"
         "  --repeats 20                default: 20\n"
         "  --smoke                     first case, 1 warm-up, 3 repeats\n";
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("missing value after " + option);
      }
      return argv[index];
    };
    if (option == "--inputs") {
      arguments.inputs = value();
    } else if (option == "--output") {
      arguments.output = value();
    } else if (option == "--cases") {
      for (const auto &item : split(value(), ',')) {
        if (!item.empty() && item != "all") {
          arguments.cases.insert(item);
        }
      }
    } else if (option == "--routes") {
      arguments.routes.clear();
      for (auto item : split(lower(value()), ',')) {
        if (item == "materialized") {
          arguments.routes.push_back(Route::MaterializedFullSort);
        } else if (item == "fused") {
          arguments.routes.push_back(Route::FusedTiledBitonic);
        } else {
          throw std::invalid_argument("unknown route " + item);
        }
      }
    } else if (option == "--mappings") {
      arguments.mappings.clear();
      for (auto item : split(lower(value()), ',')) {
        if (item == "thread") {
          arguments.mappings.push_back(Mapping::ThreadPerVector);
        } else if (item == "warp") {
          arguments.mappings.push_back(Mapping::WarpPerVector);
        } else {
          throw std::invalid_argument("unknown mapping " + item);
        }
      }
    } else if (option == "--tiles") {
      arguments.tiles.clear();
      for (const auto &item : split(value(), ',')) {
        arguments.tiles.push_back(parse_size(item, "tile size"));
      }
    } else if (option == "--modes") {
      arguments.modes.clear();
      for (auto item : split(lower(value()), ',')) {
        if (item != "resident" && item != "end_to_end") {
          throw std::invalid_argument("unknown timing mode " + item);
        }
        arguments.modes.push_back(item);
      }
    } else if (option == "--warmups") {
      arguments.warmups = parse_positive_int(value(), "warmups");
    } else if (option == "--repeats") {
      arguments.repeats = parse_positive_int(value(), "repeats");
    } else if (option == "--smoke") {
      arguments.warmups = 1;
      arguments.repeats = 3;
      arguments.cases.insert("0");
    } else if (option == "--help" || option == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option " + option);
    }
  }
  if (arguments.inputs.empty() || arguments.output.empty()) {
    throw std::invalid_argument("--inputs and --output are required");
  }
  if (arguments.routes.empty() || arguments.mappings.empty() ||
      arguments.tiles.empty() || arguments.modes.empty()) {
    throw std::invalid_argument("route/mapping/tile/mode lists cannot be empty");
  }
  return arguments;
}

std::vector<ManifestRow> load_manifest(const fs::path &directory) {
  std::ifstream stream(directory / "inputs.csv");
  if (!stream) {
    throw std::runtime_error("cannot open " +
                             (directory / "inputs.csv").string());
  }
  std::string header;
  std::getline(stream, header);
  if (header != "case,N,d,q,k,prefix") {
    throw std::runtime_error("unexpected inputs.csv header: " + header);
  }
  std::vector<ManifestRow> rows;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = split(line, ',');
    if (fields.size() != 6) {
      throw std::runtime_error("malformed inputs.csv row: " + line);
    }
    rows.push_back({fields[0],
                    {parse_size(fields[1], "N"),
                     parse_size(fields[2], "d"),
                     parse_size(fields[3], "q"),
                     parse_size(fields[4], "k")},
                    fields[5]});
  }
  if (rows.empty()) {
    throw std::runtime_error("inputs.csv contains no cases");
  }
  return rows;
}

std::string csv_number(double value) {
  if (!std::isfinite(value)) {
    return "";
  }
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

std::string optional_bool(const std::optional<bool> &value) {
  if (!value.has_value()) {
    return "";
  }
  return *value ? "1" : "0";
}

bool contains_mode(const Arguments &arguments, const std::string &mode) {
  return std::find(arguments.modes.begin(), arguments.modes.end(), mode) !=
         arguments.modes.end();
}

void write_metadata(const fs::path &output, const Arguments &arguments) {
  int device = 0;
  cudaDeviceProp properties{};
  int runtime_version = 0;
  int driver_version = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
      cudaDriverGetVersion(&driver_version) != cudaSuccess) {
    throw std::runtime_error("failed to query CUDA environment");
  }
  std::ofstream stream(output / "run.txt");
  stream << "contract=1.1\n"
         << "seed=5206\n"
         << "generator=synthetic-v1 (inputs loaded from CPU export)\n"
         << "warmups=" << arguments.warmups << "\n"
         << "repeats=" << arguments.repeats << "\n"
         << "gpu_device=" << device << "\n"
         << "gpu_name=" << properties.name << "\n"
         << "compute_capability=" << properties.major << '.'
         << properties.minor << "\n"
         << "global_memory_bytes=" << properties.totalGlobalMem << "\n"
         << "multiprocessors=" << properties.multiProcessorCount << "\n"
         << "max_threads_per_block=" << properties.maxThreadsPerBlock << "\n"
         << "shared_memory_per_block=" << properties.sharedMemPerBlock << "\n"
         << "cuda_runtime_version=" << runtime_version << "\n"
         << "cuda_driver_version=" << driver_version << "\n"
         << "timing=gpu-resident and transfer-inclusive end-to-end\n"
         << "distance_accumulation=FP32\n"
         << "tie_order=ascending distance then ascending sample ID\n"
         << "fmad=false\n";
}

std::vector<gpu_knn::SearchConfig>
make_configs(const Arguments &arguments) {
  std::vector<gpu_knn::SearchConfig> configs;
  for (const Mapping mapping : arguments.mappings) {
    if (std::find(arguments.routes.begin(), arguments.routes.end(),
                  Route::MaterializedFullSort) != arguments.routes.end()) {
      configs.push_back({Route::MaterializedFullSort, mapping, 0});
    }
    if (std::find(arguments.routes.begin(), arguments.routes.end(),
                  Route::FusedTiledBitonic) != arguments.routes.end()) {
      for (const std::size_t tile : arguments.tiles) {
        configs.push_back({Route::FusedTiledBitonic, mapping, tile});
      }
    }
  }
  return configs;
}

void write_raw(std::ofstream &raw, const ManifestRow &row,
               const RunEntry &entry, const std::string &mode, int round,
               const gpu_knn::SearchResult &result) {
  const auto &times = result.times;
  const auto &memory = result.memory;
  const bool resident = mode == "resident";
  const double complete_ms = resident ? times.device_ms : times.end_to_end_ms;
  raw << row.case_id << ',' << row.shape.n << ',' << row.shape.d << ','
      << row.shape.q << ',' << row.shape.k << ','
      << gpu_knn::route_name(entry.config.route) << ','
      << gpu_knn::mapping_name(entry.config.mapping) << ',';
  if (entry.config.route == Route::FusedTiledBitonic) {
    raw << entry.config.tile_size;
  }
  raw << ',' << mode << ',' << round << ',' << std::setprecision(12)
      << complete_ms << ',';
  if (!resident) {
    raw << times.end_to_end_ms;
  }
  raw << ',' << times.device_ms << ',';
  if (!resident) {
    raw << times.h2d_ms;
  }
  raw << ',' << csv_number(times.distance_ms) << ','
      << csv_number(times.pack_ms) << ','
      << csv_number(times.local_selection_ms) << ','
      << csv_number(times.full_sort_ms) << ','
      << csv_number(times.merge_ms) << ',' << csv_number(times.gather_ms)
      << ',';
  if (!resident) {
    raw << times.d2h_ms;
  }
  raw << ',' << memory.auxiliary_bytes << ','
      << memory.peak_allocated_bytes << ',' << result.merge_levels << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    fs::create_directories(arguments.output);
    write_metadata(arguments.output, arguments);
    const auto manifest = load_manifest(arguments.inputs);
    const auto configs = make_configs(arguments);

    std::ofstream raw(arguments.output / "raw.csv");
    std::ofstream correctness(arguments.output / "correctness.csv");
    std::ofstream memory(arguments.output / "memory.csv");
    if (!raw || !correctness || !memory) {
      throw std::runtime_error("failed to create GPU result CSV files");
    }
    raw << "case,N,d,q,k,route,mapping,tile_size,mode,round,complete_ms,"
           "end_to_end_ms,device_ms,h2d_ms,distance_ms,pack_ms,"
           "local_selection_ms,full_sort_ms,merge_ms,gather_ms,d2h_ms,"
           "aux_bytes,peak_device_bytes,merge_levels\n";
    correctness
        << "case,route,mapping,tile_size,structure,labels,sorted,"
           "distance_tolerance,FP64_ID_agreement,ID_mismatches,"
           "max_abs_error,max_rel_error,deterministic,materialized_match,"
           "passed\n";
    memory << "case,route,mapping,tile_size,input_bytes,output_bytes,"
              "aux_bytes,peak_device_bytes,dynamic_shared_bytes,"
              "cub_temp_bytes,distance_buffer_bytes,key_buffer_bytes,"
              "candidate_buffer_bytes,merge_levels\n";

    std::size_t selected_cases = 0;
    for (const auto &row : manifest) {
      if (!arguments.cases.empty() &&
          arguments.cases.count(row.case_id) == 0) {
        continue;
      }
      ++selected_cases;
      std::cout << "GPU case " << row.case_id << " N=" << row.shape.n
                << " d=" << row.shape.d << " q=" << row.shape.q
                << " k=" << row.shape.k << std::endl;
      const fs::path prefix = arguments.inputs / row.prefix;
      const auto input = gpu_knn::load_input(prefix.string(), row.shape);
      const auto oracle = gpu_knn::make_fp64_oracle(input);

      std::vector<RunEntry> entries;
      entries.reserve(configs.size());
      for (const auto &config : configs) {
        std::cout << "  prepare " << gpu_knn::route_name(config.route)
                  << " / " << gpu_knn::mapping_name(config.mapping);
        if (config.route == Route::FusedTiledBitonic) {
          std::cout << " / T=" << config.tile_size;
        }
        std::cout << std::endl;
        RunEntry entry;
        entry.config = config;
        entry.runner = std::make_unique<gpu_knn::Runner>(input, config);
        entry.first = entry.runner->run(false, true);
        const auto second = entry.runner->run(false, true);
        entry.deterministic = gpu_knn::exact_same_result(entry.first, second);
        entry.verification =
            gpu_knn::verify_fp64(input, entry.first, oracle);
        entries.push_back(std::move(entry));
      }

      std::map<std::string, const gpu_knn::SearchResult *> materialized;
      for (const auto &entry : entries) {
        if (entry.config.route == Route::MaterializedFullSort) {
          materialized[gpu_knn::mapping_name(entry.config.mapping)] =
              &entry.first;
        }
      }
      for (auto &entry : entries) {
        if (entry.config.route == Route::MaterializedFullSort) {
          entry.materialized_match = true;
        } else {
          const auto found =
              materialized.find(gpu_knn::mapping_name(entry.config.mapping));
          if (found != materialized.end()) {
            entry.materialized_match =
                gpu_knn::exact_same_result(entry.first, *found->second);
          }
        }

        const auto &v = entry.verification;
        const bool pair_ok =
            !entry.materialized_match.has_value() || *entry.materialized_match;
        const bool passed = v.passed() && entry.deterministic && pair_ok;
        correctness << row.case_id << ','
                    << gpu_knn::route_name(entry.config.route) << ','
                    << gpu_knn::mapping_name(entry.config.mapping) << ',';
        if (entry.config.route == Route::FusedTiledBitonic) {
          correctness << entry.config.tile_size;
        }
        correctness << ',' << v.structure << ',' << v.labels << ','
                    << v.sorted << ',' << v.distance_tolerance << ','
                    << v.fp64_id_agreement << ',' << v.id_mismatches << ','
                    << std::setprecision(12) << v.max_abs_error << ','
                    << v.max_rel_error << ',' << entry.deterministic << ','
                    << optional_bool(entry.materialized_match) << ',' << passed
                    << '\n';

        const auto &m = entry.runner->memory();
        memory << row.case_id << ',' << gpu_knn::route_name(entry.config.route)
               << ',' << gpu_knn::mapping_name(entry.config.mapping) << ',';
        if (entry.config.route == Route::FusedTiledBitonic) {
          memory << entry.config.tile_size;
        }
        memory << ',' << m.input_bytes << ',' << m.output_bytes << ','
               << m.auxiliary_bytes << ',' << m.peak_allocated_bytes << ','
               << m.dynamic_shared_bytes << ',' << m.cub_temp_bytes << ','
               << m.distance_buffer_bytes << ',' << m.key_buffer_bytes << ','
               << m.candidate_buffer_bytes << ','
               << entry.runner->merge_levels() << '\n';

        if (!passed) {
          throw std::runtime_error(
              std::string("correctness failure in ") +
              gpu_knn::route_name(entry.config.route) + " / " +
              gpu_knn::mapping_name(entry.config.mapping));
        }
      }
      correctness.flush();
      memory.flush();

      for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (auto &entry : entries) {
          if (contains_mode(arguments, "resident")) {
            entry.runner->run(false, false);
          }
          if (contains_mode(arguments, "end_to_end")) {
            entry.runner->run(true, true);
          }
        }
      }

      for (int round = 0; round < arguments.repeats; ++round) {
        for (std::size_t offset = 0; offset < entries.size(); ++offset) {
          RunEntry &entry = entries[(static_cast<std::size_t>(round) + offset) %
                                    entries.size()];
          const bool reverse_modes = round % 2 != 0;
          for (int mode_index = 0; mode_index < 2; ++mode_index) {
            const bool resident =
                (mode_index == 0) != reverse_modes;
            const std::string mode = resident ? "resident" : "end_to_end";
            if (!contains_mode(arguments, mode)) {
              continue;
            }
            const auto result = entry.runner->run(!resident, !resident);
            write_raw(raw, row, entry, mode, round, result);
          }
        }
        raw.flush();
      }
    }

    if (selected_cases == 0) {
      throw std::runtime_error("case filter selected no manifest rows");
    }
    std::cout << "GPU benchmark complete: " << arguments.output << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
