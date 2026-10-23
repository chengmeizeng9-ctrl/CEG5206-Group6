#include "knn.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
using namespace knn;
using Clock = std::chrono::steady_clock;

static double quantile(std::vector<double> values, double probability) {
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * probability;
  const size_t low = size_t(position);
  const size_t high = std::min(low + 1, values.size() - 1);
  return values[low] + (values[high] - values[low]) * (position - low);
}

static void export_input(const Input &input, const std::string &prefix) {
  for (int array = 0; array < 3; ++array) {
    std::ofstream out(prefix + (array == 0   ? ".X.f32"
                                : array == 1 ? ".Q.f32"
                                             : ".labels.i32"),
                      std::ios::binary);
    if (array < 2) {
      const auto &values = array == 0 ? input.x : input.queries;
      out.write(reinterpret_cast<const char *>(values.data()),
                values.size() * sizeof(float));
    } else {
      out.write(reinterpret_cast<const char *>(input.labels.data()),
                input.labels.size() * sizeof(int32_t));
    }
    if (!out)
      throw std::runtime_error("input export failed");
  }
}

int main(int argc, char **argv) {
  try {
    const std::string directory = argc > 1 ? argv[1] : "results";
    const bool smoke = argc > 2 && std::string(argv[2]) == "--smoke";
    std::filesystem::create_directories(directory);
    uint32_t endian = 1;
    if (*reinterpret_cast<unsigned char *>(&endian) != 1)
      throw std::runtime_error("export requires little endian");
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed))
      throw std::runtime_error("get affinity failed");
    int cpu = CPU_ISSET(1, &allowed) ? 1 : -1;
    if (cpu < 0)
      for (int id = 0; id < CPU_SETSIZE; ++id)
        if (CPU_ISSET(id, &allowed)) {
          cpu = id;
          break;
        }
    if (cpu < 0)
      throw std::runtime_error("no CPU available");
    cpu_set_t chosen;
    CPU_ZERO(&chosen);
    CPU_SET(cpu, &chosen);
    if (sched_setaffinity(0, sizeof chosen, &chosen))
      throw std::runtime_error("set affinity failed");

    std::ofstream metadata(directory + "/run.txt");
    metadata << "cpu=" << cpu
             << "\nthreads=1\ncontract=1.1\nexecution_rules=1.2\nseed="
                "5206\ngenerator=synthetic-v1\nwarmups=10\nrepeats=20\n"
                "timing=prevalidated steady-state complete CPU search\n"
                "routes=scalar_full_sort,avx2_topk_heap\nFTZ_DAZ=off\n"
                "rounding=nearest\nFMA_contraction=off\n";

    std::ofstream raw(directory + "/raw.csv"),
        summary(directory + "/summary.csv"),
        correctness(directory + "/correctness.csv"),
        diagnostics(directory + "/diagnostics.csv"),
        inputs(directory + "/inputs.csv");
    raw << "case,N,d,q,k,route,group,round,seconds\n";
    summary << "case,N,d,q,k,route,group,median,Q1,Q3,IQR,queries_per_second,"
               "aux_bytes,input_bytes,output_bytes,prepare_seconds,unstable\n";
    correctness << "case,route,structure,labels,sorted,fixed_distance_selection,"
                    "distance_tolerance,FP64_ID_agreement,ID_mismatches,"
                    "max_abs_error,deterministic\n";
    diagnostics << "case,route,candidates,dimensions\n";
    inputs << "case,N,d,q,k,prefix\n";

    struct Shape { size_t n, d, q, k; };
    std::vector<Shape> shapes = {
        {100000, 64, 16, 10}, {10000, 64, 16, 10},
        {100000, 32, 16, 10}, {100000, 128, 16, 10},
        {100000, 64, 1, 10},  {100000, 64, 64, 10},
        {100000, 64, 16, 5},  {100000, 64, 16, 32}};
    if (smoke)
      shapes = {{257, 9, 3, 10}};
    const Route routes[] = {Route::ScalarFullSort, Route::Avx2TopKHeap};

    for (size_t case_id = 0; case_id < shapes.size(); ++case_id) {
      const auto shape = shapes[case_id];
      std::cout << "case " << case_id << " N=" << shape.n
                << " d=" << shape.d << " q=" << shape.q << std::endl;
      Input input = synthetic(shape.n, shape.d, shape.q);
      auto validated = validate(input);
      const std::string prefix = "input_" + std::to_string(case_id);
      export_input(input, directory + "/" + prefix);
      inputs << case_id << ',' << shape.n << ',' << shape.d << ',' << shape.q
             << ',' << shape.k << ',' << prefix << '\n';
      const auto reference = oracle(input);
      Workspace workspaces[2];
      Result results[2];
      double preparation[2] = {};
      for (int index = 0; index < 2; ++index) {
        const auto start = Clock::now();
        prepare(validated, {shape.k, routes[index]}, workspaces[index],
                results[index]);
        preparation[index] =
            std::chrono::duration<double>(Clock::now() - start).count();
        search(validated, {shape.k, routes[index]}, workspaces[index],
               results[index]);
        const auto saved = results[index];
        const auto verification =
            verify(input, {shape.k, routes[index]}, results[index], reference);
        search(validated, {shape.k, routes[index]}, workspaces[index],
               results[index]);
        const bool deterministic = results[index].ids == saved.ids &&
                                   results[index].distances == saved.distances &&
                                   results[index].labels == saved.labels;
        correctness << case_id << ',' << route_name(routes[index]) << ','
                    << verification.structure << ',' << verification.labels
                    << ',' << verification.sorted << ','
                    << verification.selection << ','
                    << verification.distance_tolerance << ','
                    << verification.ids << ',' << verification.id_mismatches
                    << ',' << std::setprecision(12)
                    << verification.max_abs_error << ',' << deterministic
                    << '\n';
        if (!verification.passed() || !deterministic)
          throw std::runtime_error("benchmark correctness failure");
      }

      for (int group = 0; group < 2; ++group) {
        std::vector<double> times[2];
        for (int warmup = 0; warmup < 10; ++warmup)
          for (int index = 0; index < 2; ++index)
            search(validated, {shape.k, routes[index]}, workspaces[index],
                   results[index]);
        for (int round = 0; round < 20; ++round)
          for (int order = 0; order < 2; ++order) {
            const int index = round % 2 ? 1 - order : order;
            const auto start = Clock::now();
            search(validated, {shape.k, routes[index]}, workspaces[index],
                   results[index]);
            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            times[index].push_back(seconds);
            raw << case_id << ',' << shape.n << ',' << shape.d << ','
                << shape.q << ',' << shape.k << ',' << route_name(routes[index])
                << ',' << group << ',' << round << ',' << std::setprecision(12)
                << seconds << '\n';
          }
        bool unstable = false;
        for (int index = 0; index < 2; ++index) {
          const double median = quantile(times[index], .5);
          const double q1 = quantile(times[index], .25);
          const double q3 = quantile(times[index], .75);
          const bool noisy = (q3 - q1) / median > .1;
          unstable |= noisy;
          const size_t input_bytes =
              (input.x.capacity() + input.queries.capacity()) * sizeof(float) +
              input.labels.capacity() * sizeof(int32_t);
          const size_t output_bytes =
              results[index].ids.capacity() * sizeof(uint32_t) +
              results[index].distances.capacity() * sizeof(float) +
              results[index].labels.capacity() * sizeof(int32_t);
          summary << case_id << ',' << shape.n << ',' << shape.d << ','
                  << shape.q << ',' << shape.k << ',' << route_name(routes[index])
                  << ',' << group << ',' << std::setprecision(12) << median
                  << ',' << q1 << ',' << q3 << ',' << q3 - q1 << ','
                  << shape.q / median << ',' << workspaces[index].bytes() << ','
                  << input_bytes << ',' << output_bytes << ','
                  << preparation[index] << ',' << noisy << '\n';
        }
        raw.flush();
        summary.flush();
        if (!unstable)
          break;
      }
      for (int index = 0; index < 2; ++index) {
        Diagnostics measured;
        search(validated, {shape.k, routes[index]}, workspaces[index],
               results[index], &measured);
        diagnostics << case_id << ',' << route_name(routes[index]) << ','
                    << measured.candidates << ',' << measured.dimensions << '\n';
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
