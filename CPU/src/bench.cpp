#include "knn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>
using namespace knn;
using Clock = std::chrono::steady_clock;
static std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}
static double quantile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  double pos = (v.size() - 1) * p;
  size_t lo = size_t(pos), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}
static void snapshot(const std::string &path) {
  std::ofstream out(path);
  for (auto file : {"/proc/meminfo", "/proc/stat", "/proc/self/status"}) {
    out << file << "\n";
    std::ifstream in(file);
    out << in.rdbuf() << "\n";
  }
  out << "process_cpu_seconds " << double(std::clock()) / CLOCKS_PER_SEC
      << "\n";
}
static void export_input(const Input &i, const std::string &prefix) {
  for (int a = 0; a < 3; ++a) {
    std::ofstream out(prefix + (a == 0   ? ".X.f32"
                                : a == 1 ? ".Q.f32"
                                         : ".labels.i32"),
                      std::ios::binary);
    if (a < 2) {
      auto &v = a == 0 ? i.x : i.queries;
      out.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
    } else
      out.write(reinterpret_cast<const char *>(i.labels.data()),
                i.labels.size() * 4);
    if (!out)
      throw std::runtime_error("input export failed");
  }
}
// Fixed-distance selection diagnostic: all buffers prepared before measurement.
static void fixed_select(const std::vector<Candidate> &source, size_t k,
                         size_t t, int route, Workspace &w) {
  Better better;
  w.heap.clear();
  if (route == 0) {
    std::copy(source.begin(), source.end(), w.all.begin());
    std::sort(w.all.begin(), w.all.end(), better);
  } else if (route == 1) {
    for (auto candidate : source) {
      if (w.heap.size() < k) {
        w.heap.push_back(candidate);
        std::push_heap(w.heap.begin(), w.heap.end(), better);
      } else if (better(candidate, w.heap.front())) {
        std::pop_heap(w.heap.begin(), w.heap.end(), better);
        w.heap.back() = candidate;
        std::push_heap(w.heap.begin(), w.heap.end(), better);
      }
    }
    std::sort(w.heap.begin(), w.heap.end(), better);
  } else {
    for (size_t base = 0; base < source.size(); base += t) {
      size_t count = std::min(t, source.size() - base);
      std::copy_n(source.begin() + base, count, w.tile.begin());
      std::fill(w.tile.begin() + count, w.tile.end(),
                Candidate{INFINITY, invalid_id});
      bitonic(w.tile.data(), t);
      size_t kept = merge_topk(w.heap.data(), w.heap.size(), w.tile.data(),
                               std::min(k, count), w.merged.data(), k);
      w.heap.assign(w.merged.begin(), w.merged.begin() + kept);
    }
  }
}
int main(int argc, char **argv) {
  try {
    std::string dir = argc > 1 ? argv[1] : "results";
    bool smoke = argc > 2 && std::string(argv[2]) == "--smoke";
    std::filesystem::create_directories(dir);
    uint32_t endian = 1;
    if (*reinterpret_cast<unsigned char *>(&endian) != 1)
      throw std::runtime_error("export requires little endian");
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed))
      throw std::runtime_error("get affinity failed");
    int cpu = CPU_ISSET(1, &allowed) ? 1 : -1;
    if (cpu < 0)
      for (int j = 0; j < CPU_SETSIZE; ++j)
        if (CPU_ISSET(j, &allowed)) {
          cpu = j;
          break;
        }
    cpu_set_t chosen;
    CPU_ZERO(&chosen);
    if (cpu < 0)
      throw std::runtime_error("no CPU available");
    CPU_SET(cpu, &chosen);
    if (sched_setaffinity(0, sizeof chosen, &chosen))
      throw std::runtime_error("set affinity failed");
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) ||
        !CPU_ISSET(cpu, &allowed) || CPU_COUNT(&allowed) != 1)
      throw std::runtime_error("affinity verification");
    std::ofstream metadata(dir + "/run.txt");
    metadata << "cpu=" << cpu
             << "\nthreads=1\ncontract=1.1\nexecution_rules=1.1\nseed="
                "5206\ngenerator=synthetic-v1\nwarmup=10\nrepeats=20\nwindow_"
                "calls=1\ntiming=prevalidated steady-state complete CPU "
                "search\nFTZ_DAZ=off\nrounding=nearest\nFMA_contraction=off\n";
    snapshot(dir + "/before.txt");
    std::ofstream micro(dir + "/selection_raw.csv");
    micro << "case,N,k,T,selection,round,seconds\n";
    std::ofstream raw(dir + "/raw.csv"), summary(dir + "/summary.csv"),
        correct(dir + "/correctness.csv"), diagnostic(dir + "/diagnostics.csv"),
        manifest(dir + "/inputs.csv");
    raw << "case,N,d,q,k,T,route,group,round,seconds\n";
    summary << "case,N,d,q,k,T,route,group,median,Q1,Q3,IQR,queries_per_second,"
               "aux_bytes,input_bytes,output_bytes,prepare_seconds,unstable\n";
    correct << "case,route,structure,labels,sorted,fixed_distance_selection,"
               "distance_tolerance,FP64_ID_agreement,ID_mismatches,max_abs_"
               "error,deterministic\n";
    diagnostic << "case,route,candidates,dimensions,abandoned,mean_dimensions,"
                  "abandoned_fraction,distance_seconds,selection_seconds,"
                  "output_seconds,combined_seconds\n";
    manifest << "case,N,d,q,k,T,prefix\n";
    struct Shape {
      size_t n, d, q, k, t;
    };
    std::vector<Shape> shapes = {
        {100000, 64, 16, 10, 128}, {10000, 64, 16, 10, 128},
        {100000, 32, 16, 10, 128}, {100000, 128, 16, 10, 128},
        {100000, 64, 1, 10, 128},  {100000, 64, 64, 10, 128},
        {100000, 64, 16, 5, 128},  {100000, 64, 16, 32, 128},
        {100000, 64, 16, 10, 32},  {100000, 64, 16, 10, 64},
        {100000, 64, 16, 10, 256}};
    // The contract says 10, but the explicit base + single-factor values yield
    // 11 (1+1+2+2+2+3).
    if (smoke)
      shapes = {{257, 9, 3, 10, 32}};
    for (size_t ci = 0; ci < shapes.size(); ++ci) {
      auto s = shapes[ci];
      std::cout << "case " << ci << " N=" << s.n << " d=" << s.d << " q=" << s.q
                << std::endl;
      Input i = synthetic(s.n, s.d, s.q);
      auto token = validate(i);
      std::string prefix = "input_" + std::to_string(ci);
      export_input(i, dir + "/" + prefix);
      manifest << ci << ',' << s.n << ',' << s.d << ',' << s.q << ',' << s.k
               << ',' << s.t << ',' << prefix << '\n';
      auto o = oracle(i);
      Workspace work[8];
      Result results[8];
      double preparation[8];
      bool deterministic[8];
      for (int r = 0; r < 8; ++r) {
        auto start = Clock::now();
        prepare(token, {s.k, s.t, r}, work[r], results[r]);
        preparation[r] =
            std::chrono::duration<double>(Clock::now() - start).count();
        search(token, {s.k, s.t, r}, work[r], results[r]);
        auto saved = results[r];
        auto v = verify(i, {s.k, s.t, r}, results[r], o);
        search(token, {s.k, s.t, r}, work[r], results[r]);
        deterministic[r] = results[r].ids == saved.ids &&
                           results[r].distances == saved.distances &&
                           results[r].labels == saved.labels;
        correct << ci << ',' << route_name(r) << ',' << v.structure << ','
                << v.labels << ',' << v.sorted << ',' << v.selection << ','
                << v.distance_tolerance << ',' << v.ids << ','
                << v.id_mismatches << ',' << std::setprecision(12)
                << v.max_abs_error << ',' << deterministic[r] << '\n';
        if (!v.passed() || !deterministic[r])
          throw std::runtime_error("benchmark correctness failure");
      }
      int first_route = (!smoke && ci >= 8) ? 1 : 0;
      for (int group = 0; group < 2; ++group) {
        std::vector<double> times[8];
        for (int warm = 0; warm < 10; ++warm)
          for (int r = first_route; r < 8; ++r)
            search(token, {s.k, s.t, r}, work[r], results[r]);
        for (int round = 0; round < 20; ++round)
          for (int index = 0; index < 8 - first_route; ++index) {
            int r = round % 2 ? 7 - index : first_route + index;
            auto start = Clock::now();
            search(token, {s.k, s.t, r}, work[r], results[r]);
            double sec =
                std::chrono::duration<double>(Clock::now() - start).count();
            times[r].push_back(sec);
            raw << ci << ',' << s.n << ',' << s.d << ',' << s.q << ',' << s.k
                << ',' << (r == 0 ? "NA" : std::to_string(s.t)) << ','
                << route_name(r) << ',' << group << ',' << round << ','
                << std::setprecision(12) << sec << '\n';
          }
        bool unstable = false;
        for (int r = first_route; r < 8; ++r) {
          double med = quantile(times[r], .5), q1 = quantile(times[r], .25),
                 q3 = quantile(times[r], .75);
          bool noisy = (q3 - q1) / med > .1;
          unstable |= noisy;
          summary << ci << ',' << s.n << ',' << s.d << ',' << s.q << ',' << s.k
                  << ',' << (r == 0 ? "NA" : std::to_string(s.t)) << ','
                  << route_name(r) << ',' << group << ','
                  << std::setprecision(12) << med << ',' << q1 << ',' << q3
                  << ',' << q3 - q1 << ',' << s.q / med << ','
                  << work[r].bytes() << ','
                  << (i.x.capacity() + i.queries.capacity()) * 4 +
                         i.labels.capacity() * 4
                  << ','
                  << (results[r].ids.capacity() +
                      results[r].distances.capacity() +
                      results[r].labels.capacity()) *
                         4
                  << ',' << preparation[r] << ',' << noisy << '\n';
        }
        raw.flush();
        summary.flush();
        if (!unstable)
          break;
      }
      for (int r = 0; r < 8; ++r) {
        Diagnostics dg;
        search(token, {s.k, s.t, r}, work[r], results[r], &dg);
        double distance = 0, selection = 0, output = 0, combined = 0;
        if (r <= 2) {
          for (size_t qi = 0; qi < s.q; ++qi) {
            auto start = Clock::now();
            for (size_t id = 0; id < s.n; ++id)
              work[r].all[id] = {
                  r == 2 ? simd_distance(i.queries.data() + qi * s.d,
                                         i.x.data() + id * s.d, s.d)
                         : scalar_distance(i.queries.data() + qi * s.d,
                                           i.x.data() + id * s.d, s.d),
                  uint32_t(id)};
            distance +=
                std::chrono::duration<double>(Clock::now() - start).count();
            start = Clock::now();
            std::sort(work[r].all.begin(), work[r].all.end(), Better{});
            selection +=
                std::chrono::duration<double>(Clock::now() - start).count();
            start = Clock::now();
            for (size_t j = 0; j < s.k; ++j) {
              auto c = work[r].all[j];
              size_t idx = qi * s.k + j;
              results[r].ids[idx] = c.id;
              results[r].distances[idx] = c.distance;
              results[r].labels[idx] = i.labels[c.id];
            }
            output +=
                std::chrono::duration<double>(Clock::now() - start).count();
          }
        } else {
          auto start = Clock::now();
          search(token, {s.k, s.t, r}, work[r], results[r]);
          combined =
              std::chrono::duration<double>(Clock::now() - start).count();
        }
        diagnostic << ci << ',' << route_name(r) << ',' << dg.candidates << ','
                   << dg.dimensions << ',' << dg.abandoned << ','
                   << double(dg.dimensions) / dg.candidates << ','
                   << double(dg.abandoned) / dg.candidates << ','
                   << (r <= 2 ? number(distance) : "NA") << ','
                   << (r <= 2 ? number(selection) : "NA") << ','
                   << (r <= 2 ? number(output) : "NA") << ','
                   << (r > 2 ? number(combined) : "NA") << '\n';
      }
      std::vector<Candidate> fixed(s.n);
      for (size_t id = 0; id < s.n; ++id)
        fixed[id] = {
            scalar_distance(i.queries.data(), i.x.data() + id * s.d, s.d),
            uint32_t(id)};
      auto reference = select_fixed(fixed, s.k, 0);
      Workspace mw[3];
      for (auto &w : mw) {
        w.all.resize(s.n);
        w.heap.reserve(s.k);
        w.tile.resize(s.t);
        w.merged.resize(s.k);
      }
      for (int warm = 0; warm < 10; ++warm)
        for (int route = 0; route < 3; ++route)
          fixed_select(fixed, s.k, s.t, route, mw[route]);
      for (int round = 0; round < 20; ++round)
        for (int index = 0; index < 3; ++index) {
          int route = round % 2 ? 2 - index : index;
          auto start = Clock::now();
          fixed_select(fixed, s.k, s.t, route, mw[route]);
          double sec =
              std::chrono::duration<double>(Clock::now() - start).count();
          const auto &selected = route == 0 ? mw[route].all : mw[route].heap;
          for (size_t j = 0; j < s.k; ++j)
            if (selected[j].id != reference[j].id ||
                selected[j].distance != reference[j].distance)
              throw std::runtime_error("fixed selection verification");
          micro << ci << ',' << s.n << ',' << s.k << ',' << s.t << ',' << route
                << ',' << round << ',' << std::setprecision(12) << sec << '\n';
        }
    }
    snapshot(dir + "/after.txt");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
