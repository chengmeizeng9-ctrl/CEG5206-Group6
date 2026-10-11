#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
using namespace knn;
static size_t checks = 0, cases = 0, mismatches = 0;
static void require(bool v, const char *s) {
  ++checks;
  if (!v)
    throw std::runtime_error(s);
}
static void expect(ErrorCode code, std::function<void()> f) {
  try {
    f();
    throw std::runtime_error("expected error missing");
  } catch (const Error &e) {
    require(e.code == code, "wrong error category");
  }
}
static void exercise(const Input &i, size_t k, size_t t) {
  ++cases;
  auto o = oracle(i);
  auto baseline = search(i, {k, t, 0});
  for (int route = 0; route < 8; ++route) {
    auto val = validate(i);
    Workspace w;
    Result r;
    prepare(val, {k, t, route}, w, r);
    search(val, {k, t, route}, w, r);
    auto v = verify(i, {k, t, route}, r, o);
    require(v.passed(), "search structure/selection/numeric verification");
    mismatches += v.id_mismatches;
    auto saved = r;
    search(val, {k, t, route}, w, r);
    require(r.ids == saved.ids && r.distances == saved.distances &&
                r.labels == saved.labels,
            "repeat determinism");
    if (route != 2 && route != 4 && route != 6)
      require(r.ids == baseline.ids && r.distances == baseline.distances,
              "scalar route agreement");
    Diagnostics diag;
    search(val, {k, t, route}, w, r, &diag);
    require(diag.candidates == i.n * i.q, "diagnostic count");
  }
}
int main(int argc, char **argv) {
  try {
    floating_environment();
    exercise(synthetic(1, 1, 1), 1, 32);
    for (size_t t : {32, 64, 128, 256})
      for (size_t n : {t - 1, t, t + 1, 2 * t + 3})
        for (size_t d : {1, 7, 8, 9, 19}) {
          auto i = synthetic(n, d, 3);
          for (size_t k : {size_t(1), std::min(n, size_t(10)), n})
            exercise(i, k, t);
          if (n > t)
            exercise(i, t + 1, t);
        }
    Input zero{40,
               9,
               3,
               std::vector<float>(360, 0),
               std::vector<float>(27, 0),
               std::vector<int32_t>(40)};
    for (size_t n = 0; n < 40; ++n)
      zero.labels[n] = 100 - n;
    exercise(zero, 35, 32);
    auto zr = search(zero, {35, 32, 7});
    for (size_t rank = 0; rank < 35; ++rank)
      require(zr.ids[rank] == rank, "zero tie ID order");
    auto equal = zero;
    std::fill(equal.x.begin(), equal.x.end(), 0.5f);
    std::fill(equal.queries.begin(), equal.queries.end(), 0.5f);
    exercise(equal, 10, 32);
    Input ties{5,
               2,
               2,
               {1, 0, -1, 0, 0, 1, 0, -1, 0, 0},
               {0, 0, 1, 0},
               {11, 22, 33, 44, 55}};
    exercise(ties, 3, 32);
    auto tr = search(ties, {3, 32, 0});
    require(tr.ids[0] == 4 && tr.ids[1] == 0 && tr.ids[2] == 1,
            "exact symmetric ties");
    auto near = synthetic(8, 9, 2);
    std::fill(near.x.begin(), near.x.end(), 0.5f);
    std::fill(near.queries.begin(), near.queries.end(), 0);
    near.x[0] = std::nextafter(0.5f, 1.0f);
    exercise(near, 4, 32);
    float a[] = {0, 0}, b[] = {1, 0};
    float out = -1;
    size_t used = 0;
    require(!abandoned_distance(a, b, 2, 1, out, used) && used == 2 && out == 1,
            "tau equality continues");
    require(abandoned_distance(a, b, 2, 0.5f, out, used) && used == 1,
            "strict greater prunes");
    auto early = search(ties, {5, 32, 7});
    require(early.ids == search(ties, {5, 32, 3}).ids, "heap not full/k=N");
    for (size_t t : {32, 64, 128, 256}) {
      std::vector<Candidate> fixed;
      for (uint32_t id = 0; id < 2 * t + 3; ++id)
        fixed.push_back({float(id % 7), id});
      std::reverse(fixed.begin(), fixed.end());
      for (size_t k : {size_t(1), size_t(10), t + 1, fixed.size()}) {
        auto ref = select_fixed(fixed, k, 0, t);
        for (int route : {1, 2}) {
          auto r = select_fixed(fixed, k, route, t);
          for (size_t j = 0; j < k; ++j)
            require(r[j].id == ref[j].id && r[j].distance == ref[j].distance,
                    "fixed distance selection");
        }
      }
    }
    auto sub = synthetic(5, 9, 2);
    std::fill(sub.x.begin(), sub.x.end(),
              std::numeric_limits<float>::denorm_min());
    std::fill(sub.queries.begin(), sub.queries.end(), 0);
    exercise(sub, 5, 32);
    auto valid = synthetic(3, 2, 1);
    for (size_t k : {size_t(0), size_t(4), SIZE_MAX})
      expect(ErrorCode::INVALID_ARGUMENT, [&] { search(valid, {k, 32, 0}); });
    expect(ErrorCode::INVALID_ARGUMENT, [&] { search(valid, {1, 31, 1}); });
    auto bad = valid;
    bad.x.pop_back();
    expect(ErrorCode::INVALID_SHAPE, [&] { validate(bad); });
    for (float v : {INFINITY, -INFINITY, NAN, 1.01f, -1.01f}) {
      bad = valid;
      bad.x[0] = v;
      expect(ErrorCode::INVALID_VALUE, [&] { validate(bad); });
    }
    bad = valid;
    bad.q = 0;
    expect(ErrorCode::INVALID_ARGUMENT, [&] { validate(bad); });
    expect(ErrorCode::SIZE_OVERFLOW, [] { checked_product(SIZE_MAX, 2); });
    bad = valid;
    bad.q = SIZE_MAX;
    expect(ErrorCode::SIZE_OVERFLOW, [&] { validate(bad); });
    auto token = validate(valid);
    Workspace w;
    Result r;
    expect(ErrorCode::INVALID_SHAPE, [&] { search(token, {1, 32, 0}, w, r); });
    prepare(token, {2, 32, 3}, w, r);
    search(token, {2, 32, 3}, w, r);
    prepare(token, {1, 64, 7}, w, r);
    search(token, {1, 64, 7}, w, r);
    require(r.ids == search(valid, {1, 64, 0}).ids,
            "alternating configurations");
    if (argc > 1) {
      std::string dir = argv[1];
      for (int example = 0; example < 2; ++example) {
        const auto &i = example == 0 ? ties : near;
        auto o = oracle(i);
        std::string prefix = dir + "/fixture_" + std::to_string(example);
        std::ofstream meta(prefix + ".txt");
        meta << i.n << ' ' << i.d << ' ' << i.q << '\n';
        for (int array = 0; array < 3; ++array) {
          std::ofstream out(prefix + (array == 0   ? ".X.f32"
                                      : array == 1 ? ".Q.f32"
                                                   : ".labels.i32"),
                            std::ios::binary);
          if (array < 2) {
            const auto &v = array == 0 ? i.x : i.queries;
            out.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
          } else
            out.write(reinterpret_cast<const char *>(i.labels.data()),
                      i.labels.size() * 4);
        }
        std::ofstream refs(prefix + ".oracle.csv");
        refs.precision(17);
        refs << "query,rank,id,distance,label\n";
        for (size_t q = 0; q < i.q; ++q)
          for (size_t rank = 0; rank < i.n; ++rank) {
            auto id = o.order[q * i.n + rank];
            refs << q << ',' << rank << ',' << id << ','
                 << o.distances[q * i.n + id] << ',' << i.labels[id] << '\n';
          }
      }
    }
    auto corrupted = search(valid, {2, 32, 0});
    corrupted.labels[0] += 1;
    require(!verify(valid, {2, 32, 0}, corrupted, oracle(valid)).labels,
            "validator label mutation");
    corrupted.ids[1] = corrupted.ids[0];
    require(!verify(valid, {2, 32, 0}, corrupted, oracle(valid)).structure,
            "validator duplicate mutation");
    std::cout << "PASS cases=" << cases << " checks=" << checks
              << " FP64_ID_mismatches=" << mismatches << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << "\n";
    return 1;
  }
}
