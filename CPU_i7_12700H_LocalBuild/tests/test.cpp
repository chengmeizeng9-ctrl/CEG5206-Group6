#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
using namespace knn;
static size_t checks = 0, cases = 0, mismatches = 0;
static void require(bool value, const char *message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}
static void expect(ErrorCode code, std::function<void()> action) {
  try {
    action();
    throw std::runtime_error("expected error missing");
  } catch (const Error &error) {
    require(error.code == code, "wrong error category");
  }
}
static void exercise(const Input &input, size_t k) {
  ++cases;
  const auto reference = oracle(input);
  for (Route route : {Route::ScalarFullSort, Route::Avx2TopKHeap}) {
    auto validated = validate(input);
    Workspace workspace;
    Result result;
    Config config{k, route};
    prepare(validated, config, workspace, result);
    search(validated, config, workspace, result);
    auto verification = verify(input, config, result, reference);
    require(verification.passed(), "search structure/selection/numeric check");
    mismatches += verification.id_mismatches;
    const auto saved = result;
    search(validated, config, workspace, result);
    require(result.ids == saved.ids && result.distances == saved.distances &&
                result.labels == saved.labels,
            "repeat determinism");
    Diagnostics diagnostics;
    search(validated, config, workspace, result, &diagnostics);
    require(diagnostics.candidates == input.n * input.q,
            "diagnostic candidate count");
    require(diagnostics.dimensions == input.n * input.q * input.d,
            "diagnostic dimension count");
  }
}
static void test_candidate_merge() {
  const std::vector<Candidate> left = {{1.0f, 4}, {2.0f, 1}, {2.0f, 8}};
  const std::vector<Candidate> right = {{0.5f, 7}, {2.0f, 3}, {4.0f, 0}};
  std::vector<Candidate> combined = left;
  combined.insert(combined.end(), right.begin(), right.end());
  std::sort(combined.begin(), combined.end(), Better{});
  std::vector<Candidate> output(combined.size());
  size_t count = merge_topk(left.data(), left.size(), right.data(), right.size(),
                            output.data(), output.size());
  require(count == combined.size(), "merge returns all candidates");
  for (size_t i = 0; i < count; ++i)
    require(output[i].id == combined[i].id &&
                output[i].distance == combined[i].distance,
            "merge matches full sorted reference");
  count = merge_topk(left.data(), left.size(), right.data(), right.size(),
                     output.data(), 4);
  require(count == 4, "merge respects Top-K limit");
  for (size_t i = 0; i < count; ++i)
    require(output[i].id == combined[i].id, "truncated merge prefix");
  const std::vector<Candidate> short_list = {{1.5f, 6}, {3.0f, 2}};
  combined = short_list;
  combined.insert(combined.end(), right.begin(), right.end());
  std::sort(combined.begin(), combined.end(), Better{});
  count = merge_topk(short_list.data(), short_list.size(), right.data(),
                     right.size(), output.data(), output.size());
  require(count == combined.size(), "merge handles unequal non-empty lists");
  for (size_t i = 0; i < count; ++i)
    require(output[i].id == combined[i].id, "unequal-list merge order");
  count = merge_topk(nullptr, 0, right.data(), right.size(), output.data(), 2);
  require(count == 2 && output[0].id == right[0].id &&
              output[1].id == right[1].id,
          "merge handles an empty input");
  count = merge_topk(nullptr, 0, nullptr, 0, nullptr, 0);
  require(count == 0, "empty merge");
  expect(ErrorCode::INVALID_ARGUMENT, [&] {
    merge_topk(nullptr, 1, right.data(), right.size(), output.data(), 2);
  });
}
int main(int argc, char **argv) {
  try {
    floating_environment();
    test_candidate_merge();
    exercise(synthetic(1, 1, 1), 1);
    for (size_t n : {size_t(1), size_t(7), size_t(8), size_t(9),
                     size_t(33), size_t(67), size_t(259)})
      for (size_t d : {size_t(1), size_t(7), size_t(8), size_t(9),
                       size_t(19)}) {
        auto input = synthetic(n, d, 3);
        for (size_t k : {size_t(1), std::min(n, size_t(10)), n})
          exercise(input, k);
      }
    Input zero{40, 9, 3, std::vector<float>(360, 0),
               std::vector<float>(27, 0), std::vector<int32_t>(40)};
    for (size_t id = 0; id < zero.n; ++id)
      zero.labels[id] = static_cast<int32_t>(100 - id);
    exercise(zero, 35);
    auto zero_result = search(zero, {35, Route::Avx2TopKHeap});
    for (size_t rank = 0; rank < 35; ++rank)
      require(zero_result.ids[rank] == rank, "zero-distance tie order");

    auto equal = zero;
    std::fill(equal.x.begin(), equal.x.end(), 0.5f);
    std::fill(equal.queries.begin(), equal.queries.end(), 0.5f);
    exercise(equal, 10);

    Input ties{5, 2, 2, {1, 0, -1, 0, 0, 1, 0, -1, 0, 0},
               {0, 0, 1, 0}, {11, 22, 33, 44, 55}};
    exercise(ties, 3);
    auto tie_result = search(ties, {3, Route::ScalarFullSort});
    require(tie_result.ids[0] == 4 && tie_result.ids[1] == 0 &&
                tie_result.ids[2] == 1,
            "symmetric exact ties");

    auto near = synthetic(8, 9, 2);
    std::fill(near.x.begin(), near.x.end(), 0.5f);
    std::fill(near.queries.begin(), near.queries.end(), 0);
    near.x[0] = std::nextafter(0.5f, 1.0f);
    exercise(near, 4);

    std::vector<Candidate> fixed;
    for (uint32_t id = 0; id < 103; ++id)
      fixed.push_back({float(id % 7), id});
    std::reverse(fixed.begin(), fixed.end());
    auto sorted = fixed;
    std::sort(sorted.begin(), sorted.end(), Better{});
    for (size_t k : {size_t(1), size_t(10), size_t(64), fixed.size()}) {
      auto heap = select_topk_heap(fixed, k);
      require(heap.size() == k, "heap selector result size");
      for (size_t rank = 0; rank < k; ++rank)
        require(heap[rank].id == sorted[rank].id &&
                    heap[rank].distance == sorted[rank].distance,
                "heap selection matches full sort for fixed distances");
    }

    auto subnormal = synthetic(5, 9, 2);
    std::fill(subnormal.x.begin(), subnormal.x.end(),
              std::numeric_limits<float>::denorm_min());
    std::fill(subnormal.queries.begin(), subnormal.queries.end(), 0);
    exercise(subnormal, 5);

    auto valid = synthetic(3, 2, 1);
    for (size_t k : {size_t(0), size_t(4), SIZE_MAX})
      expect(ErrorCode::INVALID_ARGUMENT, [&] {
        search(valid, {k, Route::ScalarFullSort});
      });
    expect(ErrorCode::INVALID_ARGUMENT, [&] {
      search(valid, {1, static_cast<Route>(9)});
    });
    expect(ErrorCode::INVALID_ARGUMENT, [&] { select_topk_heap(fixed, 104); });
    auto bad = valid;
    bad.x.pop_back();
    expect(ErrorCode::INVALID_SHAPE, [&] { validate(bad); });
    for (float value : {INFINITY, -INFINITY, NAN, 1.01f, -1.01f}) {
      bad = valid;
      bad.x[0] = value;
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
    Workspace workspace;
    Result result;
    expect(ErrorCode::INVALID_SHAPE, [&] {
      search(token, {1, Route::ScalarFullSort}, workspace, result);
    });
    prepare(token, {2, Route::Avx2TopKHeap}, workspace, result);
    search(token, {2, Route::Avx2TopKHeap}, workspace, result);
    prepare(token, {1, Route::ScalarFullSort}, workspace, result);
    search(token, {1, Route::ScalarFullSort}, workspace, result);
    require(result.ids == search(valid, {1, Route::ScalarFullSort}).ids,
            "alternating route configuration");

    if (argc > 1) {
      std::string dir = argv[1];
      for (int example = 0; example < 2; ++example) {
        const auto &input = example == 0 ? ties : near;
        auto reference = oracle(input);
        std::string prefix = dir + "/fixture_" + std::to_string(example);
        std::ofstream meta(prefix + ".txt");
        meta << input.n << ' ' << input.d << ' ' << input.q << '\n';
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
        }
        std::ofstream refs(prefix + ".oracle.csv");
        refs.precision(17);
        refs << "query,rank,id,distance,label\n";
        for (size_t q = 0; q < input.q; ++q)
          for (size_t rank = 0; rank < input.n; ++rank) {
            auto id = reference.order[q * input.n + rank];
            refs << q << ',' << rank << ',' << id << ','
                 << reference.distances[q * input.n + id] << ','
                 << input.labels[id] << '\n';
          }
      }
    }
    auto corrupted = search(valid, {2, Route::ScalarFullSort});
    corrupted.labels[0] += 1;
    require(!verify(valid, {2, Route::ScalarFullSort}, corrupted,
                    oracle(valid))
                 .labels,
            "verifier detects label mutation");
    corrupted.ids[1] = corrupted.ids[0];
    require(!verify(valid, {2, Route::ScalarFullSort}, corrupted,
                    oracle(valid))
                 .structure,
            "verifier detects duplicate ID");
    std::cout << "PASS cases=" << cases << " checks=" << checks
              << " FP64_ID_mismatches=" << mismatches << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
