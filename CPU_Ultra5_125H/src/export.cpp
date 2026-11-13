#include "knn.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

using namespace knn;
static size_t number(const char *text) {
  std::string s(text);
  size_t pos = 0;
  if (s.empty() || s[0] == '-')
    throw Error(ErrorCode::INVALID_ARGUMENT, "negative/empty argument");
  auto value = std::stoull(s, &pos);
  if (pos != s.size() || value > SIZE_MAX)
    throw Error(ErrorCode::INVALID_ARGUMENT, "invalid numeric argument");
  return size_t(value);
}
template <typename T>
static std::vector<T> load(const std::string &file, size_t count) {
  size_t bytes = checked_product(count, sizeof(T));
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  if (!in)
    throw Error(ErrorCode::EXECUTION_FAILURE, "open " + file);
  if (in.tellg() < 0 || static_cast<uint64_t>(in.tellg()) != bytes)
    throw Error(ErrorCode::INVALID_SHAPE, "file length " + file);
  std::vector<T> values(count);
  in.seekg(0);
  in.read(reinterpret_cast<char *>(values.data()), bytes);
  if (!in)
    throw Error(ErrorCode::EXECUTION_FAILURE, "read " + file);
  return values;
}
template <typename T>
static void save(const std::string &file, const std::vector<T> &values) {
  std::ofstream out(file, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            values.size() * sizeof(T));
  if (!out)
    throw Error(ErrorCode::EXECUTION_FAILURE, "write " + file);
}
int main(int argc, char **argv) {
  try {
    if (argc != 8 && argc != 9) {
      std::cerr << "Usage: knn_export INPUT_PREFIX N d q k route "
                   "OUTPUT_PREFIX [--full]\n";
      return 2;
    }
    uint32_t endian = 1;
    if (*reinterpret_cast<unsigned char *>(&endian) != 1)
      throw Error(ErrorCode::UNSUPPORTED_CONFIGURATION, "little endian export");
    std::string prefix = argv[1], out = argv[7];
    size_t n = number(argv[2]), d = number(argv[3]), q = number(argv[4]);
    size_t route_value = number(argv[6]);
    if (route_value > 1)
      throw Error(ErrorCode::INVALID_ARGUMENT, "route");
    size_t k = number(argv[5]);
    if (!n || !d || !q || !k || k > n)
      throw Error(ErrorCode::INVALID_ARGUMENT, "shape or k");
    if (n > UINT32_MAX || d > INT32_MAX)
      throw Error(ErrorCode::SIZE_OVERFLOW, "N/d representation");
    if (argc == 9 && std::string(argv[8]) != "--full")
      throw Error(ErrorCode::INVALID_ARGUMENT, "expected --full");
    Input input{n,
                d,
                q,
                load<float>(prefix + ".X.f32", checked_product(n, d)),
                load<float>(prefix + ".Q.f32", checked_product(q, d)),
                load<int32_t>(prefix + ".labels.i32", n)};
    Route route = static_cast<Route>(route_value);
    Config config{k, route};
    auto result = search(input, config);
    save(out + ".ids.u32", result.ids);
    save(out + ".distances.f32", result.distances);
    save(out + ".labels.i32", result.labels);
    if (argc == 9) {
      std::ofstream stream(out + ".full.f32", std::ios::binary);
      std::vector<float> distances(n);
      for (size_t qi = 0; qi < q; ++qi) {
        for (size_t id = 0; id < n; ++id)
          distances[id] = route == Route::Avx2TopKHeap
                              ? simd_distance(input.queries.data() + qi * d,
                                              input.x.data() + id * d, d)
                              : scalar_distance(input.queries.data() + qi * d,
                                                input.x.data() + id * d, d);
        stream.write(reinterpret_cast<const char *>(distances.data()),
                     n * sizeof(float));
      }
      if (!stream)
        throw Error(ErrorCode::EXECUTION_FAILURE, "write full distance table");
    }
    std::cout << route_name(route) << " exported " << q << " queries x "
              << config.k << " neighbours\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
