#include "knn.hpp"
#include <algorithm>
#include <cmath>

namespace knn {
namespace {
void bitonic(std::vector<Candidate> &tile, size_t count) {
  Better better;
  for (size_t size = 2; size <= count; size *= 2)
    for (size_t stride = size / 2; stride; stride /= 2)
      for (size_t i = 0; i < count; ++i) {
        const size_t partner = i ^ stride;
        if (partner > i && ((i & size) == 0 ? better(tile[partner], tile[i])
                                            : better(tile[i], tile[partner])))
          std::swap(tile[i], tile[partner]);
      }
}

void insert(std::vector<Candidate> &heap, Candidate value, size_t k) {
  Better better;
  if (heap.size() < k) {
    heap.push_back(value);
    std::push_heap(heap.begin(), heap.end(), better);
  } else if (better(value, heap.front())) {
    std::pop_heap(heap.begin(), heap.end(), better);
    heap.back() = value;
    std::push_heap(heap.begin(), heap.end(), better);
  }
}
} // namespace

void optional_search(const Input &input, Config config, Workspace &w,
                     Result &result, Diagnostics *diag) {
  const size_t k = config.k, tile = config.tile_size;
  if (config.route == Route::Avx2TiledBitonic) {
    const size_t tiles = input.n / tile + (input.n % tile != 0);
    const size_t capacity = checked_product(tiles, std::min(k, tile));
    if (w.tile.size() < tile || w.candidates_a.size() < capacity ||
        w.candidates_b.size() < capacity || w.offsets_a.size() < tiles + 1 ||
        w.offsets_b.size() < tiles + 1)
      throw Error(ErrorCode::INVALID_SHAPE, "prepare Bitonic workspace first");
  } else if (w.heap.capacity() < k) {
    throw Error(ErrorCode::INVALID_SHAPE, "prepare early-abandon heap first");
  }
  for (size_t qi = 0; qi < input.q; ++qi) {
    const float *query = input.queries.data() + qi * input.d;
    const Candidate *selected = nullptr;
    if (config.route == Route::ScalarEarlyAbandon ||
        config.route == Route::ScalarTopKHeap) {
      w.heap.clear();
      for (size_t id = 0; id < input.n; ++id) {
        const float *row = input.x.data() + id * input.d;
        const float threshold = w.heap.size() == k
                                    ? w.heap.front().distance
                                    : std::numeric_limits<float>::infinity();
        float sum = 0;
        size_t examined = 0;
        bool abandoned = false;
        for (size_t j = 0; j < input.d; ++j) {
          const float delta = query[j] - row[j];
          sum += delta * delta;
          examined = j + 1;
          // Strict inequality preserves exact-distance ID ties. The scalar
          // order and rounding match R0; SIMD reassociation is disabled here.
          if (config.route == Route::ScalarEarlyAbandon && sum > threshold) {
            abandoned = true;
            break;
          }
        }
        if (diag) {
          ++diag->candidates;
          diag->dimensions += examined;
          diag->abandoned_candidates += abandoned;
        }
        if (!abandoned)
          insert(w.heap, {sum == 0 ? 0.0f : sum, static_cast<uint32_t>(id)}, k);
      }
      std::sort(w.heap.begin(), w.heap.end(), Better{});
      selected = w.heap.data();
    } else {
      const size_t tiles = input.n / tile + (input.n % tile != 0);
      size_t used = 0;
      w.offsets_a[0] = 0;
      for (size_t t = 0; t < tiles; ++t) {
        const size_t begin = t * tile;
        const size_t count = std::min(tile, input.n - begin);
        for (size_t j = 0; j < tile; ++j) {
          const size_t id = begin + j;
          w.tile[j] =
              j < count ? Candidate{simd_distance(query,
                                                  input.x.data() + id * input.d,
                                                  input.d),
                                    static_cast<uint32_t>(id)}
                        : Candidate{std::numeric_limits<float>::infinity(),
                                    invalid_id};
        }
        bitonic(w.tile, tile);
        const size_t retained = std::min(k, count);
        std::copy_n(w.tile.begin(), retained, w.candidates_a.begin() + used);
        used += retained;
        w.offsets_a[t + 1] = used;
      }
      Candidate *a = w.candidates_a.data(), *b = w.candidates_b.data();
      size_t *offsets = w.offsets_a.data(), *next = w.offsets_b.data();
      size_t lists = tiles;
      while (lists > 1) {
        size_t outputs = 0, write = 0;
        next[0] = 0;
        for (size_t list = 0; list < lists; list += 2) {
          const size_t left = offsets[list], nl = offsets[list + 1] - left;
          const size_t right = offsets[list + 1];
          const size_t nr = list + 1 < lists ? offsets[list + 2] - right : 0;
          write += merge_topk(a + left, nl, a + right, nr, b + write, k);
          next[++outputs] = write;
        }
        std::swap(a, b);
        std::swap(offsets, next);
        lists = outputs;
      }
      selected = a;
      if (diag) {
        diag->candidates += input.n;
        diag->dimensions += checked_product(input.n, input.d);
      }
    }
    for (size_t rank = 0; rank < k; ++rank) {
      const size_t out = qi * k + rank;
      result.ids[out] = selected[rank].id;
      result.distances[out] = selected[rank].distance;
      result.labels[out] = input.labels[selected[rank].id];
    }
  }
}
} // namespace knn
