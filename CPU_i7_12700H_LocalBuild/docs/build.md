# Build and Run

From this folder in Linux, using GCC 12:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/check_evidence.py
./build/knn_bench results/new_run
python3 scripts/analyze.py results/new_run
```

Use a fresh output directory. Invoke the benchmark once; it implements the conditional additional-group rule internally. Sanitizer builds are for validation only and are not used for timing. The retained measured executable is `artifacts/release/knn_bench`; its hash and exact compilation commands are in `results/build_provenance.json`.

An independent clean Release build and a Debug build with `-DSANITIZE=ON` passed the same test suite. Build trees and machine-specific CMake caches are omitted.
