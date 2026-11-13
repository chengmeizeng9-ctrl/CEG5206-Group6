# CPU Demonstration

Run the commands below from the `CPU_Ultra5_125H/` directory. The demonstration presents the required R0/R1 routes and the retained Ultra 5 125H experiment. The current build also contains tests for optional CPU routes.

## Correctness

Build and run the correctness suite:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Show that the scalar full-sort route and AVX2 Top-K heap each return complete, sorted results checked against the independent FP64 oracle. Then show fixed-distance heap selection and sorted-list merging checked against full-sort references.

## Required routes and measurements

```bash
./build/knn_bench results/demo --smoke
```

The demonstration covers R0 scalar full sort and R1 AVX2 plus Top-K heap. Present the retained CPU timing plots and explain that distance reduction order can affect near-boundary FP64 neighbor IDs even when the distance tolerance and each route's selection checks pass.

## Reproducibility

Show the eight main input configurations, stored input hashes, compiler and AVX2 instruction evidence, and the report and analysis commands in [CPU_Guide.md](../CPU_Guide.md). The demonstration uses only the required CPU implementation.
