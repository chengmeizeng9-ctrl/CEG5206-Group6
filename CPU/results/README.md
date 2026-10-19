# Required CPU Results

This directory contains the required CPU route timings, correctness records, plots, compact environment summary, instruction evidence, and shared fixtures. `main/` includes all eight main configurations and their input arrays. `main/input_sha256.json` records array hashes; `main/inputs.csv` records shapes.

The retained routes are scalar full sort and AVX2 distance with a Top-K heap. The independently tested candidate-list merge implementation is documented in the CPU integration guide. Machine-local process snapshots and CMake caches are not included; compiled targets are under `../artifacts/`.
