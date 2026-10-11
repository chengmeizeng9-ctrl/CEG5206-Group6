# Public CPU Results

This directory contains the retained CPU timing records, summary statistics, correctness results, plots, compact environment description, instruction evidence, and small shared fixtures.

The full generated database and query arrays are included in `main/`. `main/input_sha256.json` records their hashes; `main/inputs.csv` records their shapes. The generator in `src/knn.cpp` and the `synthetic-v1` rule in the report reproduce the inputs.

We exclude process and `/proc` snapshots, machine-specific build directories and compiler databases, temporary smoke runs, and language-archive files. Compiled targets are retained separately under `../artifacts/`.
