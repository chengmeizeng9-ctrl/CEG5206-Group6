# Our CPU Demonstration

## 1. Correctness

Run from the project directory:

```bash
./build/knn_tests results
python3 tests/test_external.py
```

We explain three separate checks: output structure and labels; selection under identical distances; and numerical agreement with the independent FP64 oracle. In `fixture_1`, FP32 rounding turns distinct reference distances into an exact tie, after which our ID rule determines the ranking. Passing a distance tolerance does not imply identical neighbour IDs.

## 2. Eight CPU routes

```bash
./build/knn_bench results/demo --smoke
```

The small run demonstrates that every route executes and constructs complete sorted results. Our recorded main-parameter performance remains in `results/main/`; the demonstration does not replace the main experiment.

## 3. Experiment interpretation

We present `results/main/cpu_ablation.png`, `parameter_sweeps.png`, and `selection_vs_k.png`.

- R0 to R1 isolates tiling.
- R1 to R2 isolates SIMD.
- R1/R3/R5 and R2/R4/R6 compare selection strategies.
- R3 to R7 isolates early abandonment.

We use the mean dimensions and abandonment fraction in `diagnostics.csv` to explain arithmetic savings and threshold-check overhead. Complete-search time is our primary metric. Stage measurements and fixed-distance selection are separate diagnostic evidence. IQR and retained reruns show variability; our virtual-machine observations do not establish physical-hardware peak performance.

## 4. Reproducibility and integration

We show the build commands in the README, input hashes, instruction evidence, and environment records. We explain our replaceable Bitonic/merge interfaces and independent output verifier. Real GPU comparisons remain a later group experiment requiring real hardware and recorded outputs.

These materials provide our demonstration procedure and executable programs. They do not include presentation slides or a recorded video.
