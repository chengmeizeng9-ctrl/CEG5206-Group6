# Execution Notes

This experiment builds Member 1's unchanged source code locally and follows the original procedure for one main experiment, without additional repetitions of the full experiment.

Source revision: `063769a4d6b04f6f572aa01d119c3092e4e5577d`. Source files, CMake options, data generation, eight configurations, floating-point controls, timing code and statistics are unchanged. The main experiment uses one compute thread, logical CPU 1, 10 warm-ups and 20 measured calls per route/group. If either route has IQR/median above 10%, the original program adds at most one group for both routes, retaining all records. No additional complete experiment is run.

The report uses case 0, group 0, matching Member 1's report. No samples or extra groups are discarded. Compiler flags and instruction evidence were checked locally. Release, independent clean Release and ASan/UBSan test suites passed; only Release was timed.

The local compiler is GCC 12.5.0; Member 1's retained build used GCC 12.3.1. Other differences include CPU, operating system, virtualization and dynamic libraries. The computer was plugged in at the preflight check and used the Balanced power scheme. Temperature, background load and mapping to a physical core were not locked.

The original `contract=1.1` run marker is preserved; shared contract v1.2 explicitly retains the eight CPU configurations. No original timing data are altered.
