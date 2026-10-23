# Execution Notes

Source revision: `063769a4d6b04f6f572aa01d119c3092e4e5577d`.

This experiment uses Member 1's original Release executable and follows the original procedure for one main experiment, without additional repetitions of the full experiment.

The original computation source, executable, parameters, inputs, floating-point controls, measurement code and analysis scripts are retained. The benchmark was invoked once. Its existing variability rule alone determines whether a configuration receives one additional group. The report uses case 0, group 0, as in Member 1's original report; there is no aggregation across full experiments.

The report script contains a fixed historical date. Only the generated report date was corrected to 2026-10-23; the script was not modified.

Remaining differences are the CPU, Ubuntu/WSL2 versus openEuler/VMware, dynamic runtime libraries and analysis-tool versions. Physical-core mapping, temperature and background load were not locked. The executable still contains the original `contract=1.1` marker; shared contract v1.2 explicitly retains compatibility with the eight CPU configurations.
