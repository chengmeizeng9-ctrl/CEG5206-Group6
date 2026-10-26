#!/usr/bin/env python3
"""Validate and analyse the combined CPU/CUDA exact-kNN experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


R0 = "R0_scalar_full_sort"
R1 = "R1_avx2_topk_heap"
R2 = "R2_cuda_materialized_full_sort"
R3 = "R3_cuda_fused_tiled_bitonic"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(text: str) -> float:
    return float(text) if text not in ("", None) else math.nan


def quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("cannot summarise an empty measurement group")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return float(q1), float(median), float(q3)


def load_gpu_runs(directories: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    raw: list[dict[str, str]] = []
    correctness: list[dict[str, str]] = []
    memory: list[dict[str, str]] = []
    for index, directory in enumerate(directories):
        source = f"{index}:{directory.name}"
        for target, name in ((raw, "raw.csv"),
                             (correctness, "correctness.csv"),
                             (memory, "memory.csv")):
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(path)
            for row in read_csv(path):
                row["source"] = source
                target.append(row)
    return raw, correctness, memory


def summarise_gpu(raw: list[dict[str, str]]) -> list[dict[str, object]]:
    keys = ("source", "case", "N", "d", "q", "k", "route", "mapping",
            "tile_size", "mode")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        groups[tuple(row[key] for key in keys)].append(row)

    stage_fields = [
        "device_ms", "end_to_end_ms", "h2d_ms", "distance_ms", "pack_ms",
        "local_selection_ms", "full_sort_ms", "merge_ms", "gather_ms",
        "d2h_ms",
    ]
    summaries: list[dict[str, object]] = []
    for key, rows in groups.items():
        complete = [number(row["complete_ms"]) for row in rows]
        q1, median, q3 = quantiles(complete)
        summary: dict[str, object] = dict(zip(keys, key))
        summary.update({
            "samples": len(rows),
            "median_ms": median,
            "Q1_ms": q1,
            "Q3_ms": q3,
            "IQR_ms": q3 - q1,
            "IQR_over_median": (q3 - q1) / median if median else math.inf,
            "unstable": int(median == 0 or (q3 - q1) / median > 0.10),
            "queries_per_second": int(key[4]) / (median / 1000.0),
            "aux_bytes": int(rows[0]["aux_bytes"]),
            "peak_device_bytes": int(rows[0]["peak_device_bytes"]),
            "merge_levels": int(rows[0]["merge_levels"]),
        })
        for field in stage_fields:
            values = [number(row[field]) for row in rows]
            finite = [value for value in values if math.isfinite(value)]
            summary[f"median_{field}"] = float(np.median(finite)) if finite else math.nan
        summaries.append(summary)
    summaries.sort(key=lambda row: (
        str(row["source"]), int(str(row["case"])), str(row["route"]),
        str(row["mapping"]), str(row["tile_size"]), str(row["mode"])))
    return summaries


def cpu_unified_rows(cpu_summary: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in cpu_summary:
        rows.append({
            "source": "cpu",
            "case": row["case"],
            "N": row["N"],
            "d": row["d"],
            "q": row["q"],
            "k": row["k"],
            "backend": "CPU",
            "group": row["group"],
            "route": row["route"],
            "mapping": "",
            "tile_size": "",
            "mode": "host_complete",
            "samples": 20,
            "median_ms": 1000.0 * float(row["median"]),
            "Q1_ms": 1000.0 * float(row["Q1"]),
            "Q3_ms": 1000.0 * float(row["Q3"]),
            "IQR_ms": 1000.0 * float(row["IQR"]),
            "IQR_over_median": float(row["IQR"]) / float(row["median"]),
            "unstable": int(row["unstable"]),
            "queries_per_second": float(row["queries_per_second"]),
            "aux_bytes": int(row["aux_bytes"]),
            "peak_device_bytes": "",
            "merge_levels": "",
        })
    return rows


def gpu_unified_rows(gpu_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for source in gpu_summary:
        row = dict(source)
        row["backend"] = "CUDA"
        row["group"] = "0"
        rows.append(row)
    return rows


def find_row(rows: list[dict[str, object]], *, source: str | None = None,
             case: str = "0", route: str, mode: str,
             mapping: str = "", tile: str = "") -> dict[str, object]:
    matches = []
    for row in rows:
        if source is not None and row.get("source") != source:
            continue
        if (str(row.get("case")) == case and row.get("route") == route and
                row.get("mode") == mode and
                (not mapping or row.get("mapping") == mapping) and
                (not tile or str(row.get("tile_size")) == tile)):
            matches.append(row)
    if not matches:
        raise KeyError((source, case, route, mode, mapping, tile))
    matches.sort(key=lambda row: str(row.get("group", "0")) != "0")
    return matches[0]


def plot_baseline(output: Path, cpu_rows: list[dict[str, object]],
                  gpu_rows: list[dict[str, object]], core_source: str,
                  mapping: str, tile: str, gpu_mode: str) -> None:
    rows = [
        find_row(cpu_rows, case="0", route=R0, mode="host_complete"),
        find_row(cpu_rows, case="0", route=R1, mode="host_complete"),
        find_row(gpu_rows, source=core_source, case="0", route=R2,
                 mode=gpu_mode, mapping=mapping),
        find_row(gpu_rows, source=core_source, case="0", route=R3,
                 mode=gpu_mode, mapping=mapping, tile=tile),
    ]
    labels = ["R0 scalar sort", "R1 AVX2 heap", "R2 CUDA materialised",
              "R3 CUDA fused"]
    medians = [float(row["median_ms"]) for row in rows]
    lower = [float(row["median_ms"]) - float(row["Q1_ms"]) for row in rows]
    upper = [float(row["Q3_ms"]) - float(row["median_ms"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, medians, yerr=[lower, upper], capsize=4)
    axis.set_ylabel("Median complete batch latency (ms)")
    axis.set_title("Core exact-kNN routes: " +
                   ("GPU-resident" if gpu_mode == "resident"
                    else "GPU transfer-inclusive"))
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / f"baseline_{gpu_mode}.png", dpi=180)
    plt.close(fig)


def plot_stage_breakdown(output: Path, gpu_rows: list[dict[str, object]],
                         source: str, mapping: str, tile: str) -> None:
    r2 = find_row(gpu_rows, source=source, case="0", route=R2,
                  mode="resident", mapping=mapping)
    r3 = find_row(gpu_rows, source=source, case="0", route=R3,
                  mode="resident", mapping=mapping, tile=tile)
    labels = ["R2 materialised", "R3 fused"]
    stages = [
        ("Distance", [float(r2["median_distance_ms"]), 0.0]),
        ("Pack", [float(r2["median_pack_ms"]), 0.0]),
        ("Full sort", [float(r2["median_full_sort_ms"]), 0.0]),
        ("Fused local", [0.0, float(r3["median_local_selection_ms"])]),
        ("Merge", [0.0, float(r3["median_merge_ms"])]),
        ("Gather", [float(r2["median_gather_ms"]),
                    float(r3["median_gather_ms"])]),
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    bottom = np.zeros(2)
    for name, values in stages:
        axis.bar(labels, values, bottom=bottom, label=name)
        bottom += np.asarray(values)
    axis.set_ylabel("Median stage time (ms)")
    axis.set_title("CUDA stage breakdown, baseline case")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "gpu_stage_breakdown.png", dpi=180)
    plt.close(fig)


def plot_parameter_sweeps(output: Path, cpu_rows: list[dict[str, object]],
                          gpu_rows: list[dict[str, object]], core_source: str,
                          inputs: list[dict[str, str]], mapping: str,
                          tile: str) -> None:
    shape_by_case = {row["case"]: row for row in inputs}
    factors = [
        ("N", ["1", "0"]),
        ("d", ["2", "0", "3"]),
        ("q", ["4", "0", "5"]),
        ("k", ["6", "0", "7"]),
    ]
    if not all(case in shape_by_case for _, cases in factors for case in cases):
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    route_specs = [
        (R0, "R0", cpu_rows, "host_complete", "", ""),
        (R1, "R1", cpu_rows, "host_complete", "", ""),
        (R2, "R2", gpu_rows, "resident", mapping, ""),
        (R3, "R3", gpu_rows, "resident", mapping, tile),
    ]
    for axis, (factor, cases) in zip(axes.flat, factors):
        for route, label, rows, mode, route_mapping, route_tile in route_specs:
            points = []
            for case in cases:
                try:
                    row = find_row(
                        rows,
                        source=(core_source if rows is gpu_rows else None),
                        case=case,
                        route=route,
                        mode=mode,
                        mapping=route_mapping,
                        tile=route_tile,
                    )
                except KeyError:
                    continue
                x = int(shape_by_case[case][factor])
                y = (float(row["queries_per_second"]) if factor == "q"
                     else float(row["median_ms"]))
                points.append((x, y))
            points.sort()
            if points:
                axis.plot(*zip(*points), marker="o", label=label)
        axis.set_xlabel(factor)
        axis.set_ylabel("Queries per second" if factor == "q"
                        else "Median batch latency (ms)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "parameter_sweeps.png", dpi=180)
    plt.close(fig)


def plot_baseline_memory(output: Path, cpu_rows: list[dict[str, object]],
                         gpu_rows: list[dict[str, object]], core_source: str,
                         mapping: str, tile: str) -> None:
    rows = [
        find_row(cpu_rows, case="0", route=R0, mode="host_complete"),
        find_row(cpu_rows, case="0", route=R1, mode="host_complete"),
        find_row(gpu_rows, source=core_source, case="0", route=R2,
                 mode="resident", mapping=mapping),
        find_row(gpu_rows, source=core_source, case="0", route=R3,
                 mode="resident", mapping=mapping, tile=tile),
    ]
    labels = ["R0", "R1", "R2", "R3"]
    mib = [float(row["aux_bytes"]) / (1024 ** 2) for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(labels, mib)
    axis.set_ylabel("Allocated auxiliary storage (MiB)")
    axis.set_title("Baseline auxiliary-memory comparison")
    fig.tight_layout()
    fig.savefig(output / "baseline_auxiliary_memory.png", dpi=180)
    plt.close(fig)


def source_with_mappings(gpu_rows: list[dict[str, object]]) -> str | None:
    mappings: dict[str, set[str]] = defaultdict(set)
    for row in gpu_rows:
        if str(row["case"]) == "0" and row["mode"] == "resident":
            mappings[str(row["source"])].add(str(row["mapping"]))
    return next((source for source, values in mappings.items()
                 if {"thread_per_vector", "warp_per_vector"} <= values), None)


def plot_mapping_ablation(output: Path, gpu_rows: list[dict[str, object]],
                          tile: str) -> None:
    source = source_with_mappings(gpu_rows)
    if source is None:
        return
    labels = ["R2 thread", "R2 warp", "R3 thread", "R3 warp"]
    specs = [
        (R2, "thread_per_vector", ""),
        (R2, "warp_per_vector", ""),
        (R3, "thread_per_vector", tile),
        (R3, "warp_per_vector", tile),
    ]
    rows = []
    for route, mapping, route_tile in specs:
        try:
            rows.append(find_row(gpu_rows, source=source, case="0", route=route,
                                 mode="resident", mapping=mapping,
                                 tile=route_tile))
        except KeyError:
            return
    medians = [float(row["median_ms"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, medians)
    axis.set_ylabel("Median GPU-resident latency (ms)")
    axis.set_title("CUDA thread-per-vector versus warp-per-vector")
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / "mapping_ablation.png", dpi=180)
    plt.close(fig)


def source_with_tiles(gpu_rows: list[dict[str, object]], mapping: str) -> str | None:
    tiles: dict[str, set[str]] = defaultdict(set)
    for row in gpu_rows:
        if (str(row["case"]) == "0" and row["route"] == R3 and
                row["mode"] == "resident" and row["mapping"] == mapping):
            tiles[str(row["source"])].add(str(row["tile_size"]))
    return next((source for source, values in tiles.items() if len(values) >= 2),
                None)


def plot_tile_sensitivity(output: Path, gpu_rows: list[dict[str, object]],
                          mapping: str) -> None:
    source = source_with_tiles(gpu_rows, mapping)
    if source is None:
        return
    rows = [row for row in gpu_rows
            if row["source"] == source and str(row["case"]) == "0"
            and row["route"] == R3 and row["mode"] == "resident"
            and row["mapping"] == mapping]
    rows.sort(key=lambda row: int(str(row["tile_size"])))
    tiles = [int(str(row["tile_size"])) for row in rows]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(tiles, [float(row["median_ms"]) for row in rows], marker="o",
              label="Complete")
    axis.plot(tiles, [float(row["median_local_selection_ms"]) for row in rows],
              marker="o", label="Fused local")
    axis.plot(tiles, [float(row["median_merge_ms"]) for row in rows], marker="o",
              label="Merge")
    axis.set_xlabel("Tile size T")
    axis.set_ylabel("Median GPU-resident latency (ms)")
    axis.set_title("Fused-route tile-size sensitivity")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "tile_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(tiles, [float(row["aux_bytes"]) / (1024 ** 2) for row in rows],
              marker="o")
    axis.set_xlabel("Tile size T")
    axis.set_ylabel("Allocated auxiliary storage (MiB)")
    axis.set_title("Tile size versus surviving-candidate storage")
    fig.tight_layout()
    fig.savefig(output / "tile_memory.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-results", type=Path, required=True)
    parser.add_argument("--gpu-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-mapping", default="warp_per_vector")
    parser.add_argument("--primary-tile", default="128")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cpu_summary_raw = read_csv(args.cpu_results / "summary.csv")
    cpu_correctness = read_csv(args.cpu_results / "correctness.csv")
    cpu_inputs = read_csv(args.cpu_results / "inputs.csv")
    gpu_raw, gpu_correctness, gpu_memory = load_gpu_runs(args.gpu_results)

    required_cpu_fields = ("structure", "labels", "sorted",
                           "fixed_distance_selection", "distance_tolerance",
                           "deterministic")
    if not cpu_correctness or not all(all(row[field] == "1" for field in required_cpu_fields)
               for row in cpu_correctness):
        raise RuntimeError("one or more required CPU correctness checks failed")
    if not gpu_correctness or not all(row["passed"] == "1" for row in gpu_correctness):
        raise RuntimeError("one or more required CUDA correctness checks failed")

    gpu_summary = summarise_gpu(gpu_raw)
    cpu_rows = cpu_unified_rows(cpu_summary_raw)
    gpu_rows = gpu_unified_rows(gpu_summary)
    unified = cpu_rows + gpu_rows

    common_fields = [
        "source", "case", "N", "d", "q", "k", "backend", "group", "route",
        "mapping", "tile_size", "mode", "samples", "median_ms", "Q1_ms",
        "Q3_ms", "IQR_ms", "IQR_over_median", "unstable",
        "queries_per_second", "aux_bytes", "peak_device_bytes",
        "merge_levels",
    ]
    stage_fields = sorted({key for row in gpu_rows for key in row
                           if key.startswith("median_") and key != "median_ms"})
    write_csv(args.output / "unified_summary.csv", unified,
              common_fields + stage_fields)
    write_csv(args.output / "gpu_summary.csv", gpu_summary,
              [key for key in gpu_summary[0].keys()])

    correctness_rows = []
    for row in cpu_correctness:
        correctness_rows.append({"backend": "CPU", "source": "cpu", **row})
    for row in gpu_correctness:
        correctness_rows.append({"backend": "CUDA", **row})
    correctness_fields = sorted({key for row in correctness_rows for key in row})
    write_csv(args.output / "unified_correctness.csv", correctness_rows,
              correctness_fields)
    memory_fields = sorted({key for row in gpu_memory for key in row})
    write_csv(args.output / "gpu_memory.csv", gpu_memory, memory_fields)

    core_source = f"0:{args.gpu_results[0].name}"
    plot_baseline(args.output, cpu_rows, gpu_rows, core_source,
                  args.primary_mapping, args.primary_tile, "resident")
    plot_baseline(args.output, cpu_rows, gpu_rows, core_source,
                  args.primary_mapping, args.primary_tile, "end_to_end")
    plot_stage_breakdown(args.output, gpu_rows, core_source,
                         args.primary_mapping, args.primary_tile)
    plot_parameter_sweeps(args.output, cpu_rows, gpu_rows, core_source,
                          cpu_inputs, args.primary_mapping, args.primary_tile)
    plot_baseline_memory(args.output, cpu_rows, gpu_rows, core_source,
                         args.primary_mapping, args.primary_tile)
    plot_mapping_ablation(args.output, gpu_rows, args.primary_tile)
    plot_tile_sensitivity(args.output, gpu_rows, args.primary_mapping)

    baseline = {
        "R0_ms": float(find_row(cpu_rows, case="0", route=R0,
                                mode="host_complete")["median_ms"]),
        "R1_ms": float(find_row(cpu_rows, case="0", route=R1,
                                mode="host_complete")["median_ms"]),
        "R2_resident_ms": float(find_row(
            gpu_rows, source=core_source, case="0", route=R2,
            mode="resident", mapping=args.primary_mapping)["median_ms"]),
        "R3_resident_ms": float(find_row(
            gpu_rows, source=core_source, case="0", route=R3,
            mode="resident", mapping=args.primary_mapping,
            tile=args.primary_tile)["median_ms"]),
        "R2_end_to_end_ms": float(find_row(
            gpu_rows, source=core_source, case="0", route=R2,
            mode="end_to_end", mapping=args.primary_mapping)["median_ms"]),
        "R3_end_to_end_ms": float(find_row(
            gpu_rows, source=core_source, case="0", route=R3,
            mode="end_to_end", mapping=args.primary_mapping,
            tile=args.primary_tile)["median_ms"]),
    }
    report = {
        "cpu_correctness_records": len(cpu_correctness),
        "gpu_correctness_records": len(gpu_correctness),
        "gpu_raw_measurements": len(gpu_raw),
        "all_required_correctness_passed": True,
        "primary_mapping": args.primary_mapping,
        "primary_tile": args.primary_tile,
        "variable_cpu_groups": [
            {key: row[key] for key in ("case", "route", "group", "IQR_over_median")}
            for row in cpu_rows if int(row["unstable"]) == 1
        ],
        "variable_gpu_groups": [
            {key: row[key] for key in
             ("source", "case", "route", "mapping", "tile_size", "mode")}
            for row in gpu_summary if int(row["unstable"]) == 1
        ],
        "baseline": baseline,
        "speedups": {
            "R0_over_R1": baseline["R0_ms"] / baseline["R1_ms"],
            "R0_over_R2_resident": baseline["R0_ms"] / baseline["R2_resident_ms"],
            "R0_over_R3_resident": baseline["R0_ms"] / baseline["R3_resident_ms"],
            "R2_over_R3_resident": baseline["R2_resident_ms"] / baseline["R3_resident_ms"],
            "R2_over_R3_end_to_end": baseline["R2_end_to_end_ms"] / baseline["R3_end_to_end_ms"],
        },
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
