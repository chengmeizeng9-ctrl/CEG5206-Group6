#!/usr/bin/env python3
"""Write a compact, portable summary of the CPU experiment environment."""
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)


def command_output(*command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


cpu_info = command_output("lscpu")
fields = {}
for line in cpu_info.splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
flags = set(re.split(r"\s+", fields.get("Flags", "")))
os_release = {}
release_file = Path("/etc/os-release")
if release_file.exists():
    for line in release_file.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')

summary = {
    "platform": {
        "os": os_release.get("PRETTY_NAME", platform.system()),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "reported_cpu": fields.get("Model name", "unavailable"),
        "logical_cpus": fields.get("CPU(s)", "unavailable"),
        "hypervisor": fields.get("Virtualization vendor", "unavailable"),
        "features": {name: name in flags for name in ("avx", "avx2", "fma")},
    },
    "toolchain": {
        "compiler": next(iter(command_output("g++", "--version").splitlines()), "unavailable"),
        "cmake": next(iter(command_output("cmake", "--version").splitlines()), "unavailable"),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "matplotlib": version("matplotlib"),
    },
    "experiment": {
        "contract": "1.1",
        "generator": "synthetic-v1",
        "seed": 5206,
        "threads": 1,
        "warmups_per_route": 10,
        "measured_calls_per_route": 20,
        "host_frequency_and_physical_core": "not_observable_from_guest",
    },
}
destination = output_dir / "environment_summary.json"
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(destination.name)
