#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

export CUDA_HOME="$project_root/.tools/cuda-12.8"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$CUDA_HOME/compute-sanitizer:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$CUDACXX" || ! -x .venv/bin/python ]]; then
  echo 'The project-local CUDA toolkit and Python environment are required.' >&2
  exit 1
fi

exec .venv/bin/python experiments/run_full_experiment.py \
  --repo-root "$project_root" "$@"
