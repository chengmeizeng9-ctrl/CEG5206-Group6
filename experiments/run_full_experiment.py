#!/usr/bin/env python3
"""Build, validate, measure and report the required CPU/CUDA study."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys

LOG_FILE: Path | None = None


def run(command: list[str], *, cwd: Path | None = None,
        capture: bool = False) -> str:
    command = list(map(str, command))
    heading = '+ ' + ' '.join(command)
    print(heading, flush=True)
    log = LOG_FILE.open('a') if LOG_FILE else None
    output = []
    try:
        if log:
            log.write(heading + '\n')
        with subprocess.Popen(command, cwd=cwd, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                if not capture:
                    print(line, end='', flush=True)
                if log:
                    log.write(line)
                    log.flush()
                output.append(line)
            status = process.wait()
        text = ''.join(output)
        if status:
            if capture:
                print(text, end='', flush=True)
            raise subprocess.CalledProcessError(status, command, output=text)
        return text if capture else ''
    finally:
        if log:
            log.close()


def optional_command(command: list[str], repo: Path | None = None) -> str:
    try:
        return run(command, cwd=repo, capture=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unavailable'


def nvcc_major() -> tuple[int, str]:
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('nvcc is not on PATH; use run_experiment.sh or CUDA 12.x')
    output = run([nvcc, '--version'], capture=True)
    match = re.search(r'release\s+(\d+)\.(\d+)', output)
    if not match:
        raise RuntimeError('Cannot parse nvcc version: ' + output)
    return int(match[1]), output


def choose_mapping(raw_csv: Path, tile: int) -> str:
    groups: dict[str, list[float]] = {}
    with raw_csv.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if (row['case'] == '0' and row['route'] == 'R3_cuda_fused_tiled_bitonic'
                    and row['mode'] == 'resident' and row['tile_size'] == str(tile)):
                groups.setdefault(row['mapping'], []).append(float(row['complete_ms']))
    if set(groups) != {'thread_per_vector', 'warp_per_vector'}:
        raise RuntimeError('Mapping ablation did not produce both R3 mappings')
    medians = {key: statistics.median(values) for key, values in groups.items()}
    print('Mapping ablation medians:', json.dumps(medians, indent=2))
    selected = min(medians, key=medians.get)
    print('Selected primary mapping:', selected)
    return selected


def source_hashes(repo: Path) -> dict[str, str]:
    paths = set()
    for directory in ('CPU/src', 'CPU/include', 'CPU/tests', 'CPU/scripts',
                      'GPU/src', 'GPU/include', 'GPU/tests', 'experiments'):
        paths.update(p for p in (repo / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.cpp', '.cu', '.hpp', '.py'))
    paths.update(repo / name for name in ('CPU/CMakeLists.txt', 'GPU/CMakeLists.txt',
                                         'run_experiment.sh', 'experiments/requirements.txt'))
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def require_new_results(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Results already exist at {path}. Choose a new --results-dir.')


def require_gpu_test_pass(output: str) -> None:
    if not re.search(r'PASS gpu_cases=\d+ checks=\d+', output):
        raise RuntimeError('GPU tests did not execute successfully; a skip is not a correctness pass.')


def sanitizer_gate(executable: Path, logs: Path) -> dict[str, str]:
    sanitizer = shutil.which('compute-sanitizer')
    if not sanitizer:
        raise RuntimeError('compute-sanitizer is required before final measurements')
    evidence = {}
    for tool, marker in (('memcheck', 'ERROR SUMMARY: 0 errors'),
                         ('racecheck', '0 hazards displayed (0 errors, 0 warnings)')):
        output = run([sanitizer, '--tool', tool, '--error-exitcode', '1',
                      str(executable)], capture=True)
        path = logs / f'{tool}.log'
        path.write_text(output)
        print(output, end='', flush=True)
        require_gpu_test_pass(output)
        if marker not in output:
            raise RuntimeError(f'{tool} did not report a clean summary')
        evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return evidence


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + '\n')


def run_session(args, repo: Path, build: Path, results: Path, nvcc_output: str,
                mapping_cli: str, frozen_sources: dict[str, str],
                gate_evidence: dict[str, str] | None) -> tuple[str, dict[str, str]]:
    global LOG_FILE
    require_new_results(results)
    if source_hashes(repo) != frozen_sources:
        raise RuntimeError('Source changed during this study; start a new study directory.')
    results.mkdir(parents=True, exist_ok=True)
    logs = results / 'logs'
    logs.mkdir()
    LOG_FILE = logs / 'workflow.log'
    cpu_source, gpu_source = repo / 'CPU', repo / 'GPU'
    cpu_build, gpu_build = build / 'cpu', build / 'gpu'
    cpu_results = results / 'cpu_core'
    gpu_mapping, gpu_core, gpu_tile = (results / p for p in
        ('gpu_mapping_ablation', 'gpu_core', 'gpu_tile_ablation'))
    analysis = results / 'analysis'
    environment = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': optional_command(['git', 'rev-parse', 'HEAD'], repo),
        'git_status': optional_command(['git', 'status', '--porcelain'], repo),
        'hostname': platform.node(), 'platform': platform.platform(),
        'cpu': optional_command(['lscpu']), 'python': sys.version,
        'packages': {p: version(p) for p in ('numpy', 'matplotlib')},
        'cmake': optional_command(['cmake', '--version']),
        'cxx': optional_command(['c++', '--version']),
        'cuda_architecture': 61, 'nvcc': nvcc_output.strip(),
        'primary_mapping_selection': mapping_cli, 'primary_tile': args.primary_tile,
        'warmups': 1 if args.smoke else args.warmups,
        'repeats': 3 if args.smoke else args.repeats,
        'cpu_warmups': 10, 'cpu_repeats': 20,
        'smoke': args.smoke, 'clean_build': args.clean_build,
        'nvidia_smi_before': optional_command(['nvidia-smi']),
        'source_sha256': frozen_sources,
    }
    save_json(results / 'source_sha256.json', frozen_sources)
    save_json(results / 'environment.json', environment)
    try:
        for source, destination in ((cpu_source, cpu_build), (gpu_source, gpu_build)):
            command = ['cmake', '-S', str(source), '-B', str(destination),
                       '-DCMAKE_BUILD_TYPE=Release']
            if source == gpu_source:
                command.append('-DCMAKE_CUDA_ARCHITECTURES=61')
            run(command)
            command = ['cmake', '--build', str(destination), '-j', str(args.jobs)]
            if args.clean_build:
                command.append('--clean-first')
            run(command)
            output = run(['ctest', '--test-dir', str(destination), '--verbose'], capture=True)
            (logs / f'{destination.name}_ctest.log').write_text(output)
            print(output, end='', flush=True)
            if source == gpu_source:
                require_gpu_test_pass(output)
            elif not re.search(r'PASS cases=\d+ checks=\d+', output):
                raise RuntimeError('CPU correctness tests did not execute successfully')
            for flags in destination.glob('CMakeFiles/*.dir/flags.make'):
                shutil.copy2(flags, logs / f'{destination.name}_{flags.parent.name}_flags.make')
                environment.setdefault('build_flags_sha256', {})[
                    f'{destination.name}/{flags.parent.name}'] = hashlib.sha256(flags.read_bytes()).hexdigest()

        if gate_evidence is None:
            gate_evidence = sanitizer_gate(gpu_build / 'knn_gpu_tests', logs)
        else:
            for path, digest in gate_evidence.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Sanitizer evidence changed during the study')
        environment['sanitizer_evidence_sha256'] = gate_evidence
        save_json(results / 'environment.json', environment)

        cpu_command = [str(cpu_build / 'knn_bench'), str(cpu_results)]
        run(cpu_command + (['--smoke'] if args.smoke else []))
        run([sys.executable, str(cpu_source / 'scripts/analyze.py'), str(cpu_results)])
        common = [str(gpu_build / 'knn_gpu_bench'), '--inputs', str(cpu_results),
                  '--warmups', str(environment['warmups']),
                  '--repeats', str(environment['repeats'])]
        smoke = ['--smoke'] if args.smoke else []
        run(common + ['--output', str(gpu_mapping), '--cases', '0',
                      '--routes', 'materialized,fused', '--mappings', 'thread,warp',
                      '--tiles', str(args.primary_tile)] + smoke)
        if mapping_cli == 'auto':
            mapping_name = choose_mapping(gpu_mapping / 'raw.csv', args.primary_tile)
            mapping_cli = 'thread' if mapping_name == 'thread_per_vector' else 'warp'
        else:
            mapping_name = 'thread_per_vector' if mapping_cli == 'thread' else 'warp_per_vector'
        environment['primary_mapping'] = mapping_name
        save_json(results / 'environment.json', environment)
        run(common + ['--output', str(gpu_core), '--routes', 'materialized,fused',
                      '--mappings', mapping_cli, '--tiles', str(args.primary_tile)] + smoke)
        run(common + ['--output', str(gpu_tile), '--cases', '0',
                      '--routes', 'materialized,fused', '--mappings', mapping_cli,
                      '--tiles', '32,64,128,256'] + smoke)
        run([sys.executable, str(repo / 'experiments/analyze_full.py'),
             '--cpu-results', str(cpu_results), '--gpu-results', str(gpu_core),
             str(gpu_mapping), str(gpu_tile), '--output', str(analysis),
             '--primary-mapping', mapping_name, '--primary-tile', str(args.primary_tile)])
        run([sys.executable, str(repo / 'experiments/generate_report.py'),
             '--analysis', str(analysis), '--primary-mapping', mapping_name,
             '--primary-tile', str(args.primary_tile)])
        environment['nvidia_smi'] = optional_command(['nvidia-smi'])
        if source_hashes(repo) != frozen_sources:
            raise RuntimeError('Source changed during measurements; retain this run as invalid.')
        environment['status'] = 'completed'
        environment['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(results / 'environment.json', environment)
        if not args.smoke:
            run([sys.executable, str(repo / 'experiments/verify_experiments.py'),
                 '--results', str(results), '--output', str(analysis / 'data_audit.json')])
    except BaseException as error:
        environment['status'] = 'failed'
        environment['error'] = str(error)
        raise
    finally:
        environment.setdefault('finished_at_utc', datetime.now(timezone.utc).isoformat())
        save_json(results / 'environment.json', environment)
    return mapping_cli, gate_evidence


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return result


def main() -> None:
    global LOG_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--results-dir', type=Path, help='New or empty destination; never overwritten')
    parser.add_argument('--jobs', type=positive, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument('--warmups', type=positive, default=10, help='GPU warm-ups (CPU uses 10)')
    parser.add_argument('--repeats', type=positive, default=20, help='GPU repeats (CPU uses 20)')
    parser.add_argument('--primary-tile', type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument('--primary-mapping', choices=('auto', 'thread', 'warp'), default='auto')
    parser.add_argument('--sessions', type=positive, default=1,
                        help='Independent full sessions; mapping is frozen after session 1')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--clean-build', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without running commands')
    args = parser.parse_args()
    if args.smoke and args.sessions != 1:
        parser.error('--smoke requires --sessions 1')
    repo = args.repo_root.resolve()
    build = (args.build_dir or repo / 'build_full').resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    category = 'smoke_runs' if args.smoke else ('studies' if args.sessions > 1 else 'runs')
    results = (args.results_dir or repo / 'results' / category / stamp).resolve()
    require_new_results(results)
    if args.dry_run:
        print(json.dumps({'results': str(results), 'sessions': args.sessions,
            'smoke': args.smoke, 'actions': [
                'release CPU/GPU build', 'CPU/GPU tests (skipped GPU fails gate)',
                'memcheck and racecheck before measurements',
                'CPU inputs and measurements', 'GPU mapping ablation and frozen mapping',
                'GPU core and tile measurements', 'analysis, report and full-run data audit',
                'session summary when sessions > 1'],
            'existing_results': 'preserved'}, indent=2))
        return
    for source in ('CPU/CMakeLists.txt', 'GPU/CMakeLists.txt', 'experiments/analyze_full.py'):
        if not (repo / source).is_file():
            raise FileNotFoundError(repo / source)
    major, nvcc_output = nvcc_major()
    if major != 12:
        raise RuntimeError('This GTX 1050 protocol requires a CUDA 12.x compiler.')
    sources = source_hashes(repo)
    mapping = args.primary_mapping
    evidence = None
    if args.sessions == 1:
        run_session(args, repo, build, results, nvcc_output, mapping, sources, evidence)
        print('\nResults:', results / 'analysis/final_results.md')
    else:
        results.mkdir(parents=True, exist_ok=True)
        study = {'status': 'running', 'sessions_requested': args.sessions, 'sessions': []}
        try:
            for number in range(1, args.sessions + 1):
                destination = results / f'session_{number:02d}'
                mapping, evidence = run_session(args, repo, build, destination,
                                                nvcc_output, mapping, sources, evidence)
                study['sessions'].append(destination.name)
                save_json(results / 'study.json', study)
            LOG_FILE = results / 'study.log'
            run([sys.executable, str(repo / 'experiments/summarize_sessions.py'),
                 '--study', str(results)])
            study['status'] = 'completed'
        except BaseException as error:
            study.update(status='failed', error=str(error))
            raise
        finally:
            save_json(results / 'study.json', study)
        print('\nStudy report:', results / 'study_summary.md')


if __name__ == '__main__':
    main()
