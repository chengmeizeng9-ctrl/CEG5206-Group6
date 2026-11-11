#!/usr/bin/env python3
"""Build and validate all optional routes, then measure frozen CPU-export inputs."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import run_full_experiment as workflow


def input_hashes(directory):
    manifest=directory/'inputs.csv'
    with manifest.open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['case','N','d','q','k','prefix']:
            raise ValueError('Invalid inputs.csv schema')
        entries=list(reader)
    if not entries:raise ValueError('Empty input manifest')
    hashes={}
    for row in entries:
        n,d,q,k=(int(row[key]) for key in ('N','d','q','k'))
        if min(n,d,q,k)<=0 or k>n:raise ValueError('Invalid input shape')
        for suffix,size in (('.X.f32',n*d*4),('.Q.f32',q*d*4),('.labels.i32',n*4)):
            name=row['prefix']+suffix;path=(directory/name).resolve()
            path.relative_to(directory.resolve())
            if path.stat().st_size!=size:raise ValueError('Input length mismatch: '+name)
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
            hashes[name]=digest.hexdigest()
    expected=directory/'input_sha256.json'
    if expected.exists() and hashes!=json.loads(expected.read_text()):
        raise ValueError('Inputs differ from the frozen CPU input hashes')
    return hashes


def validate_test_output(output):
    for pattern in (r'PASS gpu_cases=\d+ checks=\d+',r'PASS optional_gpu_cases=\d+ checks=\d+',
                    r'PASS cases=\d+ checks=\d+',r'PASS optional_cpu_cases=\d+ checks=\d+'):
        if not re.search(pattern,output):raise RuntimeError('A required/optional correctness suite did not execute')


def sanitizer_gate(executable,logs):
    sanitizer=shutil.which('compute-sanitizer')
    if not sanitizer:raise RuntimeError('compute-sanitizer is required before optional measurements')
    records={}
    for tool,marker in (('memcheck','ERROR SUMMARY: 0 errors'),
                        ('racecheck','0 hazards displayed (0 errors, 0 warnings)')):
        output=workflow.run([sanitizer,'--tool',tool,'--error-exitcode','1',str(executable)],capture=True)
        (logs/(tool+'.log')).write_text(output)
        print(output,end='',flush=True)
        if marker not in output or not re.search(r'PASS optional_gpu_cases=\d+ checks=\d+',output):
            raise RuntimeError(tool+' did not execute clean optional GPU tests')
        records[tool]=hashlib.sha256(output.encode()).hexdigest()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs',type=Path,help='CPU results directory with inputs.csv and exact binary arrays')
    parser.add_argument('--build-dir',type=Path)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--mapping',choices=('thread','warp'),default='warp')
    parser.add_argument('--tile',type=int,choices=(32,64,128,256),default=128)
    parser.add_argument('--chunk-size',type=workflow.positive)
    parser.add_argument('--warmups',type=workflow.positive,default=10)
    parser.add_argument('--repeats',type=workflow.positive,default=20)
    parser.add_argument('--jobs',type=workflow.positive,default=2)
    parser.add_argument('--cases',help='Comma-separated manifest case IDs')
    parser.add_argument('--stream-only',action='store_true',help='Run only bounded-memory single/double streaming; avoids full-N GPU controls')
    parser.add_argument('--validate-only',action='store_true',help='Execute correctness comparisons without warm-ups or timing samples')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--clean-build',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    repo=args.repo_root.resolve();inputs=(args.inputs or repo/'results/full/cpu_core').resolve()
    build=(args.build_dir or repo/'build_optional/gpu').resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    results=(args.results_dir or repo/'results/optional_runs'/stamp).resolve()
    chunk=args.chunk_size or (64 if args.smoke else 32768)
    workflow.require_new_results(results)
    if args.dry_run:
        print(json.dumps({'results':str(results),'inputs':str(inputs),'chunk_size':chunk,
            'stream_only':args.stream_only,'smoke':args.smoke,'validate_only':args.validate_only,
            'routes':['CPU AVX2 Bitonic','CPU scalar early abandonment','CUDA radix/scan/compaction',
                      'cuBLAS GEMM reference','single/double-buffered CUDA streaming'],
            'actions':['build optional CUDA and CPU targets','run all correctness suites',
                       'memcheck and racecheck before timing','verify frozen input hashes',
                       'validate routes with zero timing samples' if args.validate_only else 'measure optional routes with contemporaneous controls',
                       'validation record' if args.validate_only else 'audit CSVs and generate optional report/plots']},indent=2))
        return
    major,nvcc=workflow.nvcc_major()
    if major!=12:raise RuntimeError('Use CUDA 12.x for the GTX 1050 optional study')
    frozen_inputs=input_hashes(inputs);sources=workflow.source_hashes(repo)
    results.mkdir(parents=True,exist_ok=True);logs=results/'logs';logs.mkdir()
    workflow.LOG_FILE=logs/'workflow.log'
    env={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
         'hostname':platform.node(),'platform':platform.platform(),'cpu':workflow.optional_command(['lscpu']),
         'nvcc':nvcc,'git_commit':workflow.optional_command(['git','rev-parse','HEAD'],repo),
         'source_sha256':sources,'input_source':str(inputs),'input_sha256':frozen_inputs,
         'mapping':args.mapping,'tile_size':args.tile,'chunk_size':chunk,
         'warmups':1 if args.smoke else args.warmups,'repeats':3 if args.smoke else args.repeats,
         'smoke':args.smoke,'stream_only':args.stream_only,'validation_only':args.validate_only,
         'nvidia_smi_before':workflow.optional_command(['nvidia-smi'])}
    workflow.save_json(results/'environment.json',env)
    workflow.save_json(results/'source_sha256.json',sources)
    workflow.save_json(results/'input_sha256.json',frozen_inputs)
    try:
        workflow.run(['cmake','-S',str(repo/'GPU'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                      '-DCMAKE_CUDA_ARCHITECTURES=61','-DKNN_GPU_OPTIONAL_ROUTES=ON'])
        command=['cmake','--build',str(build),'-j',str(args.jobs)]
        if args.clean_build:command.append('--clean-first')
        workflow.run(command)
        output=workflow.run(['ctest','--test-dir',str(build),'--verbose'],capture=True)
        (logs/'ctest.log').write_text(output);print(output,end='',flush=True);validate_test_output(output)
        for path in build.rglob('flags.make'):
            shutil.copy2(path,logs/('_'.join(path.relative_to(build).parts)))
        env['sanitizer_evidence_sha256']=sanitizer_gate(build/'knn_optional_gpu_tests',logs)
        if workflow.source_hashes(repo)!=sources:raise RuntimeError('Source changed during optional preparation')
        workflow.save_json(results/'environment.json',env)
        command=[str(build/'knn_optional_bench'),'--inputs',str(inputs),'--output',str(results/'measurements'),
                 '--mapping',args.mapping,'--tile',str(args.tile),'--chunk-size',str(chunk),
                 '--warmups',str(env['warmups']),'--repeats',str(env['repeats'])]
        if args.cases:command+=['--cases',args.cases]
        if args.smoke:command+=['--smoke']
        if args.stream_only:command+=['--stream-only']
        if args.validate_only:command+=['--validate-only']
        workflow.run(command)
        if input_hashes(inputs)!=frozen_inputs:raise RuntimeError('Frozen inputs changed during the optional run')
        if workflow.source_hashes(repo)!=sources:raise RuntimeError('Source changed during optional measurement')
        if args.validate_only:
            with (results/'measurements/correctness.csv').open(newline='') as stream:
                checked=list(csv.DictReader(stream))
            if not checked or not all(r['passed']=='1' for r in checked):
                raise RuntimeError('Optional validation-only comparisons failed')
            with (results/'measurements/raw.csv').open(newline='') as stream:
                if list(csv.DictReader(stream)):raise RuntimeError('Validation-only run unexpectedly contains timing samples')
            workflow.save_json(results/'validation_only.json',{'status':'PASS','configurations':len(checked),'timing_samples':0})
        else:
            workflow.run([sys.executable,str(repo/'experiments/analyze_optional.py'),'--results',str(results/'measurements'),
                          '--output',str(results/'analysis')])
        env['nvidia_smi_after']=workflow.optional_command(['nvidia-smi']);env['status']='completed'
    except BaseException as error:
        env.update(status='failed',error=str(error));raise
    finally:
        env['finished_at_utc']=datetime.now(timezone.utc).isoformat();workflow.save_json(results/'environment.json',env)
    print('\nOptional results:',results/('validation_only.json' if args.validate_only else 'analysis/optional_results.md'))


if __name__=='__main__':main()
