"""Offline tests for the optional experiment schema, audit and CLI."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_optional as analysis
import run_optional_experiment as runner
import run_full_experiment as workflow


def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixture(path):
    path.mkdir(parents=True,exist_ok=True)
    routes=[('CPU','R0_scalar_full_sort'),('CPU','R1_avx2_topk_heap'),
            ('CPU','C0_cpu_scalar_topk_heap'),
            ('CPU','O1_cpu_avx2_tiled_bitonic'),('CPU','O2_cpu_scalar_early_abandon'),
            ('CUDA','R2_cuda_materialized_full_sort'),('CUDA','R3_cuda_fused_tiled_bitonic'),
            ('CUDA','O3_cuda_radix_scan_compact'),('CUDA','O4_cublas_gemm_reference'),
            ('CUDA','O5_cuda_stream_single'),('CUDA','O5_cuda_stream_double')]
    correct=[];memory=[];raw=[]
    for backend,route in routes:
        streaming=route.startswith('O5_')
        identity=dict(zip(analysis.IDENTITY,('0','32','9','2','5',backend,route,
            'scalar' if backend=='CPU' else 'warp_per_vector','32' if streaming else '',
            '7' if streaming else '',('2' if route.endswith('double') else '1') if streaming else '')))
        row=identity|dict(structure='1',labels='1',sorted='1',distance_tolerance='1',
            FP64_ID_agreement='1',ID_mismatches='0',max_abs_error='0.0000001',max_rel_error='0.0000001',
            deterministic='1',fixed_selection_match='' if streaming else '1',
            exact_reference_match='' if route.startswith('O4') else '1',reference_kind='fixture',passed='1')
        correct.append(row)
        m=identity|dict(input_device_bytes='100',output_device_bytes='12',aux_bytes='88',peak_device_bytes='200',
            pinned_host_bytes='128',logical_input_bytes='1024',cublas_workspace_bytes='0',
            setup_ms='1',database_norm_setup_ms='0',dimensions_per_candidate='3',abandoned_candidates='20')
        memory.append(m)
        modes=['host_complete'] if backend=='CPU' else (['end_to_end'] if streaming else ['resident','end_to_end'])
        for mode in modes:
            for repeat in range(3):
                sample=identity|{'mode':mode,'round':str(repeat),'complete_ms':str(5+.01*repeat),
                                'aux_bytes':'88','peak_device_bytes':'200'}
                for metric in analysis.METRICS:sample[metric]='0'
                if route in ('R2_cuda_materialized_full_sort','O3_cuda_radix_scan_compact'):
                    sample['distance_ms']='0.5'
                if route=='O4_cublas_gemm_reference':sample['gemm_ms']='0.5'
                sample['device_ms']='' if streaming or backend=='CPU' else (sample['complete_ms'] if mode=='resident' else '4')
                sample['pipeline_ms']='4' if streaming else '0'
                sample['dimensions_per_candidate']='3';sample['abandoned_candidates']='20'
                sample['estimated_overlap_ratio']='0.25' if streaming else '0'
                raw.append(sample)
    write(path/'raw.csv',raw);write(path/'correctness.csv',correct);write(path/'memory.csv',memory)
    (path/'run.txt').write_text('warmups=1\nrepeats=3\ncases=0\nstream_only=0\n')
    return raw,correct,memory


class OptionalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.raw,self.correct,self.memory=fixture(self.root/'measurements')
    def tearDown(self):self.temp.cleanup()

    def test_all_optional_routes_audit_and_report(self):
        report=analysis.generate(self.root/'measurements',self.root/'analysis',make_plots=False)
        self.assertTrue(report['all_correctness_passed'])
        self.assertEqual(report['configurations'],11)
        self.assertEqual(report['timing_groups'],15)
        self.assertEqual(report['raw_measurements'],45)
        text=(self.root/'analysis/optional_results.md').read_text()
        self.assertIn('Streaming always transfers chunks',text)
        self.assertIn('not evidence that the tested database exceeded',text)
        self.assertIn('bitwise direct/GEMM equality is not assumed',text)
        early=next(r for r in report['controlled_ratios'] if r['optional_route']=='O2_cpu_scalar_early_abandon')
        self.assertEqual(early['control_route'],'C0_cpu_scalar_topk_heap')

    def test_missing_round_rejected(self):
        write(self.root/'measurements/raw.csv',self.raw[:-1])
        with self.assertRaisesRegex(ValueError,'rounds'):analysis.audit(self.root/'measurements')

    def test_missing_whole_route_rejected(self):
        for filename,records in [('raw.csv',self.raw),('correctness.csv',self.correct),('memory.csv',self.memory)]:
            write(self.root/'measurements'/filename,[r for r in records if not r['route'].startswith('O4_')])
        with self.assertRaisesRegex(ValueError,'expected optional route'):analysis.audit(self.root/'measurements')

    def test_resident_boundary_mismatch_rejected(self):
        row=next(r for r in self.raw if r['backend']=='CUDA' and r['mode']=='resident')
        row['device_ms']='100';write(self.root/'measurements/raw.csv',self.raw)
        with self.assertRaisesRegex(ValueError,'Resident complete/device'):analysis.audit(self.root/'measurements')

    def test_resident_streaming_rejected(self):
        row=next(r for r in self.raw if r['route'].startswith('O5_'));row['mode']='resident'
        write(self.root/'measurements/raw.csv',self.raw)
        with self.assertRaisesRegex(ValueError,'cannot be GPU-resident'):analysis.audit(self.root/'measurements')

    def test_gemm_fixed_selection_failure_rejected(self):
        row=next(r for r in self.correct if r['route'].startswith('O4_'));row['fixed_selection_match']='0'
        write(self.root/'measurements/correctness.csv',self.correct)
        with self.assertRaisesRegex(ValueError,'Fixed-distance'):analysis.audit(self.root/'measurements')

    def test_corrupt_memory_rejected(self):
        row=next(r for r in self.memory if r['backend']=='CUDA');row['peak_device_bytes']='201'
        write(self.root/'measurements/memory.csv',self.memory)
        with self.assertRaisesRegex(ValueError,'allocations'):analysis.audit(self.root/'measurements')

    def test_fp64_id_differences_remain_visible(self):
        self.correct[0]['FP64_ID_agreement']='0';self.correct[0]['ID_mismatches']='2'
        write(self.root/'measurements/correctness.csv',self.correct)
        _,_,report=analysis.audit(self.root/'measurements')
        self.assertEqual(report['FP64_ID_position_differences'],2)

    def test_all_four_test_suites_must_execute(self):
        output='PASS gpu_cases=44 checks=2294\nPASS cases=111 checks=1145\nPASS optional_cpu_cases=482 checks=4824\n'
        with self.assertRaises(RuntimeError):runner.validate_test_output(output)
        runner.validate_test_output(output+'PASS optional_gpu_cases=17 checks=3514\n')

    def test_optional_preview_never_runs_a_command(self):
        destination=self.root/'new-results'
        with mock.patch.object(sys,'argv',['optional','--dry-run','--smoke','--results-dir',str(destination)]), \
             mock.patch.object(workflow,'run',side_effect=AssertionError('Unexpected command execution')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            runner.main()
        plan=json.loads(output.getvalue());self.assertEqual(plan['chunk_size'],64)
        self.assertFalse(destination.exists())

    def test_frozen_input_hashes_detect_mutation(self):
        directory=self.root/'inputs';directory.mkdir()
        write(directory/'inputs.csv',[dict(case='0',N='1',d='2',q='1',k='1',prefix='input_0')])
        (directory/'input_0.X.f32').write_bytes(struct.pack('<ff',.1,.2))
        (directory/'input_0.Q.f32').write_bytes(struct.pack('<ff',0,0))
        (directory/'input_0.labels.i32').write_bytes(struct.pack('<i',7))
        hashes=runner.input_hashes(directory)
        (directory/'input_sha256.json').write_text(json.dumps(hashes))
        (directory/'input_0.X.f32').write_bytes(struct.pack('<ff',.1,.3))
        with self.assertRaisesRegex(ValueError,'frozen CPU'):runner.input_hashes(directory)

    def test_shared_cuda_header_is_in_source_snapshot(self):
        path=self.root/'GPU/src/knn_kernels.cuh';path.parent.mkdir(parents=True);path.write_text('kernel v1')
        first=workflow.source_hashes(self.root)
        path.write_text('kernel v2');second=workflow.source_hashes(self.root)
        self.assertIn('GPU/src/knn_kernels.cuh',first)
        self.assertNotEqual(first,second)


if __name__=='__main__':unittest.main()
