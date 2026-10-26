"""Offline workflow tests. These never launch CPU/CUDA benchmarks."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_full_experiment as driver
import summarize_sessions as sessions
import generate_report as report


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_rows(cpu_time=100, heap_time=10):
    data = []
    for route, mode, latency in ((sessions.R0, 'host_complete', cpu_time),
                                (sessions.R1, 'host_complete', heap_time),
                                (sessions.R2, 'resident', 8),
                                (sessions.R3, 'resident', 4),
                                (sessions.R2, 'end_to_end', 12),
                                (sessions.R3, 'end_to_end', 7)):
        cpu = route in (sessions.R0, sessions.R1)
        data.append(dict(case='0', N='100000', d='64', q='16', k='10',
                         backend='CPU' if cpu else 'CUDA', route=route,
                         mapping='' if cpu else 'warp_per_vector',
                         tile_size='128' if route == sessions.R3 else '', mode=mode,
                         group='0', source='cpu' if cpu else '0:gpu_core',
                         samples='20', median_ms=str(latency), Q1_ms=str(latency*.95),
                         Q3_ms=str(latency*1.05), IQR_ms=str(latency*.1),
                         IQR_over_median='0.1', queries_per_second=str(16000/latency),
                         unstable='0', aux_bytes='100', peak_device_bytes='200', merge_levels='1',
                         median_distance_ms='2', median_full_sort_ms='3',
                         median_local_selection_ms='2', median_merge_ms='1'))
    return data


def fingerprint(path):
    audit = {'status': 'PASS', 'cpu_variable_groups': [], 'gpu_variable_groups': [],
             'artifact_sha256': {p: hashlib.sha256((path/p).read_bytes()).hexdigest()
                                 for p in ('environment.json', 'analysis/unified_summary.csv')}}
    (path/'analysis/data_audit.json').write_text(json.dumps(audit))


def make_session(path, cpu_time, heap_time):
    (path/'cpu_core').mkdir(parents=True)
    (path/'gpu_core').mkdir()
    (path/'environment.json').write_text(json.dumps(dict(
        status='completed', smoke=False, hostname='test-machine', primary_mapping='warp_per_vector',
        primary_tile=128, cuda_architecture=61, nvcc='CUDA 12 test fixture', warmups=10, repeats=20,
        source_sha256={'kernel.cu': 'fixture-source-hash'})))
    (path/'cpu_core/input_sha256.json').write_text(json.dumps({'X': 'fixture-input-hash'}))
    (path/'gpu_core/run.txt').write_text('gpu_name=test-gpu\ncompute_capability=6.1\n'
                                      'cuda_runtime_version=12080\ncuda_driver_version=13000\n')
    write_csv(path/'analysis/unified_summary.csv', make_rows(cpu_time, heap_time))
    fingerprint(path)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        driver.LOG_FILE = None
        self.temp.cleanup()

    def test_existing_results_are_preserved(self):
        marker = self.root/'raw.csv'
        marker.write_text('retained evidence')
        with self.assertRaises(FileExistsError):
            driver.require_new_results(self.root)
        self.assertEqual(marker.read_text(), 'retained evidence')

    def test_dry_run_does_not_invoke_commands_or_create_results(self):
        destination = self.root/'not-created'
        with mock.patch.object(sys, 'argv', ['runner', '--dry-run', '--sessions', '3',
                                            '--results-dir', str(destination)]), \
             mock.patch.object(driver, 'run', side_effect=AssertionError('Unexpected execution')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            driver.main()
        self.assertFalse(destination.exists())
        self.assertEqual(json.loads(output.getvalue())['sessions'], 3)

    def test_skipped_gpu_test_blocks_all_measurements(self):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            if command[0] == 'ctest':
                return 'SKIP no CUDA device available\n' if 'gpu' in command[2] else 'PASS cases=1 checks=1\n'
            return ''
        args = argparse.Namespace(primary_tile=128, smoke=False, warmups=10,
                                  repeats=20, jobs=1, clean_build=False)
        destination = self.root/'results'
        with mock.patch.object(driver, 'source_hashes', return_value={'source': 'hash'}), \
             mock.patch.object(driver, 'optional_command', return_value='fixture'), \
             mock.patch.object(driver, 'version', return_value='fixture'), \
             mock.patch.object(driver, 'run', side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'did not execute'):
                driver.run_session(args, self.root, self.root/'build', destination,
                                   'fixture nvcc', 'auto', {'source': 'hash'}, None)
        self.assertFalse(any('knn_bench' in str(c) or 'knn_gpu_bench' in str(c) for c in commands))
        self.assertEqual(json.loads((destination/'environment.json').read_text())['status'], 'failed')

    def test_sanitizer_warning_blocks_gate(self):
        output = 'PASS gpu_cases=44 checks=2294\nERROR SUMMARY: 0 errors\n'
        with mock.patch.object(driver.shutil, 'which', return_value='/fixture/sanitizer'), \
             mock.patch.object(driver, 'run', return_value=output), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'racecheck'):
                driver.sanitizer_gate(Path('fixture'), self.root)

    def test_successful_session_freezes_metadata_before_audit(self):
        commands, audited_digest = [], []
        destination = self.root/'results'
        def fake_run(command, **kwargs):
            commands.append(command)
            if command[0] == 'ctest':
                return 'PASS gpu_cases=44 checks=2294\n' if 'gpu' in command[2] else 'PASS cases=111 checks=1145\n'
            if '--output' in command:
                output = Path(command[command.index('--output')+1])
                if output.name == 'gpu_mapping_ablation':
                    write_csv(output/'raw.csv', [dict(case='0', route=sessions.R3, mode='resident',
                        tile_size='128', mapping=m, complete_ms=t) for m,t in
                        [('thread_per_vector', '8'), ('warp_per_vector', '4')]])
            if any(str(item).endswith('verify_experiments.py') for item in command):
                env = destination/'environment.json'
                self.assertEqual(json.loads(env.read_text())['status'], 'completed')
                audited_digest.append(hashlib.sha256(env.read_bytes()).hexdigest())
            return ''
        args = argparse.Namespace(primary_tile=128, smoke=False, warmups=10,
                                  repeats=20, jobs=1, clean_build=False)
        with mock.patch.object(driver, 'source_hashes', return_value={'source': 'hash'}), \
             mock.patch.object(driver, 'optional_command', return_value='fixture'), \
             mock.patch.object(driver, 'version', return_value='fixture'), \
             mock.patch.object(driver, 'sanitizer_gate', return_value={'fixture': 'hash'}) as gate, \
             mock.patch.object(driver, 'run', side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            mapping, _ = driver.run_session(args, self.root, self.root/'build', destination,
                                            'fixture nvcc', 'auto', {'source': 'hash'}, None)
        self.assertEqual(mapping, 'warp')
        gate.assert_called_once()
        self.assertEqual(audited_digest[0], hashlib.sha256((destination/'environment.json').read_bytes()).hexdigest())

    def test_three_sessions_share_the_first_selected_mapping(self):
        destination = self.root/'study'
        with mock.patch.object(sys, 'argv', ['runner', '--sessions', '3', '--results-dir', str(destination)]), \
             mock.patch.object(driver, 'nvcc_major', return_value=(12, 'fixture')), \
             mock.patch.object(driver, 'source_hashes', return_value={'source': 'hash'}), \
             mock.patch.object(driver, 'run_session', return_value=('warp', {})) as session, \
             mock.patch.object(driver, 'run', return_value=''), \
             contextlib.redirect_stdout(io.StringIO()):
            # The fixture supplies only the paths checked before orchestration.
            repo = Path(driver.__file__).resolve().parents[1]
            self.assertTrue((repo/'GPU/CMakeLists.txt').exists())
            driver.main()
        self.assertEqual([c.args[5] for c in session.call_args_list], ['auto','warp','warp'])
        self.assertEqual(json.loads((destination/'study.json').read_text())['status'], 'completed')

    def test_paired_ratios_are_not_ratio_of_pooled_medians(self):
        paths = [self.root/f'session_{i}' for i in range(3)]
        for path, cpu, heap in zip(paths, (100, 100, 1000), (10, 50, 200)):
            make_session(path, cpu, heap)
        _, ratios, _ = sessions.summarize(paths)
        result = next(r for r in ratios if r['ratio'] == 'R0_over_R1')
        self.assertEqual(result['median_paired_speedup'], 5)
        self.assertEqual(result['min_paired_speedup'], 2)
        self.assertEqual(result['max_paired_speedup'], 10)

    def test_changed_inputs_are_not_combined(self):
        paths = [self.root/f'session_{i}' for i in range(2)]
        for path in paths:
            make_session(path, 100, 10)
        (paths[1]/'cpu_core/input_sha256.json').write_text('{"X": "different"}')
        with self.assertRaisesRegex(ValueError, 'input hashes differ'):
            sessions.summarize(paths)

    def test_changed_source_is_not_combined(self):
        paths = [self.root/f'session_{i}' for i in range(2)]
        for path in paths:
            make_session(path, 100, 10)
        env = json.loads((paths[1]/'environment.json').read_text())
        env['source_sha256'] = {'kernel.cu': 'changed'}
        (paths[1]/'environment.json').write_text(json.dumps(env))
        fingerprint(paths[1])
        with self.assertRaisesRegex(ValueError, 'source hashes differ'):
            sessions.summarize(paths)

    def test_summary_files_are_generated_from_separate_sessions(self):
        names = [f'session_{i:02d}' for i in range(1, 4)]
        for name in names:
            make_session(self.root/name, 100, 10)
        (self.root/'study.json').write_text(json.dumps(dict(status='completed',
            sessions=names, sessions_requested=3)))
        with mock.patch.object(sys, 'argv', ['summary', '--study', str(self.root)]), \
             contextlib.redirect_stdout(io.StringIO()):
            sessions.main()
        text = (self.root/'study_summary.md').read_text()
        self.assertIn('3 separate sessions', text)
        self.assertIn('not ratios of pooled medians', text)
        self.assertEqual(len(sessions.read_csv(self.root/'paired_speedups.csv')), 7)
        self.assertEqual(len(sessions.read_csv(self.root/'session_summary.csv')), 6)

    def test_changed_mapping_is_not_combined(self):
        paths = [self.root/f'session_{i}' for i in range(2)]
        for path in paths:
            make_session(path, 100, 10)
        env = json.loads((paths[1]/'environment.json').read_text())
        env['primary_mapping'] = 'thread_per_vector'
        (paths[1]/'environment.json').write_text(json.dumps(env))
        fingerprint(paths[1])
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            sessions.summarize(paths)

    def test_changed_audited_csv_is_not_combined(self):
        path = self.root/'session'
        make_session(path, 100, 10)
        with (path/'analysis/unified_summary.csv').open('a') as stream:
            stream.write('changed\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            sessions.load_session(path)

    def test_report_qualifies_cpu_noise_and_missing_smoke_plots(self):
        analysis = self.root/'analysis'
        data = make_rows()
        data[1]['unstable'] = '1'
        data[1]['IQR_over_median'] = '0.2'
        write_csv(analysis/'unified_summary.csv', data)
        write_csv(analysis/'unified_correctness.csv', [dict(backend='CPU', ID_mismatches='0',
                                                         max_abs_error='0', materialized_match='')])
        (analysis/'analysis.json').write_text(json.dumps(dict(all_required_correctness_passed=True,
            variable_gpu_groups=[], speedups={'R0_over_R1': 10, 'R2_over_R3_resident': 2,
                                             'R2_over_R3_end_to_end': 12/7})))
        with mock.patch.object(sys, 'argv', ['report', '--analysis', str(analysis),
                                            '--primary-mapping', 'warp_per_vector', '--primary-tile', '128']), \
             contextlib.redirect_stdout(io.StringIO()):
            report.main()
        text = (analysis/'final_results.md').read_text()
        self.assertIn('**1 CPU** and **0 GPU**', text)
        self.assertIn('20.0%', text)
        self.assertIn('universal FP64 ID equality is not implied', text)
        self.assertNotIn('![', text)


if __name__ == '__main__':
    unittest.main()
