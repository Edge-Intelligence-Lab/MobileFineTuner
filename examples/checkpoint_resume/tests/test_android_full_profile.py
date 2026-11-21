import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

import android_full_profile as profile


def archive_bytes(remove=(), overrides=None):
    files = {name: b'{}' for name in ('spec.json', 'control.json', 'ui_state.json',
                                    'profile_result.json', 'profile_attempts.json', 'run/LATEST')}
    files['attempts/1.log'] = b'fixture native log\n'
    files['run/training_state/g1/state.safetensors'] = b'x' * 4096
    files.update(overrides or {})
    for name in remove:
        del files[name]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:') as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


class AndroidFullProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'fixture, not model weights')
        self.data = self.root / 'data'
        self.data.mkdir()
        for name in ('train', 'valid'):
            (self.data / f'{name}.jsonl').write_text('{"ids": [1, 2], "mask": [1, 1]}\n')
        self.initial = self.root / 'initial.safetensors'
        self.initial.write_bytes(b'fixture, not adapter weights')
        self.archive = archive_bytes()
        self.failures = {}
        self.instrumentation_output = 'OK (1 test)\n'
        self.instrumentation_returncode = 0
        self.sequence = 0
        self.calls = []

    def fake_run(self, command, **kwargs):
        timeout = kwargs.get('timeout')
        self.assertIsNotNone(timeout, command)
        self.assertTrue(math.isfinite(timeout) and timeout > 0, command)
        if 'exec-out' in command:
            stage = 'extraction'
            kwargs['stdout'].write(self.archive)
            kwargs['stderr'].write('mock tar diagnostics\n')
        elif 'instrument' in command:
            stage = 'instrumentation'
            kwargs['stdout'].write(self.instrumentation_output)
        elif 'force-stop' in command:
            stage = 'cleanup'
        elif 'exec-in' in command:
            stage = 'staging'
            self.assertTrue(kwargs['stdin'].read())
        elif 'mkdir' in command:
            stage = 'mkdir'
        else:
            stage = 'preflight'
        self.calls.append((stage, command, kwargs))
        failure = self.failures.get(stage)
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, timeout)
        if failure == 'oserror':
            raise OSError(f'{stage} fixture I/O failure')
        if failure == 'exit':
            if kwargs.get('check'):
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 1)
        code = self.instrumentation_returncode if stage == 'instrumentation' else 0
        return subprocess.CompletedProcess(command, code)

    def invoke(self, serial='emulator-5556', output=None):
        self.sequence += 1
        self.output = output or self.root / f'output-{self.sequence}'
        self.calls = []
        argv = ['profile', '--adb', 'mock-adb', '--serial', serial,
                '--model', str(self.model), '--data', str(self.data),
                '--initial', str(self.initial), '--output', str(self.output)]
        with mock.patch('sys.argv', argv), mock.patch.object(profile.subprocess, 'run', side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            code = 0
            try:
                profile.main()
            except SystemExit as error:
                code = error.code
        return json.loads((self.output / 'result.json').read_text()), code

    def test_success_keeps_profile_settings_and_requires_both_results(self):
        with mock.patch.object(profile.time, 'monotonic', side_effect=[1, 5, 15]):
            result, code = self.invoke()
        self.assertEqual(code, 0)
        self.assertTrue(result['passed'])
        self.assertTrue(result['instrumentation_passed'])
        self.assertTrue(result['evidence_collected'])
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['extraction_returncode'], 0)
        self.assertEqual(result['elapsed_sec'], 10)
        self.assertEqual(result['scope'], 'Full Qwen seq64 via TrainingRunner; not SAF or background Worker')
        self.assertEqual(json.loads((self.output / 'spec.json').read_text()),
                         dict(model='qwen', steps=5, batch_size=1, seq_len=64, lora_r=8,
                              grad_accum=1, lr=0.0002, threads=4, slice_seconds=120,
                              pause_grace_seconds=300, min_available_mb=1024))
        instrument = next(call for call in self.calls if call[0] == 'instrumentation')
        self.assertEqual(instrument[2]['timeout'], 1300)
        self.assertIn('com.mobilefinetuner.visualizer.continuity.FullProfileCheck', instrument[1])
        self.assertEqual([call[0] for call in self.calls].count('staging'), 6)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / 'historical'
        output.mkdir()
        result_file = output / 'result.json'
        result_file.write_text('original evidence')
        with self.assertRaises(FileExistsError):
            self.invoke(output=output)
        self.assertEqual(result_file.read_text(), 'original evidence')
        self.assertFalse(self.calls)

    def test_non_emulator_preflight_preserves_failure_without_adb(self):
        result, code = self.invoke(serial='physical-device')
        self.assertEqual(code, 1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['stage'], 'preflight')
        self.assertIn('Disposable emulator only', (self.output / 'harness.log').read_text())
        self.assertFalse(self.calls)

    def test_missing_local_asset_is_recorded_without_device_mutation(self):
        self.initial.unlink()
        result, code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(result['stage'], 'preflight')
        self.assertIn('Missing staging asset', result['error'])
        self.assertTrue((self.output / 'harness.log').read_text())
        self.assertFalse(self.calls)

    def test_failed_preflight_does_not_extract_existing_remote_evidence(self):
        for failure in ('exit', 'timeout', 'oserror'):
            with self.subTest(failure=failure):
                self.failures = {'preflight': failure}
                result, code = self.invoke()
                self.assertEqual(code, 1)
                self.assertEqual(result['stage'], 'preflight')
                self.assertEqual(result['timeout'], failure == 'timeout')
                self.assertEqual([call[0] for call in self.calls], ['preflight'])
                self.assertFalse((self.output / 'records.tar').exists())
                self.assertTrue((self.output / 'harness.log').read_text())

    def test_mkdir_and_staging_failures_keep_logs_result_and_best_effort_archive(self):
        for stage in ('mkdir', 'staging'):
            for failure in ('exit', 'timeout', 'oserror'):
                with self.subTest(stage=stage, failure=failure):
                    self.failures = {stage: failure}
                    result, code = self.invoke()
                    self.assertEqual(code, 1)
                    self.assertFalse(result['passed'])
                    self.assertEqual(result['stage'], 'staging')
                    self.assertIn('error', result)
                    self.assertTrue((self.output / 'harness.log').read_text())
                    self.assertTrue((self.output / 'records.tar').exists())
                    self.assertNotIn('instrumentation', [call[0] for call in self.calls])

    def test_failed_instrumentation_cannot_pass_with_successful_extraction(self):
        for returncode, output in ((1, 'OK (1 test)\n'), (0, 'FAILURES!!!\n')):
            with self.subTest(returncode=returncode, output=output):
                self.instrumentation_returncode = returncode
                self.instrumentation_output = output
                result, code = self.invoke()
                self.assertEqual(code, 1)
                self.assertFalse(result['passed'])
                self.assertFalse(result['instrumentation_passed'])
                self.assertTrue(result['evidence_collected'])
                self.assertEqual(result['returncode'], returncode)

    def test_instrumentation_timeout_and_cleanup_failure_still_collect_and_report(self):
        for failure in ('exit', 'timeout', 'oserror'):
            with self.subTest(cleanup=failure):
                self.failures = {'instrumentation': 'timeout', 'cleanup': failure}
                self.instrumentation_output = 'partial instrumentation log\n'
                result, code = self.invoke()
                self.assertEqual(code, 1)
                self.assertTrue(result['timeout'])
                self.assertIn('cleanup_error', result)
                self.assertFalse(result['passed'])
                self.assertTrue(result['evidence_collected'])
                self.assertEqual((self.output / 'instrumentation.log').read_text(), self.instrumentation_output)
                self.assertIn('cleanup', [call[0] for call in self.calls])
                self.assertEqual(self.calls[-1][0], 'extraction')

    def test_tar_failure_cannot_pass_even_with_valid_archive(self):
        self.failures = {'extraction': 'exit'}
        result, code = self.invoke()
        self.assertEqual(code, 1)
        self.assertTrue(result['instrumentation_passed'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['evidence_collected'])
        self.assertEqual(result['extraction_returncode'], 1)
        self.assertEqual((self.output / 'records.tar').read_bytes(), self.archive)
        self.assertTrue((self.output / 'records.log').read_text())

    def test_extraction_timeout_or_io_error_retains_prior_results(self):
        for failure in ('timeout', 'oserror'):
            with self.subTest(failure=failure):
                self.failures = {'extraction': failure}
                result, code = self.invoke()
                self.assertEqual(code, 1)
                self.assertTrue(result['instrumentation_passed'])
                self.assertEqual(result['returncode'], 0)
                self.assertFalse(result['passed'])
                self.assertIn('extraction_error', result)
                self.assertTrue((self.output / 'records.tar').exists())

    def test_missing_invalid_or_truncated_archive_cannot_pass(self):
        complete = archive_bytes()
        with tarfile.open(fileobj=io.BytesIO(complete), mode='r:') as archive:
            offset = archive.getmember('run/training_state/g1/state.safetensors').offset_data
        archives = {
            'invalid_tar': b'not a tar archive',
            'missing_result': archive_bytes(remove=('profile_result.json',)),
            'missing_pointer': archive_bytes(remove=('run/LATEST',)),
            'missing_attempt': archive_bytes(remove=('attempts/1.log',)),
            'empty_attempt': archive_bytes(overrides={'attempts/1.log': b''}),
            'bad_json': archive_bytes(overrides={'profile_result.json': b'{broken'}),
            'non_object_json': archive_bytes(overrides={'profile_result.json': b'[]'}),
            'truncated_tensor_payload': complete[:offset + 256],
        }
        for name, content in archives.items():
            with self.subTest(archive=name):
                self.archive = content
                result, code = self.invoke()
                self.assertEqual(code, 1)
                self.assertTrue(result['instrumentation_passed'])
                self.assertEqual(result['extraction_returncode'], 0)
                self.assertFalse(result['passed'])
                self.assertFalse(result['evidence_collected'])
                self.assertIn('extraction_error', result)
                self.assertEqual((self.output / 'records.tar').read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
