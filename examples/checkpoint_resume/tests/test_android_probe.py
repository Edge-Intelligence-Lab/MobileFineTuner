import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from android_native_probe import collect_evidence, probe_metadata, outcome_matches, main


class AndroidProbeTests(unittest.TestCase):
    def test_failed_pull_is_not_successful_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch('android_native_probe.subprocess.run',
                            return_value=subprocess.CompletedProcess([], 1)):
                result = collect_evidence(['adb'], '/remote/run', output)
            self.assertFalse(result['evidence_collected'])
            self.assertEqual(result['pull_returncode'], 1)

    def test_success_requires_expected_pulled_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch('android_native_probe.subprocess.run',
                            return_value=subprocess.CompletedProcess([], 0)):
                self.assertFalse(collect_evidence(['adb'], '/run', output)['evidence_collected'])
                (output / 'run').mkdir()
                (output / 'run' / 'LATEST').write_text('{}')
                self.assertTrue(collect_evidence(['adb'], '/run', output)['evidence_collected'])

    def test_resume_does_not_invent_saved_math_configuration(self):
        for resume in (False, True):
            value = probe_metadata(argparse.Namespace(resume=resume, steps=5, seq=16, optimization='combined'))
            self.assertNotIn('blas', value)
            self.assertEqual('requested_training' in value, not resume)
            if not resume:
                self.assertEqual(16, value['requested_training']['seq'])
            self.assertEqual(value['mode'], 'resume' if resume else 'fresh')

    def test_observed_completion_must_match_fresh_or_suspend_request(self):
        state = dict(completed_updates=5, total_steps=5, status='completed')
        args = argparse.Namespace(resume=False, steps=10, stop=0)
        self.assertFalse(outcome_matches(state, args))
        args.steps = 5
        self.assertTrue(outcome_matches(state, args))
        args.stop = 2
        self.assertFalse(outcome_matches(state, args))
        self.assertTrue(outcome_matches(dict(completed_updates=2, total_steps=5, status='suspended'), args))
        args.resume = True
        args.steps = 1000
        args.stop = 0
        self.assertTrue(outcome_matches(state, args))  # Cumulative state, not 1000 executed updates.

    def test_preflight_failure_is_bounded_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            argv = ['probe', '--adb', 'adb', '--serial', 'emulator-5556',
                    '--remote', '/data/local/tmp/mft_vm17_fixture', '--output', str(output)]
            with mock.patch('sys.argv', argv), mock.patch('android_native_probe.subprocess.run',
                    side_effect=subprocess.TimeoutExpired('adb', 15)) as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            result = json.loads((output / 'result.json').read_text())
            self.assertFalse(result['passed'])
            self.assertEqual(result['stage'], 'preflight')
            self.assertTrue(result['timeout'])
            self.assertEqual(run.call_args.kwargs['timeout'], 15)


if __name__ == '__main__':
    unittest.main()
