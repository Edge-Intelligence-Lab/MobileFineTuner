"""Temporary fixtures for the stdlib local checkpoint controller/exporter."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import effective_metrics as metrics
import resume_control as control


class ResumeControlTests(unittest.TestCase):
    def setUp(self):
        # Keep every write within the task's tools/ boundary, including fixtures.
        self.temporary = tempfile.TemporaryDirectory(prefix="resume tests ", dir=str(TOOLS))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run with spaces ; not-a-command"
        (self.run / "sessions").mkdir(parents=True)
        (self.run / "training_state").mkdir()
        (self.run / "run.lock").touch()
        self.events = {}
        self.write(self.run / "run_config.json", {"schema": 1})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def event(self, session, kind="session_start", **fields):
        events = self.events.setdefault(session, [])
        value = dict(type=kind, session_id=session, event_seq=len(events) + 1,
                     session_elapsed_sec=float(len(events)), **fields)
        events.append(value)
        path = self.run / "sessions" / session / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")
        return value

    def update(self, session, step, **fields):
        return self.event(session, "update", completed_updates=step, core_step_sec=0.5,
                          loss=1.0 / step, **fields)

    def checkpoint(self, generation, session, step, previous="", publish=True,
                   status="running", total=10):
        directory = self.run / "training_state" / generation
        state = {"schema": 1, "status": status,
                 "trainer": {"completed_updates": step, "total_steps": total,
                             "best_path": "", "timing": {"updates": step, "core_step_sec": step * 0.5}},
                 "dataset_state": "{}", "optimizer": {}}
        self.write(directory / "state.json", state)
        (directory / "state.safetensors").write_bytes(b"fixture payload (not loaded by exporter)")
        manifest = {"schema": 1, "generation": generation, "previous": previous,
                    "session_id": session, "event_seq": len(self.events.get(session, [])),
                    "completed_updates": step, "best_file": None,
                    "files": {name: (directory / name).stat().st_size
                              for name in ("state.json", "state.safetensors")}}
        self.write(directory / "manifest.json", manifest)
        if publish:
            self.write(self.run / "LATEST", {"schema": 1, "current": generation, "previous": previous})
            self.event(session, "checkpoint_committed", generation=generation,
                       completed_updates=step, status=status)
        return directory

    def basic(self, status="suspended", total=10):
        self.event("a")
        self.update("a", 1)
        return self.checkpoint("g1", "a", 1, status=status, total=total)

    def assert_valid_steps(self, report, steps):
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual([event["completed_updates"] for event in report["effective_updates"]], steps)

    def test_committed_orphan_replay_and_retained_manifests(self):
        first = self.basic()
        self.update("a", 2)
        self.checkpoint("g2", "a", 2, previous="g1")
        self.update("a", 3)
        self.checkpoint("g-orphan", "a", 3, previous="g2", publish=False)
        self.event("b")
        self.event("b", "resume", generation="g2", fallback=False, completed_updates=2)
        self.update("b", 3)
        self.checkpoint("g3", "b", 3, previous="g2")
        for name in ("state.json", "state.safetensors"):
            (first / name).unlink()
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        report = metrics.build_report(self.run)
        self.assert_valid_steps(report, [1, 2, 3])
        self.assertEqual(report["session_cutoffs"], {"b": 3, "a": 4})
        self.assertEqual(report["lineage"], ["g3", "g2", "g1"])
        self.assertEqual(report["summary"]["physical_update_count"], 4)
        self.assertEqual(report["summary"]["total_core_step_sec"], 1.5)
        abandoned = [item for item in report["physical_attempts"]
                     if item["classification"] == "after_cutoff" and item["event"]["type"] == "update"]
        self.assertEqual(len(abandoned), 1)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_fallback_excludes_corrupt_branch_and_abandoned_session(self):
        self.basic()
        self.update("a", 2)
        corrupt = self.checkpoint("bad", "a", 2, previous="g1")
        (corrupt / "manifest.json").write_text("{broken", encoding="utf-8")
        self.event("orphan")
        self.update("orphan", 99)
        self.event("b")
        self.event("b", "resume", generation="g1", fallback=True, completed_updates=1)
        self.update("b", 2)
        self.checkpoint("good", "b", 2, previous="g1", status="suspended")
        report = metrics.build_report(self.run)
        self.assert_valid_steps(report, [1, 2])
        self.assertEqual(report["lineage"], ["good", "g1"])
        self.assertEqual(report["session_cutoffs"]["a"], 2)
        self.assertEqual(report["effective_updates"][1]["session_id"], "b")
        self.assertTrue(any(item["classification"] == "abandoned_session"
                            for item in report["physical_attempts"]))

    def test_zero_update_checkpoint(self):
        self.event("a")
        self.checkpoint("g0", "a", 0)
        self.assert_valid_steps(metrics.build_report(self.run), [])

    def test_evaluation_replay_only_exports_committed_branch(self):
        self.basic()
        self.event("a", "evaluation", completed_updates=2, valid_ppl=99.0)
        self.event("b")
        self.event("b", "evaluation", completed_updates=2, valid_ppl=4.5)
        self.update("b", 2)
        self.checkpoint("g2", "b", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assert_valid_steps(report, [1, 2])
        self.assertEqual([e["valid_ppl"] for e in report["effective_evaluations"]], [4.5])
        self.assertEqual(report["summary"]["effective_evaluation_count"], 1)
        abandoned = [a for a in report["physical_attempts"]
                     if a["event"]["type"] == "evaluation" and a["classification"] == "after_cutoff"]
        self.assertEqual(len(abandoned), 1)
        output = io.StringIO()
        metrics.write_report(report, output, "csv")
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(sum(r["record_type"] == "effective_evaluation" for r in rows), 1)

    def test_duplicate_committed_evaluations_fail_closed(self):
        self.basic()
        for _ in range(2):
            self.event("a", "evaluation", completed_updates=2, valid_ppl=4.5)
        self.update("a", 2)
        self.checkpoint("g2", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertEqual(report["effective_evaluations"], [])

    def test_future_evaluation_is_not_repaired_by_later_progress(self):
        self.event("a")
        self.update("a", 1)
        self.event("a", "evaluation", completed_updates=2, valid_ppl=4.5)
        self.checkpoint("g1", "a", 1)
        self.update("a", 2)
        self.checkpoint("g2", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertEqual(report["effective_evaluations"], [])
        self.assertTrue(any("committing manifest" in error for error in report["errors"]))

    def test_evaluation_cannot_relabel_an_earlier_update(self):
        self.basic()
        self.event("a", "evaluation", completed_updates=1, valid_ppl=4.5)
        self.update("a", 2)
        self.checkpoint("g2", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertEqual(report["effective_evaluations"], [])

    def test_invalid_evaluation_step_returns_invalid_report(self):
        self.basic()
        self.event("a", "evaluation", completed_updates=None, valid_ppl=4.5)
        self.event("a", "evaluation", completed_updates=2, valid_ppl=4.5)
        self.update("a", 2)
        self.checkpoint("g2", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertIn("Invalid committed evaluation update", report["errors"])

    def test_no_power_zero_fill(self):
        self.basic()
        report = metrics.build_report(self.run)
        self.assertIsNone(report["effective_updates"][0]["power_w"])
        self.assertIsNone(report["effective_updates"][0]["energy_j"])
        self.update("a", 2, power_w=0.0, energy_j=1.5)
        self.checkpoint("g2", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertEqual(report["effective_updates"][1]["power_w"], 0.0)
        self.assertEqual(report["effective_updates"][1]["energy_j"], 1.5)

    def test_duplicate_logical_updates_fail_closed_keep_physical(self):
        self.basic()
        self.update("a", 1)
        self.checkpoint("duplicate", "a", 2, previous="g1")
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertEqual(report["effective_updates"], [])
        self.assertEqual(report["summary"]["physical_update_count"], 2)
        self.assertIsNone(report["summary"]["effective_update_count"])

    def test_missing_logical_update_fails_closed(self):
        self.basic()
        self.update("a", 3)
        self.checkpoint("gap", "a", 3, previous="g1")
        self.assertFalse(metrics.build_report(self.run)["valid"])

    def test_missing_committed_log_fails_without_deleting_other_attempts(self):
        self.basic()
        self.event("orphan")
        self.update("orphan", 2)
        (self.run / "sessions/a/events.jsonl").unlink()
        report = metrics.build_report(self.run)
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["physical_attempts"]), 2)
        self.assertTrue(any("Cannot read" in error for error in report["errors"]))

    def test_out_of_order_logical_updates_fail(self):
        self.event("a")
        self.update("a", 2)
        self.update("a", 1)
        self.checkpoint("reversed", "a", 2)
        self.assertFalse(metrics.build_report(self.run)["valid"])

    def test_invalid_ancestor_prefix_fails_even_with_valid_latest_steps(self):
        self.event("a")
        self.update("a", 1)
        self.checkpoint("wrong", "a", 2)
        self.update("a", 2)
        self.checkpoint("latest", "a", 2, previous="wrong")
        self.assertFalse(metrics.build_report(self.run)["valid"])

    def test_committed_event_sequence_corruption_fails(self):
        self.basic()
        path = self.run / "sessions/a/events.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        for replacement in ("{broken\n", lines[0], json.dumps(dict(self.events["a"][1], session_id="other")) + "\n"):
            with self.subTest(replacement=replacement):
                path.write_text(lines[0] + replacement + lines[2])
                report = metrics.build_report(self.run)
                self.assertFalse(report["valid"])
                self.assertEqual(len(report["physical_attempts"]), 3)

    def test_truncated_uncommitted_tail_is_retained_warning(self):
        self.basic()
        path = self.run / "sessions/a/events.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"type":"update","event_seq":4')
        report = metrics.build_report(self.run)
        self.assert_valid_steps(report, [1])
        self.assertTrue(report["warnings"])
        self.assertIn("raw_line", report["physical_attempts"][-1])

    def test_nonfinite_or_invalid_utf8_tail_does_not_break_export(self):
        self.basic()
        with (self.run / "sessions/a/events.jsonl").open("ab") as stream:
            stream.write(b'{"type":"update","loss":1e999}\n\xff\n')
        report = metrics.build_report(self.run)
        self.assert_valid_steps(report, [1])
        metrics.write_report(report, io.StringIO())
        self.assertEqual(len(report["warnings"]), 2)

    def test_invalid_pointer_does_not_choose_orphan(self):
        self.basic(status="completed", total=1)
        for text in ('{"schema":1,"current":"../escape","previous":""}',
                     '{broken', '[]', '{"schema":1,"schema":1}',
                     '{"schema":1,"current":"missing","previous":"g1"}'):
            with self.subTest(text=text):
                (self.run / "LATEST").write_text(text)
                status = metrics.inspect_status(self.run)
                self.assertEqual(status["pointer_status"], "invalid")
                self.assertIsNone(status["status"])
                self.assertIsNone(status["completed_updates"])
                report = metrics.build_report(self.run)
                self.assertFalse(report["valid"])
                self.assertTrue(report["physical_attempts"])

    def test_missing_pointer_is_not_completion(self):
        self.basic(status="completed", total=1)
        (self.run / "LATEST").unlink()
        status = metrics.inspect_status(self.run)
        self.assertEqual(status["pointer_status"], "missing")
        self.assertIsNone(status["status"])

    def test_invalid_state_preserves_manifest_progress_without_completion(self):
        directory = self.basic()
        (directory / "state.json").write_text("[]")
        status = metrics.inspect_status(self.run)
        self.assertEqual(status["pointer_status"], "valid")
        self.assertEqual(status["completed_updates"], 1)
        self.assertEqual(status["state_status"], "invalid")
        self.assertIsNone(status["status"])

    def test_manifest_cycle_and_pointer_mismatch(self):
        directory = self.basic()
        manifest = metrics.read_object(directory / "manifest.json")
        manifest["previous"] = "g1"
        self.write(directory / "manifest.json", manifest)
        self.assertFalse(metrics.build_report(self.run)["valid"])
        self.write(self.run / "LATEST", {"schema": 1, "current": "g1", "previous": "g1"})
        self.assertIn("Cycle", metrics.build_report(self.run)["errors"][0])

    def test_nonmonotonic_ancestor_cutoff_rejected(self):
        first = self.basic()
        self.event("a", "evaluation")
        self.checkpoint("g2", "a", 1, previous="g1")
        self.event("a", "evaluation")
        self.checkpoint("g3", "a", 1, previous="g2")
        manifest = metrics.read_object(first / "manifest.json")
        manifest["event_seq"] = 5  # g3=6, g2=4: cannot hide reversal under the max.
        self.write(first / "manifest.json", manifest)
        self.assertIn("cutoff decreases", metrics.build_report(self.run)["errors"][0])

    def test_status_reports_persisted_suspended_and_completed(self):
        self.basic()
        status = metrics.inspect_status(self.run)
        self.assertEqual(status["status"], "suspended")
        self.assertEqual(status["runtime_status"], "suspended")
        self.assertEqual(status["trainer"]["timing"]["updates"], 1)
        self.assertEqual(status["trainer"]["best_path"], "")
        self.checkpoint("done", "a", 1, previous="g1", status="completed", total=1)
        self.assertEqual(metrics.inspect_status(self.run)["status"], "completed")

    @unittest.skipUnless(os.name == "posix", "POSIX flock")
    def test_flock_distinguishes_live_dead_and_suspended(self):
        import fcntl
        self.basic(status="running")
        status = metrics.inspect_status(self.run)
        self.assertFalse(status["process_active"])
        self.assertEqual(status["runtime_status"], "running_dead")
        with (self.run / "run.lock").open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = metrics.inspect_status(self.run)
            self.assertTrue(status["process_active"])
            self.assertEqual(status["runtime_status"], "active")
            # A read-only probe must not disturb the trainer's lock.
            self.assertTrue(metrics.inspect_status(self.run)["process_active"])
        self.assertFalse(metrics.inspect_status(self.run)["process_active"])

    def test_missing_lock_is_unknown_and_not_created(self):
        self.basic(status="running")
        (self.run / "run.lock").unlink()
        status = metrics.inspect_status(self.run)
        self.assertIsNone(status["process_active"])
        self.assertEqual(status["runtime_status"], "unknown")
        self.assertFalse((self.run / "run.lock").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX flock")
    def test_resumed_process_can_hold_lock_with_persisted_suspended_state(self):
        import fcntl
        self.basic(status="suspended")
        with (self.run / "run.lock").open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = metrics.inspect_status(self.run)
            self.assertEqual(status["status"], "suspended")
            self.assertEqual(status["runtime_status"], "active")

    def test_suspend_atomic_replace_and_cleanup(self):
        replaced = []
        original = os.replace

        def replace(source, destination):
            self.assertEqual(Path(source).parent, self.run)
            self.assertEqual(json.loads(Path(source).read_text())["request"], "suspend")
            replaced.append(destination)
            original(source, destination)

        with mock.patch.object(control.os, "replace", side_effect=replace):
            result = control.request_suspend(self.run)
        self.assertEqual(replaced, [self.run / "SUSPEND"])
        self.assertFalse(result["acknowledged"])
        self.assertEqual(list(self.run.glob(".SUSPEND.*")), [])
        control.request_suspend(self.run)
        self.assertTrue((self.run / "SUSPEND").exists())

    def test_failed_suspend_does_not_replace_old_request_or_leave_temp(self):
        (self.run / "SUSPEND").write_text("old")
        with mock.patch.object(control.os, "replace", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                control.request_suspend(self.run)
        self.assertEqual((self.run / "SUSPEND").read_text(), "old")
        self.assertEqual(list(self.run.glob(".SUSPEND.*")), [])

    def executable_config(self):
        directory = self.root / "working directory with spaces"
        directory.mkdir()
        executable = self.root / "original executable ; echo not-shell"
        executable.write_text("#!" + sys.executable + "\n"
                              "import json, os, sys\n"
                              "from pathlib import Path\n"
                              "Path('child.json').write_text(json.dumps({\n"
                              "'argv': sys.argv, 'cwd': os.getcwd(),\n"
                              "'omp': os.environ.get('OMP_NUM_THREADS'),\n"
                              "'mkl': os.environ.get('MKL_NUM_THREADS')}))\n"
                              "sys.exit(7)\n")
        executable.chmod(0o700)
        config = {"schema": 1, "executable": {"path": str(executable)},
                  "working_directory": str(directory), "program": "fixture",
                  "arguments": ["old argv[0]", "--trainargs", "do not forward"],
                  "thread_environment": {key: None for key in control.THREAD_ENVIRONMENT}}
        config["thread_environment"]["OMP_NUM_THREADS"] = "2"
        self.write(self.run / "run_config.json", config)
        self.write(self.run / "LATEST", {"schema": 1, "current": "fixture", "previous": ""})
        return config, executable, directory

    def test_resume_real_subprocess_spaces_environment_and_no_trainargs(self):
        config, executable, directory = self.executable_config()
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "99", "MKL_NUM_THREADS": "9"}):
            code = control.resume_run(self.run, checkpoint_every_updates=3,
                                      suspend_after_updates=20, slice_seconds=0.5)
        self.assertEqual(code, 7)
        child = json.loads((directory / "child.json").read_text())
        self.assertEqual(child["argv"], [str(executable), "--resume_state", str(self.run),
                                         "--checkpoint_every_updates", "3", "--suspend_after_updates", "20",
                                         "--slice_seconds", "0.5"])
        self.assertEqual(child["cwd"], config["working_directory"])
        self.assertEqual(child["omp"], "2")
        self.assertIsNone(child["mkl"])

    def test_missing_saved_executable_refuses_fallback_and_no_launch(self):
        config, executable, _ = self.executable_config()
        for saved in (None, {}, {"path": "relative"}, {"path": str(self.root / "missing")}):
            with self.subTest(saved=saved):
                config["executable"] = saved
                config["arguments"] = [str(executable)]
                self.write(self.run / "run_config.json", config)
                with mock.patch.object(control.subprocess, "run") as child:
                    with self.assertRaises(metrics.RunError):
                        control.resume_run(self.run)
                    child.assert_not_called()

    def test_resume_invalid_controls_and_saved_configuration(self):
        config, _, _ = self.executable_config()
        for options in ({"checkpoint_every_updates": 0}, {"suspend_after_updates": -1},
                        {"slice_seconds": float("nan")}, {"slice_seconds": float("inf")},
                        {"resume_microbatch": 0}, {"resume_microbatch": True}, {"resume_microbatch": 2**31}):
            with self.subTest(options=options), mock.patch.object(control.subprocess, "run") as child:
                with self.assertRaises(metrics.RunError):
                    control.resume_run(self.run, **options)
                child.assert_not_called()
        for key, value in (("working_directory", "relative"), ("thread_environment", {})):
            with self.subTest(key=key):
                self.write(self.run / "run_config.json", dict(config, **{key: value}))
                with self.assertRaises(metrics.RunError):
                    control.resume_run(self.run)

    def test_resume_defaults_use_argv_list_and_never_shell(self):
        _, executable, directory = self.executable_config()
        with mock.patch.object(control.subprocess, "run", return_value=mock.Mock(returncode=0)) as child:
            self.assertEqual(control.resume_run(self.run), 0)
        args, options = child.call_args
        self.assertEqual(args, ([str(executable), "--resume_state", str(self.run)],))
        self.assertEqual(options["cwd"], str(directory))
        self.assertIs(options["shell"], False)
        self.assertIs(options["check"], False)

    def test_resume_microbatch_is_an_explicit_runtime_control(self):
        self.executable_config()
        with mock.patch.object(control.subprocess, "run", return_value=mock.Mock(returncode=0)) as child:
            self.assertEqual(control.resume_run(self.run, resume_microbatch=4), 0)
        self.assertEqual(child.call_args.args[0][-2:], ["--resume_microbatch", "4"])

    def test_json_csv_keep_both_views_and_null_measurements(self):
        self.basic()
        self.update("a", 2)
        report = metrics.build_report(self.run)
        stream = io.StringIO()
        metrics.write_report(report, stream)
        self.assertEqual(json.loads(stream.getvalue()), report)
        stream = io.StringIO()
        metrics.write_report(report, stream, "csv")
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[0]["record_type"], "summary")
        effective = [row for row in rows if row["record_type"] == "effective_update"]
        physical = [row for row in rows if row["record_type"] == "physical_attempt"]
        self.assertEqual(len(effective), 1)
        self.assertEqual(len(physical), 4)
        self.assertEqual(effective[0]["power_w"], "")
        self.assertIsNone(json.loads(effective[0]["details_json"])["power_w"])

    def test_cli_help_status_report_and_failure_exit(self):
        self.basic()
        for arguments in (["--help"], ["resume", "--help"], ["status", str(self.run)],
                          ["report", str(self.run)], ["report", str(self.run), "--format", "csv"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-B", str(TOOLS / "resume_control.py")] + arguments,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
        (self.run / "LATEST").write_text("broken")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(control.main(["report", str(self.run)]), 2)
        self.assertFalse(json.loads(output.getvalue())["valid"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(control.main(["status", str(self.run)]), 2)
        self.assertEqual(json.loads(output.getvalue())["pointer_status"], "invalid")

    def test_standalone_exporter_cli(self):
        self.basic()
        result = subprocess.run([sys.executable, "-B", str(TOOLS / "effective_metrics.py"), str(self.run)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_valid_steps(json.loads(result.stdout), [1])


if __name__ == "__main__":
    unittest.main()
