#!/usr/bin/env python3
"""Real-Qwen cold-resume regression, using only Python's standard library.

Run: python3 tests/test_resume_process.py --binary /absolute/test_resume_process
     --artifacts /absolute/empty/output-directory

Defaults exercise accumulation 1 and 3, each with 10=4+6 and 200=73+127,
plus a separate post-update/pre-commit crash and recovery for each case.
All child logs, per-update tensor payloads, manifests and comparisons are kept.
The generated JSONL asset lives in its own retained /tmp directory.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import traceback


CRASH_EXIT_CODE = 86
FIXTURE_SAMPLES = 17
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS", "MFT_NUM_THREADS", "BLIS_NUM_THREADS",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_payloads(path):
    """Independently decode the tiny F32 safetensors files, without numpy."""
    raw = Path(path).read_bytes()
    require(len(raw) >= 8, f"Truncated tensor file: {path}")
    header_size = struct.unpack_from("<Q", raw)[0]
    require(0 < header_size <= len(raw) - 8, f"Invalid tensor header: {path}")
    header = json.loads(raw[8:8 + header_size])
    payload = raw[8 + header_size:]
    result, position = {}, 0
    for name, spec in sorted(header.items()):
        if name == "__metadata__":
            continue
        require(spec["dtype"] == "F32", f"Unexpected dtype: {path}: {name}")
        elements = math.prod(spec["shape"])
        begin, end = spec["data_offsets"]
        require(elements > 0 and begin == position and end == begin + elements * 4 and end <= len(payload),
                f"Invalid tensor offsets: {path}: {name}")
        data = payload[begin:end]
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", data)), f"Nonfinite tensor: {name}")
        result[name] = (spec["shape"], data)
        position = end
    require(result and position == len(payload), f"Unexpected tensor bytes: {path}")
    return result


def equal_tensor_files(left, right, context):
    a, b = tensor_payloads(left), tensor_payloads(right)
    require(a.keys() == b.keys(), f"{context}: tensor names differ")
    for name in a:
        require(a[name] == b[name], f"{context}: exact tensor mismatch: {name}")
    require(Path(left).read_bytes() == Path(right).read_bytes(), f"{context}: serialized tensor bytes differ")


def sessions(run_dir):
    result = {}
    for event_file in sorted((run_dir / "sessions").glob("*/events.jsonl")):
        records = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line]
        sid = event_file.parent.name
        require([event["event_seq"] for event in records] == list(range(1, len(records) + 1)),
                f"Noncontiguous event sequence: {event_file}")
        require(all(event["session_id"] == sid for event in records), f"Wrong event session: {event_file}")
        result[sid] = records
    return result


def committed_updates(run_dir):
    """Follow LATEST's committed lineage, never infer commits from raw events."""
    by_session = sessions(run_dir)
    generation = read_json(run_dir / "LATEST")["current"]
    seen, effective = set(), {}
    while generation:
        require(generation not in seen and Path(generation).name == generation, "Invalid generation lineage")
        seen.add(generation)
        manifest = read_json(run_dir / "training_state" / generation / "manifest.json")
        require(manifest["generation"] == generation, "Manifest generation mismatch")
        update = manifest["completed_updates"]
        if update:
            require(update not in effective, f"Duplicate committed update {update}")
            candidates = [event for event in by_session[manifest["session_id"]]
                          if event["event_seq"] <= manifest["event_seq"] and event["type"] == "update"
                          and event["completed_updates"] == update]
            require(len(candidates) == 1, f"Cannot resolve committed update {update}")
            effective[update] = candidates[0]
            best = manifest["best_file"]
            require(best["path"] == candidates[0]["trainer"]["best_path"], "Committed best pointer mismatch")
            require(Path(best["path"]).stat().st_size == best["bytes"], "Committed best payload missing")
        generation = manifest["previous"]
    require(sorted(effective) == list(range(1, len(effective) + 1)), "Committed trajectory has gaps")
    return [effective[step] for step in sorted(effective)]


def current_state(run_dir):
    generation = read_json(run_dir / "LATEST")["current"]
    directory = run_dir / "training_state" / generation
    return read_json(directory / "state.json"), directory / "state.safetensors"


def comparable_trainer(trainer):
    result = copy.deepcopy(trainer)
    result.pop("best_path")
    return result


def comparable_update(event):
    result = copy.deepcopy(event)
    for key in ("event_seq", "session_id", "session_elapsed_sec", "pid"):
        result.pop(key)
    result["snapshot"].pop("path")
    result["trainer"] = comparable_trainer(result["trainer"])
    return result


def validate_trajectory(events, total, accumulation):
    require(len(events) == total, f"Expected {total} effective updates, got {len(events)}")
    tokens_seen, targets_seen = 0, 0
    for step, event in enumerate(events, 1):
        require(event["completed_updates"] == step, "Wrong logical update")
        require(len(event["microbatches"]) == accumulation, "Wrong accumulation factor")
        tokens, targets = 0, 0
        for batch in event["microbatches"]:
            rows, sequence = batch["shape"]
            require(0 < rows <= 2 and sequence == 8, "Wrong microbatch shape")
            require(len(batch["ids"]) == len(batch["labels"]) == len(batch["attention_mask"]) == rows * sequence,
                    "Wrong microbatch payload length")
            tokens += sum(value != 0 for value in batch["attention_mask"])
            count = sum(0 <= label < 32 for index, label in enumerate(batch["labels"]) if index % sequence)
            require(count == batch["valid_targets"], "Masked target count mismatch")
            targets += count
        tokens_seen += tokens
        targets_seen += targets
        require(tokens == event["tokens"] and targets == event["targets"], "Update token accounting mismatch")
        require(event["trainer"]["tokens_seen"] == tokens_seen and event["trainer"]["targets_seen"] == targets_seen,
                "Cumulative token accounting mismatch")
        require(event["trainer"]["total_steps"] == total, "Scheduler budget changed on resume")
        require(event["trainer"]["grad_accum"] == accumulation, "Accumulation changed on resume")
        require(event["trainer"]["lr_bits"] == event["lr_bits"], "Trainer LR differs")
        optimizer = event["snapshot"]["optimizer"]
        require(optimizer["config"]["decoupled_weight_decay"] and optimizer["config"]["amsgrad"],
                "Test did not use AdamW/AMSGrad")
        lr_bits = struct.unpack("<I", struct.pack("<f", optimizer["config"]["learning_rate"]))[0]
        require(lr_bits == event["lr_bits"], "Saved optimizer LR differs")
        expected_names = set()
        for parameter in optimizer["parameters"]:
            require(parameter["initialized"] and parameter["step"] == step, "Optimizer step restarted or skipped")
            expected_names.add("parameter/" + parameter["name"])
            expected_names.update("optimizer/" + moment + "/" + parameter["name"] for moment in ("m", "v", "vhat"))
        require(tensor_payloads(event["snapshot"]["path"]).keys() == expected_names, "Fullstate tensors missing/extra")
        norm = struct.unpack("<d", struct.pack("<Q", event["gradient_squared_norm_bits"]))[0]
        require(math.isfinite(norm) and norm > 0, "No real autograd update")
        if step > 1:
            require(event["dataset_before"] == events[step - 2]["dataset_after"], "Sampler boundary discontinuity")
    require(events[-1]["lr"] < events[0]["lr"], "Cosine LR did not advance")
    orders = {tuple(event["dataset_after"]["order"]) for event in events}
    require(len(orders) > 1, "Test failed to cross a shuffled epoch boundary")
    require(events[0]["dataset_before"]["order"] != list(range(FIXTURE_SAMPLES)), "Dataset was not shuffled")


def compare_runs(baseline, candidate, total, accumulation):
    reference, actual = committed_updates(baseline), committed_updates(candidate)
    validate_trajectory(reference, total, accumulation)
    validate_trajectory(actual, total, accumulation)
    comparisons = []
    for a, b in zip(reference, actual):
        step = a["completed_updates"]
        require(comparable_update(a) == comparable_update(b), f"Exact event/metadata mismatch at update {step}")
        equal_tensor_files(a["snapshot"]["path"], b["snapshot"]["path"], f"Update {step}")
        equal_tensor_files(a["trainer"]["best_path"], b["trainer"]["best_path"], f"Best at update {step}")
        comparisons.append({"update": step, "loss_bits": a["loss_bits"], "lr_bits": a["lr_bits"],
                            "tokens_seen": a["trainer"]["tokens_seen"],
                            "fullstate_sha256": sha256(a["snapshot"]["path"]),
                            "reference": a["snapshot"]["path"], "candidate": b["snapshot"]["path"]})
    left, left_tensors = current_state(baseline)
    right, right_tensors = current_state(candidate)
    require(left["status"] == right["status"] == "completed", "Final run not completed")
    require(left["optimizer"] == right["optimizer"] and left["dataset_state"] == right["dataset_state"],
            "Final committed optimizer/dataset state differs")
    require(comparable_trainer(left["trainer"]) == comparable_trainer(right["trainer"]), "Final trainer state differs")
    equal_tensor_files(left_tensors, right_tensors, "Final committed fullstate")
    return comparisons


def check_restart(run_dir, pid, previous):
    records = [event for group in sessions(run_dir).values() for event in group]
    starts = [event for event in records if event["type"] == "worker_start" and event["pid"] == pid]
    require(len(starts) == 1, "Missing fresh-process start record")
    start = starts[0]
    require(start["mode"] == "resume" and start["completed_updates"] == previous["completed_updates"],
            "Child did not resume at the saved update")
    require(start["trainer"] == previous["trainer"] and start["dataset"] == previous["dataset_after"],
            "Restored counters/best pointer/dataset changed")
    require(start["snapshot"]["optimizer"] == previous["snapshot"]["optimizer"], "Restored Adam metadata differs")
    equal_tensor_files(start["snapshot"]["path"], previous["snapshot"]["path"], "Fresh process initial state")
    return start


def make_fixture():
    directory = Path(tempfile.mkdtemp(prefix="mft-resume-jsonl-", dir="/tmp"))
    path = directory / "samples.jsonl"
    with path.open("x", encoding="utf-8") as output:
        for sample in range(FIXTURE_SAMPLES):
            length = 5 + sample % 4
            ids = [sample + 1] + [1 + (sample * 7 + position * 3) % 31 for position in range(1, length)]
            mask = [0] + [int((position + sample) % 3 != 0) for position in range(1, length)]
            mask[-1] = 1
            output.write(json.dumps({"ids": ids, "mask": mask}) + "\n")
    return path


class Driver:
    def __init__(self, binary, artifacts, fixture, timeout):
        self.binary, self.artifacts, self.fixture, self.timeout = binary, artifacts, fixture, timeout
        self.invocations = []
        self.environment = os.environ.copy()
        self.environment.update({name: "1" for name in THREAD_VARIABLES})

    def child(self, tag, run_dir, total, accumulation, mode, stop=0, crash=0):
        command = [str(self.binary), "--mode", mode, "--run-dir", str(run_dir),
                   "--data", str(self.fixture), "--total-steps", str(total),
                   "--grad-accum", str(accumulation), "--stop-after", str(stop), "--crash-after", str(crash)]
        record = {"tag": tag, "command": command, "expected_returncode": CRASH_EXIT_CODE if crash else 0}
        self.invocations.append(record)
        started = time.monotonic()
        with (self.artifacts / (tag + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=self.environment)
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                record["timed_out"] = True
                raise
            finally:
                record["wall_seconds"] = time.monotonic() - started
                write_json(self.artifacts / "invocations.json", self.invocations)
        require(record["returncode"] == record["expected_returncode"], f"Child {tag} failed; see {tag}.log")
        require(len({item["pid"] for item in self.invocations}) == len(self.invocations), "Worker PIDs were reused")
        starts = [event for group in sessions(run_dir).values() for event in group
                  if event["type"] == "worker_start" and event["pid"] == process.pid]
        require(len(starts) == 1, "Executed child PID does not match its recorded PID")
        return process.pid

    def case(self, total, cut, accumulation):
        label = f"steps{total}-accum{accumulation}"
        directory = self.artifacts / label
        directory.mkdir()
        baseline, split, crash_run = (directory / name for name in ("continuous", "split", "crash"))
        self.child(label + "-continuous", baseline, total, accumulation, "new")
        self.child(label + "-suspend", split, total, accumulation, "new", stop=cut)
        boundary = committed_updates(split)[-1]
        require(boundary["completed_updates"] == cut, "Wrong suspend boundary")
        require(current_state(split)[0]["status"] == "suspended", "Suspend was not committed")
        resumed_pid = self.child(label + "-resume", split, total, accumulation, "resume")
        check_restart(split, resumed_pid, boundary)
        split_comparison = compare_runs(baseline, split, total, accumulation)
        self.child(label + "-crash-prefix", crash_run, total, accumulation, "new", stop=cut)
        boundary = committed_updates(crash_run)[-1]
        latest_before = read_json(crash_run / "LATEST")
        killed_pid = self.child(label + "-crash-attempt", crash_run, total, accumulation, "resume", crash=cut + 1)
        check_restart(crash_run, killed_pid, boundary)
        require(read_json(crash_run / "LATEST") == latest_before, "Uncommitted crash changed LATEST")
        require(len(committed_updates(crash_run)) == cut, "Failed attempt became committed")
        attempts = [event for group in sessions(crash_run).values() for event in group if event["type"] == "update"]
        orphan = [event for event in attempts if event["pid"] == killed_pid]
        require(len(orphan) == 1 and orphan[0]["completed_updates"] == cut + 1, "Missing extra physical attempt")
        orphan = orphan[0]
        orphan_best = orphan["trainer"]["best_path"]
        require(Path(orphan_best).exists() and orphan_best != boundary["trainer"]["best_path"], "No orphan best candidate")
        recovered_pid = self.child(label + "-recovery", crash_run, total, accumulation, "resume")
        recovered_start = check_restart(crash_run, recovered_pid, boundary)
        require(recovered_start["trainer"]["best_path"] != orphan_best, "Recovery adopted uncommitted best candidate")
        crash_comparison = compare_runs(baseline, crash_run, total, accumulation)
        replayed = committed_updates(crash_run)[cut]
        require(comparable_update(orphan) == comparable_update(replayed), "Recomputed failed attempt changed trajectory")
        equal_tensor_files(orphan["snapshot"]["path"], replayed["snapshot"]["path"], "Abandoned/replayed attempt")
        require(replayed["trainer"]["best_path"] != orphan_best and Path(orphan_best).exists(), "Best attempt identity lost")
        all_attempts = [event for group in sessions(crash_run).values() for event in group if event["type"] == "update"]
        require(len(all_attempts) == total + 1, "Physical attempt count must include the abandoned update")
        for run in (baseline, split):
            count = sum(event["type"] == "update" for group in sessions(run).values() for event in group)
            require(count == total, "Unexpected physical attempt in uninterrupted/graceful run")
        initial = next(event for group in sessions(baseline).values() for event in group if event["type"] == "worker_start")
        initial_tensors = tensor_payloads(initial["snapshot"]["path"])
        final_tensors = tensor_payloads(current_state(baseline)[1])
        require(any(initial_tensors[name] != final_tensors[name] for name in initial_tensors), "Model weights never changed")
        result = {"status": "passed", "total_steps": total, "cut": cut, "grad_accum": accumulation,
                  "split": split_comparison, "crash": crash_comparison,
                  "physical_attempts": total + 1, "committed_updates": total,
                  "abandoned_update": cut + 1, "crashed_pid": killed_pid, "recovered_pid": recovered_pid,
                  "orphan_best_path": orphan_best, "restored_best_path": boundary["trainer"]["best_path"],
                  "best_policy": "synthetic_negative_update_for_pointer_test"}
        write_json(directory / "comparison.json", result)
        print(f"PASS {label}: {total}={cut}+{total-cut}; crash replay exact; physical attempts={total+1}", flush=True)
        return {key: value for key, value in result.items() if key not in ("split", "crash")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, help="Must be absent or empty; defaults to a retained /tmp directory")
    parser.add_argument("--grad-accums", default="1,3", help="Comma-separated accumulation factors (default: 1,3)")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds per child process")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    require(binary.is_file() and os.access(binary, os.X_OK), "--binary is not executable")
    accumulations = [int(value) for value in args.grad_accums.split(",")]
    require(accumulations and len(set(accumulations)) == len(accumulations) and all(1 <= value <= 8 for value in accumulations),
            "Accumulations must be unique integers in 1..8")
    require(math.isfinite(args.timeout) and args.timeout > 0, "Timeout must be positive")
    artifacts = (args.artifacts or Path(tempfile.mkdtemp(prefix="mft-resume-process-", dir="/tmp"))).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    require(not any(artifacts.iterdir()), "Artifact directory must be empty; refusing to overwrite evidence")
    summary = {"status": "running", "artifacts": str(artifacts), "binary": str(binary),
               "binary_sha256": sha256(binary), "thread_environment": {name: "1" for name in THREAD_VARIABLES}, "cases": []}
    print(f"Artifacts: {artifacts}", flush=True)
    try:
        fixture = make_fixture()
        summary.update({"fixture": str(fixture), "fixture_sha256": sha256(fixture)})
        write_json(artifacts / "summary.json", summary)
        driver = Driver(binary, artifacts, fixture, args.timeout)
        for accumulation in accumulations:
            for total, cut in ((10, 4), (200, 73)):
                summary["cases"].append(driver.case(total, cut, accumulation))
                write_json(artifacts / "summary.json", summary)
        summary["child_processes"] = len(driver.invocations)
        summary["status"] = "passed"
        write_json(artifacts / "summary.json", summary)
        print(f"PASS all fresh-process cases; report: {artifacts / 'summary.json'}", flush=True)
        return 0
    except Exception as error:
        summary.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        write_json(artifacts / "summary.json", summary)
        traceback.print_exc()
        print(f"FAIL; retained artifacts: {artifacts}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
