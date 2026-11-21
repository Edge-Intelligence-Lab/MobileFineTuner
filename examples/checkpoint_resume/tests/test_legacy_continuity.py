"""Bounded actual-Qwen comparison of no-session versus session training.

Uses one existing executable and one immutable initial adapter for two 10-update
runs. Compares normal inference exports, not full-state checkpoint payloads.
This is a compatibility check, not a performance or full-precision loss claim.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT.parents[1] / "reviews/checkpoint_resume_impl_20261114"
STEPS = 10
CONTROL_AND_OUTPUT = {
    "--run_dir", "--resume_state", "--checkpoint_every_updates",
    "--suspend_after_updates", "--slice_seconds", "--steps", "--max_steps",
    "--stop_after_steps", "--resume_from", "--output_dir", "--measurement_dir",
    "--eval_out", "--shard_dir",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stamp(path):
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def strict_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def adapter(path):
    raw = path.read_bytes()
    require(len(raw) >= 8, "Missing safetensors header: " + str(path))
    length = struct.unpack("<Q", raw[:8])[0]
    require(0 < length <= len(raw) - 8, "Invalid header length: " + str(path))
    header = json.loads(raw[8:8 + length], object_pairs_hook=strict_pairs)
    metadata = header.pop("__metadata__")
    require(metadata == {"rank": "2", "alpha": "4", "dropout": "0", "target_mode": "qv"},
            "Unexpected Qwen adapter metadata: " + str(path))
    expected = {f"layer.{layer}.attn.{target}.lora_{side}"
                for layer in range(24) for target in ("q", "v") for side in ("A", "B")}
    require(set(header) == expected, "Not the complete 96-tensor Qwen Q/V adapter: " + str(path))
    intervals = []
    for name, tensor in header.items():
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        require(tensor["dtype"] == "F32" and len(shape) == 2 and all(d > 0 for d in shape),
                "Invalid regular adapter tensor: " + name)
        require(len(offsets) == 2 and offsets[1] - offsets[0] == math.prod(shape) * 4,
                "Invalid payload length: " + name)
        intervals.append(tuple(offsets))
    offset = 0
    for begin, end in sorted(intervals):
        require(begin == offset and end >= begin, "Overlapping or gapped adapter payload")
        offset = end
    require(8 + length + offset == len(raw), "Truncated or trailing adapter payload")
    return raw, {"path": str(path), "bytes": len(raw), "tensor_count": len(header),
                 "sha256": hashlib.sha256(raw).hexdigest(), "metadata": metadata}


def common_arguments(command, initial):
    require(isinstance(command, list) and all(isinstance(x, str) for x in command),
            "Command fixture must be a JSON string array")
    result = []
    i = 1
    while i < len(command):
        argument = command[i]
        if argument in CONTROL_AND_OUTPUT:
            require(i + 1 < len(command) and not command[i + 1].startswith("--"),
                    "Missing command fixture value: " + argument)
            i += 2
        else:
            result.append(argument)
            i += 1
    require("--fixed_batch_bin" not in result and "--fixed_batch_steps" not in result,
            "Fixed-batch fixtures are outside this test")
    if "--save_initial_adapter" not in result:
        result.append("--save_initial_adapter")
    return result + ["--steps", str(STEPS), "--resume_from", str(initial)]


def printed_trace(log):
    text = log.read_text()
    train = re.findall(
        r"^\[Train\] step (\d+)/(\d+) lr (\S+) loss (\S+) ppl (\S+) "
        r"ema_loss (\S+) grad_norm (\S+) tokens (\d+) step_time_sec", text, re.MULTILINE)
    require([int(row[0]) for row in train] == list(range(1, STEPS + 1)),
            "Missing, extra or duplicate printed updates in " + str(log))
    require(all(int(row[1]) == STEPS for row in train), "Schedule budget changed")
    require(all(math.isfinite(float(value)) for row in train for value in row[2:7]),
            "Non-finite printed training metric")
    evaluation = re.findall(
        r"^\[Eval\] step (\d+)/(\d+) valid_ppl (\S+) ema_loss (\S+) total_tokens (\d+)",
        text, re.MULTILINE)
    require([int(row[0]) for row in evaluation] == [3, 6, 9], "Unexpected evaluation cadence")
    summary = re.findall(r"^Total steps (\d+), total tokens (\d+), final EMA loss (\S+)$",
                         text, re.MULTILINE)
    require(len(summary) == 1 and int(summary[0][0]) == STEPS, "Missing final summary")
    require("training done." in text and "SUSPENDED" not in text and "[ERROR]" not in text,
            "Run did not finish normally")
    return {"train": train, "evaluation": evaluation, "summary": summary[0]}


def verify(root, initial):
    legacy = root / "legacy"
    run = root / "session"
    require(not (legacy / "LATEST").exists() and not (legacy / "sessions").exists(),
            "No-session run unexpectedly created training session state")
    sessions = list((run / "sessions").iterdir())
    require(len(sessions) == 1, "Expected exactly one fresh training session")
    artifacts = sessions[0] / "artifacts"
    latest = json.loads((run / "LATEST").read_text())
    generation = run / "training_state" / latest["current"]
    state = json.loads((generation / "state.json").read_text())
    trainer = state["trainer"]
    require(state["status"] == "completed" and trainer["completed_updates"] == STEPS and
            trainer["total_steps"] == STEPS, "Session did not commit completed update 10")
    manifest = json.loads((generation / "manifest.json").read_text())
    require(manifest["session_id"] == sessions[0].name, "Unexpected session lineage")
    events = [json.loads(line) for line in (sessions[0] / "events.jsonl").read_text().splitlines()]
    updates = [event for event in events if event.get("type") == "update" and
               event["event_seq"] <= manifest["event_seq"]]
    require([event["completed_updates"] for event in updates] == list(range(1, STEPS + 1)),
            "Unexpected authoritative committed update trace")
    source_bytes, source_info = adapter(initial)
    comparisons = []
    for name in ("lora_initial.safetensors", "lora_step3.safetensors", "lora_step6.safetensors",
                 "lora_step9.safetensors", "lora.safetensors"):
        a, ai = adapter(legacy / name)
        b, bi = adapter(artifacts / name)
        require(a == b, "Regular adapter byte mismatch: " + name)
        if name == "lora_initial.safetensors":
            require(a == source_bytes, "Warm start changed the immutable initialization")
        comparisons.append({"name": name, "byte_equal": True, "legacy": ai, "session": bi})
    best = Path(trainer["best_path"])
    require(best.is_absolute() and best.parent == artifacts and best.name.startswith("lora_best_step"),
            "Best reference is not a session-local immutable adapter")
    a, ai = adapter(legacy / "lora_best.safetensors")
    b, bi = adapter(best)
    require(a == b, "Best adapter differs")
    comparisons.append({"name": "best", "byte_equal": True, "legacy": ai, "session": bi})
    left, right = printed_trace(root / "legacy.log"), printed_trace(root / "session.log")
    require(left == right, "Printed training, evaluation or final-summary trace differs")
    require(int(left["summary"][1]) == trainer["total_tokens"] == updates[-1]["total_tokens"],
            "Committed token count differs from the legacy summary")
    for row, event in zip(right["train"], updates):
        for index, field, precision in ((2, "lr", 6), (3, "loss", 4),
                                        (5, "ema_loss", 4), (6, "grad_norm", 3)):
            require(row[index] == format(event[field], f".{precision}f"),
                    "Printed metric disagrees with authoritative event: " + field)
    for path in (legacy / "eval.jsonl", sessions[0] / "eval.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require([row["step"] for row in rows] == [3, 6, 9], "Invalid regular eval output")
        require(all(all(math.isfinite(row[key]) for key in ("valid_ppl", "ema_loss", "total_tokens"))
                    for row in rows), "Invalid regular eval metrics")
    return {"status": "passed", "updates_per_run": STEPS, "common_initial": source_info,
            "regular_adapter_comparisons": comparisons, "printed_trace_equal": True,
            "printed_trace": left, "session_completed": True,
            "legacy_has_no_session_state": True, "regular_eval_outputs_valid": True,
            "full_precision_legacy_loss_compared": False, "performance_claim": False,
            "scope": "Same existing binary: no-session versus session; no old-binary or interruption comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build-final/mft_qwen_resume")
    parser.add_argument("--command", type=Path,
                        default=DEFAULT_REPORT / "qwen_real_200/baseline_0.command.json")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT / "legacy_compat")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    binary, fixture, root = args.binary.resolve(), args.command.resolve(), args.output.resolve()
    require(args.timeout > 0, "Timeout must be positive")
    initial = args.initial
    if initial is None:
        matches = list((fixture.parent / "baseline_continuous/sessions").glob(
            "*/artifacts/lora_initial.safetensors"))
        require(len(matches) == 1, "Specify one immutable --initial adapter")
        initial = matches[0]
    initial = initial.resolve()
    adapter(initial)
    binary_before, initial_before = stamp(binary), stamp(initial)
    common = common_arguments(json.loads(fixture.read_text()), initial)
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    env.pop("MFT_NUM_THREADS", None)
    root.mkdir(parents=True, exist_ok=False)
    commands = {
        "legacy": [str(binary)] + common + ["--output_dir", str(root / "legacy")],
        "session": [str(binary)] + common + ["--run_dir", str(root / "session")],
    }
    report = {"binary_before": binary_before, "initial_before": initial_before,
              "source_command": str(fixture), "common_training_arguments": common,
              "thread_environment": {key: env.get(key) for key in
                                     ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                      "MFT_NUM_THREADS")}, "processes": {}}
    try:
        for name, command in commands.items():
            require(stamp(binary) == binary_before, "Executable changed before run: " + name)
            write_json(root / (name + ".command.json"), command)
            with (root / (name + ".log")).open("w") as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            report["processes"][name] = {"returncode": process.returncode}
            require(process.returncode == 0, name + " failed; see its log")
        require(stamp(binary) == binary_before, "Executable changed during verification")
        require(stamp(initial) == initial_before, "Immutable initialization changed")
        report.update(verify(root, initial))
        report["binary_unchanged"] = report["initial_source_unchanged"] = True
    except Exception as error:
        report.update(status="failed", error=str(error), performance_claim=False)
        write_json(root / "results.json", report)
        print(str(error), file=sys.stderr)
        return 1
    write_json(root / "results.json", report)
    print(json.dumps({"status": report["status"], "updates_per_run": STEPS,
                      "regular_adapter_pairs_equal": len(report["regular_adapter_comparisons"]),
                      "printed_trace_equal": report["printed_trace_equal"],
                      "results": str(root / "results.json")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
