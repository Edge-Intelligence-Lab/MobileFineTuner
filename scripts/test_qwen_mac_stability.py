#!/usr/bin/env python3
"""Exercise the detached macOS stability controller with a tiny native model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from test_qwen_training_cli import write_model


def wait_status(output, predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = output / "status.json"
        if path.exists():
            state = json.loads(path.read_text())
            if predicate(state):
                return state
        time.sleep(0.5)
    raise RuntimeError(f"Stability controller timed out: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This integration check requires macOS")
    runner = Path(__file__).with_name("run_qwen_mac_stability.py")
    with tempfile.TemporaryDirectory(prefix="mft-stability-") as temporary:
        root = Path(temporary)
        model, data = root / "model", root / "data"
        write_model(model)
        data.mkdir()
        for split in ["train", "valid", "test"]:
            (data / f"wiki.{split}.raw").write_text("a b c d e f g " * 8 + "\n")
        common = [
            sys.executable,
            str(runner),
            "--binary",
            str(args.binary.resolve()),
            "--model-dir",
            str(model),
            "--data-dir",
            str(data),
            "--steps",
            "4",
            "--resume-steps",
            "2",
            "--stress-steps",
            "1",
            "--seq-len",
            "4",
            "--stress-seq-len",
            "8",
            "--rank",
            "2",
            "--alpha",
            "4",
            "--dropout",
            "0",
            "--background",
        ]
        output = root / "complete"
        subprocess.run(
            common + ["--output", str(output)],
            check=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            timeout=15,
        )
        try:
            state = wait_status(
                output, lambda s: s["state"] in ("complete", "stopped_or_failed")
            )
            if state["state"] != "complete":
                raise RuntimeError(state)
            assert len(state["jobs"]) == 4
            assert [j["summary"]["steps"] for j in state["jobs"]] == [3, 4, 2, 1]
            plan = json.loads((output / "plan.json").read_text())
            frozen_hash = hashlib.sha256(
                (output / "bin/train_wikitext").read_bytes()
            ).hexdigest()
            assert frozen_hash == plan["binary_sha256"]
            assert (output / "REPORT.md").is_file()
        finally:
            (output / "control.json").write_text('{"stop_requested":true}')
        # A queued run must stop on request without starting a training process.
        previous = root / "previous.json"
        previous.write_text('{"state":"running"}')
        queued = root / "queued"
        subprocess.run(
            common + ["--output", str(queued), "--after-status", str(previous)],
            check=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            timeout=15,
        )
        wait_status(queued, lambda s: s["state"] == "waiting_for_prior_run")
        (queued / "control.json").write_text('{"stop_requested":true}')
        state = wait_status(
            queued, lambda s: s["state"] == "stopped_or_failed", timeout=20
        )
        assert state["error"] == "user_stop_requested" and not state["jobs"]
    print(
        "PASS: detached queue, pinned binary, native training/resume, reports and queued stop"
    )


if __name__ == "__main__":
    main()
