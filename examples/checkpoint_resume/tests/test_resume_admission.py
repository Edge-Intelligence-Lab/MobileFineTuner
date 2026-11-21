"""Run admission checks against a completed real-model parity fixture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    run = fixture / "baseline_resumed"
    config = json.loads((run / "run_config.json").read_text())
    env = os.environ.copy()
    for key, value in config["thread_environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    latest = (run / "LATEST").read_bytes()
    results = []
    for name in ("QWEN_DEBUG", "DISABLE_QK_NORM"):
        for phase in ("start", "resume"):
            child_env = dict(env, **{name: "0"})
            if phase == "start":
                target = fixture / ("rejected_" + name.lower())
                command = config["arguments"] + ["--run_dir", str(target)]
            else:
                command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                           "tools/resume_control.py"), "resume", str(run)]
            result = subprocess.run(command, cwd=config["working_directory"], env=child_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=30)
            (fixture / (phase + "_" + name + ".log")).write_text(result.stdout)
            assert result.returncode != 0, result.stdout
            assert "Session mode rejects QWEN_DEBUG and DISABLE_QK_NORM" in result.stdout
            assert (run / "LATEST").read_bytes() == latest
            if phase == "start":
                assert not (target / "LATEST").exists()
            results.append({"phase": phase, "flag": name, "exit_code": result.returncode,
                            "rejected_before_training": True, "committed_state_unchanged": True})
    (fixture / "admission_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
