"""Exercise real CLI pause requests/signals and cold continuation against a reference."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_real_resume import load_state, updates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True, help="run_real_resume output directory")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = args.reference.resolve() / "baseline_continuous"
    ga, sa = load_state(reference)
    command = json.loads((args.reference / "baseline_0.command.json").read_text())
    index = command.index("--run_dir")
    command = command[:index]
    initial = next(reference.glob("sessions/*/artifacts/lora_initial.safetensors"))
    command += ["--resume_from", str(initial)]
    controller = Path(__file__).resolve().parents[1] / "tools" / "resume_control.py"
    config = json.loads((reference / "run_config.json").read_text())
    env = dict(os.environ)
    for key, value in config["thread_environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    results = []
    for mode in ["request_file", "sigterm", "sigkill", "time_slice"]:
        run = root / mode
        with (root / (mode + "_start.log")).open("w") as log:
            controls = ["--slice_seconds", "0.001"] if mode == "time_slice" else []
            child = subprocess.Popen(command + ["--run_dir", str(run)] + controls, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=config["working_directory"])
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if mode == "time_slice":
                        break
                    if child.poll() is not None:
                        raise RuntimeError("Child ended before pause request")
                    if (run / "LATEST").exists():
                        _, state = load_state(run)
                        if state["trainer"]["completed_updates"] >= 5:
                            break
                    time.sleep(0.01)
                else:
                    raise RuntimeError("No checkpoint progress before timeout")
                if mode == "request_file":
                    contender = subprocess.run([sys.executable, str(controller), "resume", str(run)],
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                    (root / "lock_contention.log").write_bytes(contender.stdout)
                    assert contender.returncode != 0 and b"already active" in contender.stdout
                    subprocess.run([sys.executable, str(controller), "suspend", str(run)], check=True)
                elif mode != "time_slice":
                    child.send_signal(signal.SIGTERM if mode == "sigterm" else signal.SIGKILL)
                code = child.wait(timeout=120)
                assert code == (-signal.SIGKILL if mode == "sigkill" else 0)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        _, stopped = load_state(run)
        if mode == "time_slice":
            assert stopped["trainer"]["completed_updates"] == 0
        else:
            assert 5 <= stopped["trainer"]["completed_updates"] < sa["trainer"]["completed_updates"]
        if mode != "sigkill":
            assert stopped["status"] == "suspended"
            assert "SUSPENDED completed_updates=" in (root / (mode + "_start.log")).read_text()
        with (root / (mode + "_resume.log")).open("w") as log:
            subprocess.run([sys.executable, str(controller), "resume", str(run)], check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        gb, sb = load_state(run)
        assert sb["status"] == "completed"
        assert sa["optimizer"] == sb["optimizer"] and sa["dataset_state"] == sb["dataset_state"]
        assert (ga / "state.safetensors").read_bytes() == (gb / "state.safetensors").read_bytes()
        ua, ub = updates(reference), updates(run)
        assert len(ua) == len(ub) == sa["trainer"]["completed_updates"]
        for a, b in zip(ua, ub):
            assert all(a[key] == b[key] for key in ["loss", "lr", "grad_norm", "total_tokens", "ema_loss"])
        with (root / (mode + "_report.json")).open("w") as report:
            subprocess.run([sys.executable, str(controller), "report", str(run)], stdout=report, check=True)
        results.append({"mode": mode, "stopped_at": stopped["trainer"]["completed_updates"],
                        "final_updates": len(ub), "exact_parity": True})
        (root / "results.json").write_text(json.dumps(results, indent=2))
        print(results[-1], flush=True)
    # A completed run can be reopened without replaying an optimizer update.
    with (root / "completed_resume.log").open("w") as log:
        subprocess.run([sys.executable, str(controller), "resume", str(run)], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=120)
    gc, sc = load_state(run)
    assert sc["trainer"]["completed_updates"] == sa["trainer"]["completed_updates"]
    assert (gc / "state.safetensors").read_bytes() == (ga / "state.safetensors").read_bytes()
    assert len(updates(run)) == len(updates(reference))


if __name__ == "__main__":
    main()
