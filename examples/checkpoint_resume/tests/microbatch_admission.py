"""Exercise native rejection and checkpoint fallback on a disposable Qwen run."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from run_real_resume import load_state


def check(binary, root):
    fixture = Path(__file__).resolve().parents[1] / "android/app/src/main/assets/tiny_qwen"
    target = root / "run"
    fresh = ["--model_dir", fixture / "model", "--jsonl_train", fixture / "train.jsonl",
             "--jsonl_valid", fixture / "valid.jsonl", "--batch", 8, "--steps", 4, "--seq_len", 12,
             "--lora_dropout", 0, "--lora_r", 2, "--run_dir", target, "--suspend_after_updates", 1]
    resume = ["--resume_state", target]

    def invoke(args, ok):
        p = subprocess.run([str(binary), *map(str, args)], cwd=root, text=True, capture_output=True, timeout=90)
        assert (p.returncode == 0) == ok, (args, p.stdout, p.stderr)
        assert p.returncode >= 0, ("trainer terminated by signal", p.returncode)
        return p.stdout + p.stderr

    invoke([*fresh, "--resume_microbatch", 4], False)
    assert not target.exists()
    undersized = list(fresh)
    undersized[undersized.index("--batch") + 1] = 32
    undersized[undersized.index("--run_dir") + 1] = root / "undersized"
    assert "smaller than the original logical batch" in invoke(undersized, False)
    assert not (root / "undersized" / "LATEST").exists()
    invoke(fresh, True)
    latest = target / "LATEST"
    pause = target / "SUSPEND"
    original = latest.read_bytes()
    sessions = set((target / "sessions").iterdir())
    pause.touch()
    for flags in (["--resume_microbatch", 0], ["--resume_microbatch", -1],
                  ["--resume_microbatch", 9], ["--resume_microbatch", 16],
                  ["--resume_microbatch", "1.5"], ["--resume_microbatch", 1, "--resume_microbatch", 4],
                  ["--batch", 1]):
        invoke([*resume, *flags], False)
        assert latest.read_bytes() == original and pause.exists()
        assert set((target / "sessions").iterdir()) == sessions
    invoke([*resume, "--resume_microbatch", 3, "--suspend_after_updates", 2], True)
    invoke([*resume, "--resume_microbatch", 1, "--suspend_after_updates", 3], True)
    gen, state = load_state(target)
    assert state["trainer"]["physical_batch"] == 1
    # Same-length corruption passes the existing size check, testing semantic
    # validation and restore of the previous generation's plan, not a CLI default.
    file = gen / "state.json"
    raw = file.read_bytes()
    assert raw.count(b'"physical_batch": 1') == 1
    file.write_bytes(raw.replace(b'"physical_batch": 1', b'"physical_batch": 9'))
    invoke(resume, True)
    gen, state = load_state(target)
    assert state["trainer"]["physical_batch"] == 3 and state["status"] == "completed"
    session = json.loads((gen / "manifest.json").read_text())["session_id"]
    events = [json.loads(line) for line in (target / "sessions" / session / "events.jsonl").read_text().splitlines()]
    assert any(e["type"] == "resume" and e["fallback"] and e["completed_updates"] == 2 for e in events)
    original = latest.read_bytes()
    invoke([*resume, "--resume_microbatch", 1], False)
    assert latest.read_bytes() == original


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        check(a.binary.resolve(), Path(tmp))
    print("PASS: fresh/invalid requests rejected, pause marker preserved, semantic fallback replays at 3, completed override rejected")
