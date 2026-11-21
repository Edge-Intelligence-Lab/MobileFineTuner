"""Behavioral Qwen session admission and unchanged-math regression (tiny fixture)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

from compare_archived_runs import compare, state
from test_resume_process import equal_tensor_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--previous-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parents[1] / "android/app/src/main/assets/tiny_qwen"
    shutil.copytree(fixture, root / "intact")
    shutil.copytree(fixture, root / "partial")
    weights = root / "partial/model/model.safetensors"
    raw = weights.read_bytes()
    size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8:8 + size])
    missing = "model.layers.0.self_attn.q_proj.weight"
    del header[missing]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:])
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "MFT_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")})
    invocations = []

    def run(binary, label, asset="intact", warm=None, stop=0, resume=None):
        if resume:
            command = [str(binary), "--resume_state", str(root / resume)]
        else:
            base = root / asset
            command = [str(binary), "--model_dir", str(base / "model"),
                       "--jsonl_train", str(base / "train.jsonl"), "--jsonl_valid", str(base / "valid.jsonl"),
                       "--run_dir", str(root / label), "--steps", "10", "--batch_size", "1",
                       "--seq_len", "16", "--lora_r", "2", "--lora_alpha", "4", "--lora_dropout", "0",
                       "--lr", "0.0002", "--grad_accum", "2", "--seed", "42", "--shuffle",
                       "--lr_scheduler", "cosine", "--warmup_steps", "2", "--eval_interval", "3",
                       "--eval_batches", "1", "--save_every", "3", "--save_initial_adapter",
                       "--checkpoint_every_updates", "1", "--suspend_after_updates", str(stop)]
            if warm:
                command += ["--resume_from", str(warm)]
        with (root / f"{label}.log").open("w") as log:
            child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        invocations.append({"label": label, "command": command, "exit_code": child.returncode})
        (root / "invocations.json").write_text(json.dumps(invocations, indent=2))
        return child.returncode

    assert run(args.previous_binary, "old_partial", "partial") == 0, "Baseline defect did not reproduce"
    assert (root / "old_partial/LATEST").exists()
    assert run(args.binary, "new_partial", "partial") != 0
    assert not (root / "new_partial/LATEST").exists()
    assert f"Missing Qwen base tensor: {missing}" in (root / "new_partial.log").read_text()
    assert run(args.previous_binary, "old_intact") == 0
    initial = next((root / "old_intact/sessions").glob("*/artifacts/lora_initial.safetensors"))
    assert run(args.binary, "new_intact", warm=initial) == 0
    assert run(args.binary, "new_split", warm=initial, stop=4) == 0
    assert state(root / "new_split")[0]["trainer"]["completed_updates"] == 4
    assert run(args.binary, "resume", resume="new_split") == 0
    # Comparison includes final state, effective loss/LR/evaluation trajectory and best.
    before_after = compare(root / "old_intact", root / "new_split")
    continuation = compare(root / "new_intact", root / "new_split")
    equal_tensor_files(state(root / "old_intact")[1], state(root / "new_intact")[1], "Admission-only change")
    result = dict(passed=True, partial_old_accepted=True, partial_new_rejected=True,
                  no_commit_for_partial=True, before_after=before_after, continuation=continuation,
                  scope="Tiny Qwen, same host/settings, not full pretrained model quality")
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
