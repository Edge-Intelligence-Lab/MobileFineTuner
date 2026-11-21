"""One paired host diagnostic: fork one committed update and run the same next update."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_real_resume import load_state, updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "bundle", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    bundle, root = a.bundle.resolve(), a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MFT_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")

    def run(label, args):
        cmd = [str(a.binary.resolve()), *map(str, args)]
        (root / (label + ".args.json")).write_text(json.dumps(cmd, indent=2))
        with (root / (label + ".log")).open("x") as log:
            subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1200)

    prefix = root / "prefix"
    run("prefix", ["--run_dir", prefix, "--model_dir", bundle / "model", "--jsonl_train", bundle / "train.jsonl",
                   "--jsonl_valid", bundle / "valid.jsonl", "--steps", 2, "--batch", 8, "--grad_accum", 1,
                   "--seq_len", 32, "--lora_r", 8, "--lora_alpha", 16, "--lora_dropout", 0,
                   "--lr", "0.0002", "--lr_scheduler", "constant", "--warmup_steps", 0,
                   "--eval_interval", 2, "--eval_batches", 1, "--suspend_after_updates", 1])
    source, state = load_state(prefix)
    assert state["status"] == "suspended" and state["trainer"]["completed_updates"] == 1
    samples = []
    reference = None
    for size in (8, 4, 1):
        branch = root / ("batch-" + str(size))
        shutil.copytree(prefix, branch)
        run(branch.name, ["--resume_state", branch, "--resume_microbatch", size])
        generation, result = load_state(branch)
        manifest = json.loads((generation / "manifest.json").read_text())
        assert manifest["previous"] == source.name and result["status"] == "completed"
        event = updates(branch)[-1]
        assert event["completed_updates"] == 2 and result["trainer"]["physical_batch"] == size
        exact = (result["dataset_state"], result["optimizer"], event["lr"], event["total_tokens"])
        if reference is None:
            reference = exact
        assert reference == exact
        samples.append(dict(physical_batch=size, step=2, core_step_s=event["core_step_sec"],
                            process_peak_to_update_MiB=event["process_peak_so_far_bytes"] / 2**20,
                            loss=event["loss"], valid_ppl=result["trainer"]["best_valid_ppl"]))
        (root / "comparison.json").write_text(json.dumps(dict(samples=samples, seq=32,
            shared_generation=source.name, repetitions=1, scope="host, cold processes, not a device RAM guarantee"),
            indent=2) + "\n")
        print(samples[-1], flush=True)


if __name__ == "__main__":
    main()
