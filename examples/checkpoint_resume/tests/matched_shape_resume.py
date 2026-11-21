"""Compare one extra cold restart, holding the checkpoint and all batch shapes fixed."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

from run_real_resume import load_state
from test_elastic_microbatch import fixture, lines, only
from test_resume_process import (THREAD_VARIABLES, read_json, require, sha256,
                                 tensor_payloads, write_json)


def experiment(binary, model, root):
    root.mkdir(parents=True, exist_ok=False)
    fixture(root / "train.jsonl", 27, 0)
    fixture(root / "valid.jsonl", 11, 41)
    threads = dict.fromkeys(THREAD_VARIABLES, "1")
    report = {"binary_sha256": sha256(binary), "threads": threads, "processes": {}}
    trace = {}

    def launch(label, run, args, steps, cap, status, restore=None):
        command = list(map(str, [binary, *args, "--checkpoint_every_updates", 1]))
        with (root / (label + ".log")).open("x") as log:
            with subprocess.Popen(command, cwd=root, env={**os.environ, **threads},
                                  stdout=log, stderr=subprocess.STDOUT) as child:
                try:
                    require(child.wait(timeout=900) == 0, f"Failed {label}; see child log")
                finally:
                    if child.poll() is None:
                        child.kill()
        gen, state = load_state(run)
        require(state["status"] == status and state["trainer"]["completed_updates"] == steps[-1]
                and state["trainer"]["physical_batch"] == cap, f"{label}: unexpected commit")
        session = read_json(gen / "manifest.json")["session_id"]
        events = lines(run / "sessions" / session / "events.jsonl")
        if restore:
            resumed = only(e for e in events if e["type"] == "resume")
            require(resumed["generation"] == restore and not resumed["fallback"], "Wrong resume point")
        plan = only(e for e in events if e["type"] == "batch_plan")
        require([plan[k] for k in ("logical_batch", "physical_batch", "logical_accumulation",
                                   "physical_accumulation", "effective_batch")] ==
                [8, cap, 2, 2 * math.ceil(8 / cap), 16], "Wrong batch plan")
        rows, evals = ([e for e in events if e["type"] == kind] for kind in ("update", "evaluation"))
        for stream in (rows, evals):
            require([e["completed_updates"] for e in stream] == steps, "Unexpected update coverage")
        for row, evaluation in zip(rows, evals):
            metrics = {k: row[k] for k in ("loss", "ema_loss", "grad_norm", "lr", "total_tokens")}
            metrics["valid_ppl"] = evaluation["valid_ppl"]
            require(all(math.isfinite(v) for v in metrics.values()), "Nonfinite metric")
            trace.setdefault(run.name, {})[row["completed_updates"]] = metrics
        report["processes"][label] = dict(command=command, pid=child.pid, session=session,
                                          restored=restore, generation=gen.name, steps=steps, cap=cap)
        return gen, state

    prefix, direct, extra = (root / name for name in ("prefix", "direct", "extra"))
    origin, _ = launch("prefix", prefix, [
        "--run_dir", prefix, "--model_dir", model, "--jsonl_train", root / "train.jsonl",
        "--jsonl_valid", root / "valid.jsonl", "--batch", 8, "--grad_accum", 2,
        "--steps", 4, "--suspend_after_updates", 1, "--seq_len", 6, "--lora_r", 2,
        "--lora_alpha", 4, "--lora_dropout", 0, "--lr", 0.0002, "--lr_scheduler", "constant",
        "--warmup_steps", 0, "--no_shuffle", "--seed", 42, "--eval_interval", 1,
        "--eval_batches", 2, "--log_interval", 1], [1], 8, "suspended")
    report["shared_checkpoint"] = {f: sha256(origin / f) for f in ("state.json", "state.safetensors")}
    for arm in (direct, extra):
        shutil.copytree(prefix, arm)
        gen, _ = load_state(arm)
        require(gen.name == origin.name and all(sha256(gen / f) == digest for f, digest
                in report["shared_checkpoint"].items()), "Checkpoint copy differs")

    ga, a = launch("direct-2-4", direct, ["--resume_state", direct, "--resume_microbatch", 3],
                   [2, 3, 4], 3, "completed", origin.name)
    mid, _ = launch("extra-2", extra, ["--resume_state", extra, "--resume_microbatch", 3,
                    "--suspend_after_updates", 2], [2], 3, "suspended", origin.name)
    gb, b = launch("extra-3-4", extra, ["--resume_state", extra], [3, 4], 3, "completed", mid.name)
    paths = [g / "state.safetensors" for g in (ga, gb)]
    left, right = map(tensor_payloads, paths)
    params = a["optimizer"]["parameters"]
    expected = {prefix + p["name"] for p in params for prefix in ("parameter/", "optimizer/m/", "optimizer/v/")}
    checks = {
        "full_adam_inventory": bool(params) and left.keys() == right.keys() == expected
                               and all(p["initialized"] and p["step"] == 4 for p in params),
        "weights_and_moments_exact": left == right,
        "full_tensor_file_exact": paths[0].read_bytes() == paths[1].read_bytes(),
        "optimizer_metadata_exact": a["optimizer"] == b["optimizer"],
        "dataset_state_exact": a["dataset_state"] == b["dataset_state"],
        "update_and_eval_metrics_exact": trace["direct"] == trace["extra"],
    }
    report.update(checks=checks, passed=all(checks.values()), trace=trace,
                  final_tensor_sha256=list(map(sha256, paths)), tensors=len(left),
                  adapter_tensors=len(params))
    write_json(root / "comparison.json", report)
    require(report["passed"], str(checks))
    print(f"PASS: {len(left)} tensors and all three cap-3 updates exact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    experiment(args.binary.resolve(strict=True), args.model.resolve(strict=True), args.output.resolve())
