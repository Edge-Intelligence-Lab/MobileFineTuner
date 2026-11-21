"""Fresh-process resume parity on local pretrained models, not a quality benchmark."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def load_state(run):
    latest = json.loads((run / "LATEST").read_text())
    generation = run / "training_state" / latest["current"]
    return generation, json.loads((generation / "state.json").read_text())


def updates(run):
    # Walk only the selected committed branch, never orphan directories.
    latest = json.loads((run / "LATEST").read_text())["current"]
    cutoffs = {}
    while latest:
        m = json.loads((run / "training_state" / latest / "manifest.json").read_text())
        cutoffs[m["session_id"]] = max(cutoffs.get(m["session_id"], 0), m["event_seq"])
        latest = m["previous"]
    rows = []
    for session, cutoff in cutoffs.items():
        for line in (run / "sessions" / session / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["event_seq"] <= cutoff and e["type"] == "update":
                rows.append(e)
    return sorted(rows, key=lambda e: e["completed_updates"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--piqa-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=["baseline", "combined"])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--stop", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--path-launch", action="store_true", help="Start via PATH and resume via the controller")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    with args.piqa_csv.open() as f:
        source = list(csv.DictReader(f))[:40]
    rows = []
    for row in source:
        text = row["question"] + "\nA: " + row["A"] + "\nB: " + row["B"] + "\nAnswer: " + row["answer"]
        ids = tok.encode(text).ids[:args.seq]
        rows.append({"ids": ids, "mask": [1] * len(ids)})
    for name, subset in [("train", rows[:32]), ("valid", rows[32:])]:
        (root / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in subset))
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    env.pop("MFT_NUM_THREADS", None)
    if args.path_launch:
        env["PATH"] = str(args.binary.resolve().parent) + os.pathsep + env.get("PATH", "")
    results = []
    modes = {"baseline": [], "accumulation": ["--grad_accum", "2"],
             "checkpointing": ["--activation_checkpointing"],
             "streaming": ["--memory_efficient_attention"],
             "sharding": ["--shard_enable", "--shard_budget_mb", "768", "--shard_fp16_disk", "false"]}
    modes["combined"] = sum((modes[x] for x in ["accumulation", "checkpointing", "streaming", "sharding"]), [])
    for mode in args.modes:
        base = [str(args.binary.resolve()), "--model_dir", str(args.model.resolve()),
                "--jsonl_train", str(root / "train.jsonl"), "--jsonl_valid", str(root / "valid.jsonl"),
                "--steps", str(args.steps), "--seq_len", str(args.seq), "--batch_size", str(args.batch),
                "--lora_r", str(args.rank), "--lora_alpha", str(2 * args.rank), "--lora_dropout", "0",
                "--lr", "0.0002", "--lr_scheduler", "cosine", "--warmup_steps", "2",
                "--log_interval", "1", "--eval_interval", "3", "--eval_batches", "1",
                "--save_every", "3", "--seed", "42", "--shuffle", "--save_initial_adapter"] + modes[mode]
        continuous, resumed = root / (mode + "_continuous"), root / (mode + "_resumed")
        commands = [base + ["--run_dir", str(continuous)],
                    base + ["--run_dir", str(resumed), "--suspend_after_updates", str(args.stop)],
                    [str(args.binary.resolve()), "--resume_state", str(resumed)]]
        if args.path_launch:
            for command in commands[:2]:
                command[0] = args.binary.name
            commands[2] = [sys.executable, str(Path(__file__).resolve().parents[1] /
                           "tools/resume_control.py"), "resume", str(resumed)]
        for i, cmd in enumerate(commands):
            if i == 1:
                initial = list((continuous / "sessions").glob("*/artifacts/lora_initial.safetensors"))
                assert len(initial) == 1
                cmd += ["--resume_from", str(initial[0])]
            (root / f"{mode}_{i}.command.json").write_text(json.dumps(cmd, indent=2))
            start = time.monotonic()
            with (root / f"{mode}_{i}.log").open("w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3600)
            if proc.returncode:
                raise RuntimeError(f"Training failed ({proc.returncode}): {mode}_{i}.log")
            print(f"{mode} process {i} finished in {time.monotonic()-start:.2f}s", flush=True)
            if i == 1:
                _, paused = load_state(resumed)
                assert paused["status"] == "suspended" and paused["trainer"]["completed_updates"] == args.stop
                if args.path_launch:
                    config = json.loads((resumed / "run_config.json").read_text())
                    assert config["executable"]["path"] == str(args.binary.resolve())
                    assert config["arguments"][0] == str(args.binary.resolve())
            # These are disposable frozen-backbone caches, not committed state.
            # Removing them also exercises cold resume without an old shard cache.
            for cache in (continuous if i == 0 else resumed).glob("sessions/*/artifacts/shards"):
                shutil.rmtree(cache)
        ga, sa = load_state(continuous)
        gb, sb = load_state(resumed)
        exact_tensors = (ga / "state.safetensors").read_bytes() == (gb / "state.safetensors").read_bytes()
        exact_adam = sa["optimizer"] == sb["optimizer"]
        exact_data = sa["dataset_state"] == sb["dataset_state"]
        ua, ub = updates(continuous), updates(resumed)
        assert [r["completed_updates"] for r in ua] == list(range(1, args.steps + 1))
        assert [r["completed_updates"] for r in ub] == list(range(1, args.steps + 1))
        checked = ["loss", "lr", "grad_norm", "total_tokens", "ema_loss"]
        assert all(k in ua[0] and k in ub[0] for k in checked)
        exact_trace = all(all(x.get(k) == y.get(k) for k in checked) for x, y in zip(ua, ub))
        result = dict(mode=mode, updates=args.steps, split=args.stop, batch=args.batch, seq=args.seq,
                      rank=args.rank, path_launch=args.path_launch, tensor_bytes_equal=exact_tensors,
                      optimizer_metadata_equal=exact_adam, dataset_state_equal=exact_data,
                      update_trace_equal=exact_trace, trace_fields=checked,
                      continuous_status=sa["status"], resumed_status=sb["status"])
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2))
        if not all([exact_tensors, exact_adam, exact_data, exact_trace]):
            raise AssertionError(result)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
