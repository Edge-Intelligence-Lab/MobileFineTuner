"""Host regression: four fresh processes per arm; numerical, not bitwise, parity.

Run with python3 -B tests/test_elastic_microbatch.py --binary ... --model ... --output NEW_DIR
"""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import traceback

from run_real_resume import load_state
from test_resume_process import THREAD_VARIABLES, read_json, require, sha256, tensor_payloads, write_json


# Per-tensor infinity norm, relative to the baseline *update from initialization*
# for adapters, not the potentially much larger magnitude of the initial weights.
RTOL = 1e-3
ATOL = {"parameter": 1e-9, "m": 1e-15, "v": 1e-23}
METRIC_RTOL = 1e-4


def fixture(path, count, offset):
    rows = []
    for i in range(offset, offset + count):
        length = 3 + i % 4
        rows.append({"ids": [2 + (i * 3 + j * 7) % 28 for j in range(length)],
                     "mask": [0] + [int(j == 1 or (i + j) % 3 != 0) for j in range(1, length)]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def only(records):
    row, = records
    return row


def values(payload):
    return (v for v, in struct.iter_unpack("<f", payload[1]))


def compare_tensors(a, b, initial, report, check):
    require(a.keys() == b.keys(), "different tensor inventory")
    groups = {}
    for name, ref in a.items():
        group = "parameter" if name.startswith("parameter/") else name.split("/")[1]
        require(ref[0] == b[name][0], f"shape mismatch: {name}")
        peak, index = max((abs(x - y), i) for i, (x, y) in enumerate(zip(values(ref), values(b[name]))))
        signal = (max(abs(x - z) for x, z in zip(values(ref), values(initial[name])))
                  if group == "parameter" else max(abs(x) for x in values(ref)))
        limit = ATOL[group] + RTOL * signal
        entries = groups.setdefault(group, {})
        entries[name] = {"signal": signal, "max_abs_error": peak, "limit": limit, "index": index,
                         "normalized_error": peak / signal if signal else None}
        check(peak <= limit, f"{name}: error {peak:.5g} > {limit:.5g}")
    report["tensors"] = groups
    for group, atol in ATOL.items():
        check(any(e["signal"] > 100 * atol for e in groups.get(group, {}).values()),
              f"{group}: insufficient baseline signal")


def run(binary, model, out, result, warm, ragged):
    result["binary_sha256"] = sha256(binary)
    if warm:
        result["initial_adapter_sha256"] = sha256(warm)
    train = fixture(out / "train.jsonl", 27, 0)
    fixture(out / "valid.jsonl", 11, 41)
    env = dict(os.environ, **{k: "1" for k in THREAD_VARIABLES})
    base = ["--model_dir", model, "--jsonl_train", out / "train.jsonl",
            "--jsonl_valid", out / "valid.jsonl", "--steps", 4, "--seq_len", 6,
            "--batch_size", 8, "--grad_accum", 2, "--lr", 0.0002,
            "--lr_scheduler", "constant", "--warmup_steps", 0, "--lora_r", 2,
            "--lora_alpha", 4, "--lora_dropout", 0, "--seed", 42, "--no_shuffle",
            "--log_interval", 1, "--eval_interval", 1, "--eval_batches", 2,
            "--save_initial_adapter"]
    configs = {}
    cursor = total_tokens = 0

    def check(ok, message):
        if not ok:
            result["failures"].append(f"step {step}: {message}")
            print(result["failures"][-1], flush=True)

    for step in range(1, 5):
        rows = []
        for _ in range(2):
            if cursor + 8 > len(train):
                cursor = 0
            rows.extend(train[cursor:cursor + 8])
            cursor += 8
        total_tokens += sum(len(r["ids"]) for r in rows)
        pair = {}
        report = {"step": step, "arms": {}}
        result["updates"].append(report)
        for arm in ("baseline", "elastic"):
            path = out / arm
            caps = (8, 3, 5, 5) if ragged else (8, 4, 1, 1)
            physical = 8 if arm == "baseline" else caps[step - 1]
            if step == 1:
                args = [*base, "--run_dir", path]
                if arm == "elastic":
                    args += ["--resume_from", only((out / "baseline/sessions").glob("*/artifacts/lora_initial.safetensors"))]
                elif warm:
                    args += ["--resume_from", warm]
            else:
                args = ["--resume_state", path]
                if arm == "elastic" and step in (2, 3):
                    args += ["--resume_microbatch", physical]
            args += ["--checkpoint_every_updates", 1, "--suspend_after_updates", step if step < 4 else 0]
            cmd = list(map(str, [binary, *args]))
            label = f"{arm}-{step}"
            logpath = out / (label + ".log")
            rec = report["arms"][arm] = {"command": cmd, "log": str(logpath)}
            with logpath.open("x") as log, subprocess.Popen(cmd, cwd=out, env=env, stdout=log,
                                                            stderr=subprocess.STDOUT) as proc:
                rec["pid"] = proc.pid
                try:
                    rec["returncode"] = proc.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise
            require(not rec["returncode"], f"{label}: exit {rec['returncode']}, see {logpath}")
            gen, state = load_state(path)
            # Native retention prunes old payloads. Pin each committed snapshot for postmortems.
            snapshot = out / "snapshots" / label
            snapshot.mkdir(parents=True)
            for filename in ("state.json", "state.safetensors"):
                os.link(gen / filename, snapshot / filename)
            session = path / "sessions" / read_json(gen / "manifest.json")["session_id"]
            config = (path / "run_config.json").read_bytes()
            check(config == configs.setdefault(arm, config), arm + " immutable run_config changed")
            saved_args = json.loads(config)["arguments"]
            check(saved_args[saved_args.index("--batch_size") + 1] == "8" and
                  saved_args[saved_args.index("--grad_accum") + 1] == "2" and
                  "--resume_microbatch" not in saved_args, arm + " immutable arguments")
            events = lines(session / "events.jsonl")
            plans = [e for e in events if e["type"] == "batch_plan"]
            fields = ("physical_batch", "logical_batch", "physical_accumulation", "effective_batch")
            rec["plan"] = [[e.get(k) for k in fields] for e in plans]
            check(rec["plan"] == [[physical, 8, 2 * math.ceil(8 / physical), 16]], arm + " batch_plan")
            check(state["trainer"]["physical_batch"] == physical, arm + " committed physical cap")
            event = only(e for e in events if e["type"] == "update")
            evaluation = only(e for e in events if e["type"] == "evaluation")
            measures = lines(session / "measurements.jsonl")
            micros = [e for e in measures if e["event"] == "micro_step"]
            counts = [sum(sum(r["mask"][1:]) for r in logical[i:i + physical])
                      for logical in (rows[:8], rows[8:]) for i in range(0, 8, physical)]
            rec["micro_targets"] = [e["supervised_tokens"] for e in micros]
            check(rec["micro_targets"] == counts, arm + " supervised targets per physical microbatch")
            weighted_loss = sum(e["loss"] * n for e, n in zip(micros, counts)) / sum(counts)
            check(math.isclose(event["loss"], weighted_loss, rel_tol=1e-5), arm + " token-weighted update loss")
            check(event["total_tokens"] == total_tokens, arm + " cumulative input tokens")
            data = json.loads(state["dataset_state"])
            sampler = json.loads(data["train"])
            check(sampler["order"] == list(range(27)) and sampler["cursor"] == cursor and
                  json.loads(data["valid"])["cursor"] == 0, arm + " sampler order/cursor")
            check(state["status"] == ("completed" if step == 4 else "suspended") and
                  state["trainer"]["completed_updates"] == event["completed_updates"] == step, arm + " progress")
            params = state["optimizer"]["parameters"]
            check(params and all(p["initialized"] and p["step"] == step for p in params), arm + " Adam t")
            tensors = tensor_payloads(gen / "state.safetensors")
            require(set(tensors) == {prefix + p["name"] for prefix in ("parameter/", "optimizer/m/", "optimizer/v/")
                                     for p in params}, arm + " missing or extra state tensors")
            if step == 1:
                prev = read_json(path / "LATEST")["previous"]
                origin_path = path / "training_state" / prev / "state.safetensors"
                os.link(origin_path, snapshot / "initial.safetensors")
                origin = tensor_payloads(origin_path)
                if arm == "baseline":
                    initial = origin
                else:
                    require(origin == initial, "arms started from different adapter weights")
            metrics = {k: event[k] for k in ("loss", "ema_loss", "grad_norm", "lr")}
            metrics["valid_ppl"] = evaluation["valid_ppl"]
            rec["metrics"] = metrics
            check(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in metrics.values()),
                  arm + " nonfinite/nonpositive metric")
            pair[arm] = state, tensors, metrics
        a, b = pair["baseline"], pair["elastic"]
        check(a[0]["optimizer"] == b[0]["optimizer"], "Adam metadata mismatch")
        check(a[0]["dataset_state"] == b[0]["dataset_state"], "dataset mismatch")
        for k, ref in a[2].items():
            val = b[2][k]
            check(ref is not None and val is not None and math.isclose(ref, val, rel_tol=METRIC_RTOL),
                  f"{k}: {ref} vs {val}")
        compare_tensors(a[1], b[1], initial, report, check)
        write_json(out / "result.json", result)
        print(f"compared update {step}; failures={len(result['failures'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--ragged", action="store_true", help="resume at 3, then 5, then retain 5")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "updates": [], "failures": [], "tensor_rtol": RTOL,
              "tensor_atol": ATOL, "metric_rtol": METRIC_RTOL}
    try:
        run(args.binary.resolve(strict=True), args.model.resolve(strict=True), out, result,
            args.initial_adapter.resolve(strict=True) if args.initial_adapter else None, args.ragged)
    except Exception:
        result["failures"].append(traceback.format_exc())
        traceback.print_exc()
    result["status"] = "failed" if result["failures"] else "passed"
    write_json(out / "result.json", result)
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
