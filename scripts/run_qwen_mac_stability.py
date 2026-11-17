#!/usr/bin/env python3
"""Run serial native Qwen stability checks on macOS, with pinned binary and reports.

Uses only the Python standard library. No training or result upload service is used.
"""
import csv
import datetime
import hashlib
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
import argparse
import array
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--calibration-data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--resume-steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stress-seq-len", type=int, default=128)
    parser.add_argument("--stress-steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rss-limit-mib", type=float, default=8192)
    parser.add_argument("--swap-growth-limit-mib", type=float, default=3072)
    parser.add_argument(
        "--after-status",
        type=Path,
        help="Wait for a previous queue to complete before starting",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Detach; poll REPORT.md and status.json for results",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This resource monitor targets macOS")
    if (
        min(args.steps, args.rank, args.threads) <= 0
        or min(args.seq_len, args.stress_seq_len) < 2
        or min(args.resume_steps, args.stress_steps) < 0
    ):
        parser.error("Invalid positive step/rank/thread/sequence settings")
    if (
        not all(
            math.isfinite(v)
            for v in [
                args.alpha,
                args.dropout,
                args.lr,
                args.rss_limit_mib,
                args.swap_growth_limit_mib,
            ]
        )
        or args.alpha <= 0
        or not 0 <= args.dropout < 1
        or args.lr <= 0
        or min(args.rss_limit_mib, args.swap_growth_limit_mib) <= 0
    ):
        parser.error("Invalid numeric training/resource settings")
    return args


def expected_adapters(directory):
    config = json.loads((directory / "config.json").read_text())
    kind = config.get("model_type")
    if kind == "qwen3_5":
        config = config["text_config"]
        kind = config.get("model_type", "qwen3_5_text")
    layers = config["num_hidden_layers"]
    if kind in ("qwen2", "qwen3"):
        return layers * 14
    if kind != "qwen3_5_text":
        raise ValueError("Unsupported text architecture: " + str(kind))
    interval = config.get("full_attention_interval", 4)
    types = config.get("layer_types") or [
        "linear_attention" if (i + 1) % interval else "full_attention"
        for i in range(layers)
    ]
    if len(types) != layers or any(
        t not in ("linear_attention", "full_attention") for t in types
    ):
        raise ValueError("Invalid hybrid layer types")
    return sum(16 if t == "linear_attention" else 14 for t in types)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ARGS = parse_args()
ROOT = Path(__file__).resolve().parents[1]
OUT = ARGS.output.resolve()
BIN = ARGS.binary.resolve()
DATA = ARGS.data_dir.resolve()
EXCERPT = (ARGS.calibration_data_dir or ARGS.data_dir).resolve()
ENV = dict(
    os.environ,
    VECLIB_MAXIMUM_THREADS=str(ARGS.threads),
    OMP_NUM_THREADS=str(ARGS.threads),
    OPENBLAS_NUM_THREADS=str(ARGS.threads),
    PYTHONDONTWRITEBYTECODE="1",
    NSUnbufferedIO="YES",
)
CASES = []
for i, path in enumerate(ARGS.model_dir):
    directory = path.resolve()
    CASES.append((f"model{i+1}", str(directory), expected_adapters(directory)))
STATUS = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "runner_pid": os.getpid(),
    "state": "starting",
    "jobs": [],
}
STOP_REQUESTED = False


def on_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stop_requested():
    control = OUT / "control.json"
    return STOP_REQUESTED or (
        control.exists() and json.loads(control.read_text()).get("stop_requested")
    )


def write_report():
    lines = [
        "# Qwen Mac stability run",
        "",
        f"State: **{STATUS['state']}**",
        "",
        "This is a native CPU stability check. Pending jobs are not counted as passes.",
        "Adapter resume uses a fresh optimizer; optimizer/RNG state is not restored.",
        "",
        "| Job | Model | State | Completed/planned | Last loss | Sampled RSS MiB |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for job in STATUS["jobs"]:
        result = job.get("summary", {})
        lines.append(
            f"| {job['name']} | {Path(job['model']).name} | {job['state']} | {result.get('steps',job.get('completed_steps',0))}/{job['planned_steps']} | {result.get('loss_last',job.get('last_loss','—'))} | {result.get('rss_peak_mib_sampled',job.get('rss_mib','—'))} |"
        )
    flags = [
        j["name"]
        for j in STATUS["jobs"]
        if j.get("summary", {}).get("rss_growth_flag")
        or j.get("summary", {}).get("slowdown_flag")
    ]
    if flags:
        lines += ["", "Trend flags requiring review: " + ", ".join(flags)]
    if STATUS.get("error"):
        lines += ["", "Stop/failure reason: " + STATUS["error"]]
    lines += [
        "",
        "Checks: exact planned step counts; finite losses, all adapter gradients and parameters; finite and updated B matrices; periodic snapshots; all tensors changed after adapter resume.",
        "",
        "Trend thresholds: median RSS growth above max(512 MiB, 20%); late/early median step-time ratio above 1.5. Other applications and compilation can affect both.",
        "",
        "Resources are sampled approximately every 5 seconds. Sampled peak RSS can miss transient peaks; pmset warnings are not chip temperature measurements.",
        "",
        "See plan.json, status.json, per-job CSV/JSONL files and summary.json for configuration, commands and raw evidence. This does not establish convergence, phone support or production readiness.",
        "",
        'To stop this queue, set control.json to {"stop_requested": true}. Existing logs and snapshots are retained.',
        "",
    ]
    tmp = OUT / "REPORT.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(OUT / "REPORT.md")


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def record():
    atomic(OUT / "status.json", STATUS)
    write_report()


def readout(cmd):
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15
    )
    return p.stdout.strip()


def swap_mib():
    m = re.search(r"used\s*=\s*([\d.]+)M", readout(["sysctl", "vm.swapusage"]))
    return float(m[1]) if m else None


def adapter_check(directory, expected):
    path = directory / "adapter_model.safetensors"
    blob = path.read_bytes()
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + n])
    payload = memoryview(blob)[8 + n :]
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError("Non-FP32 adapter: " + name)
        start, end = spec["data_offsets"]
        data = bytes(payload[start:end])
        values = array.array("f")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != math.prod(spec["shape"]) or not all(
            math.isfinite(v) for v in values
        ):
            raise ValueError("Invalid adapter tensor: " + name)
        if ".lora_B." in name and not any(v != 0 for v in values):
            raise ValueError("An entire B matrix stayed zero: " + name)
        tensors[name] = data
    b = [k for k in tensors if ".lora_B." in k]
    if len(tensors) != expected or len(b) != expected // 2:
        raise ValueError("Adapter tensor count mismatch")
    result = {
        "tensors": len(tensors),
        "nonzero_B": len(b),
        "finite": True,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    atomic(directory / "adapter_check.json", result)
    return tensors, result


def metrics(directory):
    p = directory / "step_metrics.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return [r for r in csv.DictReader(f) if r.get("elapsed_s")]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inspect_completed(job):
    directory = OUT / job["name"]
    rows = metrics(directory)
    require(
        (len(rows) == job["planned_steps"]),
        f"Incomplete run: {len(rows)}/{job['planned_steps']}",
    )
    require(
        (all(int(r["step"]) == i + 1 for i, r in enumerate(rows))),
        "Stability validation failed",
    )
    with (directory / "stability_metrics.csv").open() as f:
        stability = list(csv.DictReader(f))
    require((len(stability) == len(rows)), "Missing stability observations")
    require(
        (all(math.isfinite(float(v)) for r in rows for v in r.values())),
        "Stability validation failed",
    )
    require(
        (all(math.isfinite(float(v)) for r in stability for v in r.values())),
        "Stability validation failed",
    )
    require(
        (all(float(r["grad_l2_before_clip"]) > 0 for r in stability)),
        "All-zero gradient at a training step",
    )
    require(
        all(float(r["grad_l2_after_clip"]) <= 1.00001 for r in stability),
        "Gradient clipping exceeded its configured norm bound",
    )
    arrays, adapter = adapter_check(directory, job["expected_tensors"])
    if job.get("resume_from"):
        previous, _ = adapter_check(OUT / job["resume_from"], job["expected_tensors"])
        changed = sum(arrays[k] != previous[k] for k in arrays)
        require(
            (changed == len(arrays)),
            f"Only {changed}/{len(arrays)} tensors changed on resumed training",
        )
        adapter["changed_tensors_on_resume"] = changed
    samples = [
        json.loads(s) for s in (directory / "resources.jsonl").read_text().splitlines()
    ]
    rss = [s["rss_mib"] for s in samples if s.get("rss_mib") is not None]
    losses = [float(r["loss"]) for r in rows]
    times = [float(r["step_time_s"]) for r in rows]
    import statistics

    result = {
        "steps": len(rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_first50_mean": statistics.mean(losses[:50]),
        "loss_last50_mean": statistics.mean(losses[-50:]),
        "step_seconds_median": statistics.median(times),
        "rss_peak_mib_sampled": max(rss) if rss else None,
        "grad_l2_preclip_max": max(float(r["grad_l2_before_clip"]) for r in stability),
        "grad_l2_postclip_max": max(float(r["grad_l2_after_clip"]) for r in stability),
        "adapter": adapter,
    }
    if len(rows) >= 100:
        early = [
            s["rss_mib"]
            for s in samples
            if 30 <= s.get("step", 0) <= 80 and s.get("rss_mib") is not None
        ]
        late = [
            s["rss_mib"]
            for s in samples
            if s.get("step", 0) >= len(rows) - 50 and s.get("rss_mib") is not None
        ]
        if early and late:
            growth = statistics.median(late) - statistics.median(early)
            result.update(
                rss_early_median_mib=statistics.median(early),
                rss_late_median_mib=statistics.median(late),
                rss_growth_mib=growth,
                rss_growth_flag=growth > max(512, 0.2 * statistics.median(early)),
            )
        first = statistics.median(times[20:70])
        last = statistics.median(times[-50:])
        result.update(
            step_seconds_early_median=first,
            step_seconds_late_median=last,
            slowdown_ratio=last / first,
            slowdown_flag=last / first > 1.5,
        )
        require(
            ((directory / "checkpoint-100/adapter_model.safetensors").exists()),
            "Stability validation failed",
        )
        # Final periodic snapshot must match the final export exactly.
        if job["planned_steps"] % 100 == 0:
            require(
                (
                    sha256_file(
                        directory
                        / f"checkpoint-{job['planned_steps']}"
                        / "adapter_model.safetensors"
                    )
                    == adapter["sha256"]
                ),
                "Stability validation failed",
            )
    atomic(directory / "summary.json", result)
    return result


def run_job(name, model, steps, expected, *, seq=None, calibration=False, resume=None):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    if seq is None:
        seq = ARGS.seq_len
    if stop_requested():
        raise RuntimeError("user_stop_requested")
    cmd = [
        str(BIN),
        "--model_dir",
        str(model),
        "--data_dir",
        str(EXCERPT if calibration else DATA),
        "--seq_len",
        str(seq),
        "--batch_size",
        "1",
        "--max_steps",
        str(steps),
        "--lora_r",
        str(ARGS.rank),
        "--lora_alpha",
        str(ARGS.alpha),
        "--lora_dropout",
        str(ARGS.dropout),
        "--lora_targets",
        "all-linear",
        "--lr",
        str(ARGS.lr),
        "--max_grad_norm",
        "1",
        "--base_weight_storage",
        "auto",
        "--stability_checks",
        "--save_every",
        "2" if calibration else "100",
        "--output_dir",
        str(directory),
    ]
    if resume:
        cmd += ["--resume_from", str(OUT / resume), "--seed", "43"]
    job = {
        "name": name,
        "model": model,
        "planned_steps": steps,
        "seq_len": seq,
        "expected_tensors": expected,
        "resume_from": resume,
        "command": cmd,
        "state": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    STATUS.update(state="running", current=name)
    STATUS["jobs"].append(job)
    baseline = swap_mib()
    start = time.monotonic()
    last_progress = start
    last_step = 0
    critical_count = 0
    swap_count = 0
    reason = None
    with (directory / "train.log").open("w") as log, (
        directory / "resources.jsonl"
    ).open("w") as resources:
        log.write("COMMAND " + json.dumps(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=ENV,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        job.update(pid=p.pid, baseline_swap_mib=baseline)
        record()
        print(name, "started", p.pid, flush=True)
        iteration = 0
        last_system = {}
        while p.poll() is None:
            time.sleep(5)
            if p.poll() is not None:
                break
            now = time.monotonic()
            rows = metrics(directory)
            step = len(rows)
            if step != last_step:
                last_step = step
                last_progress = now
            line = readout(["ps", "-o", "rss=,pcpu=", "-p", str(p.pid)]).split()
            rss = (
                float(line[0]) / 1024 if len(line) >= 2 and line[0].isdigit() else None
            )
            level = readout(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
            if iteration % 6 == 0:
                last_system = {
                    "pressure_level": int(level) if level.isdigit() else None,
                    "swap_used_mib": swap_mib(),
                    "thermal": readout(["pmset", "-g", "therm"]),
                }
            last_system["pressure_level"] = int(level) if level.isdigit() else None
            sample = {
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "elapsed_s": round(now - start, 3),
                "step": step,
                "rss_mib": rss,
                "cpu_percent": float(line[1])
                if len(line) >= 2 and line[0].isdigit()
                else None,
                **last_system,
            }
            resources.write(json.dumps(sample) + "\n")
            resources.flush()
            job.update(
                completed_steps=step,
                rss_mib=rss,
                elapsed_s=sample["elapsed_s"],
                pressure_level=sample.get("pressure_level"),
                last_loss=float(rows[-1]["loss"]) if rows else None,
            )
            critical_count = (
                critical_count + 1 if (sample.get("pressure_level") or 0) >= 4 else 0
            )
            swap_growth = (sample.get("swap_used_mib") or 0) - (baseline or 0)
            swap_count = (
                swap_count + 1
                if swap_growth > ARGS.swap_growth_limit_mib
                and (sample.get("pressure_level") or 0) >= 2
                else 0
            )
            if rss is not None and rss > ARGS.rss_limit_mib:
                reason = "resource_guard_process_rss_limit"
            if critical_count >= 3:
                reason = "resource_guard_sustained_critical_memory_pressure"
            if swap_count >= 6:
                reason = "resource_guard_swap_growth_limit"
            if now - last_progress > 1200:
                reason = "watchdog_no_completed_step_for_20min"
            if now - start > 10800:
                reason = "watchdog_job_over_3hours"
            if stop_requested():
                reason = "user_stop_requested"
            record()
            iteration += 1
            if reason:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                break
        job.update(
            exit_code=p.wait(),
            elapsed_s=round(time.monotonic() - start, 3),
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    if reason:
        job.update(state="stopped", reason=reason)
        record()
        raise RuntimeError(name + ": " + reason)
    if job["exit_code"] != 0:
        job.update(state="failed")
        record()
        raise RuntimeError(name + " native process failed; inspect train.log")
    try:
        job["summary"] = inspect_completed(job)
    except Exception as error:
        job.update(state="failed_validation", reason=str(error))
        record()
        raise
    job["state"] = "passed"
    record()
    print(name, "passed", job["elapsed_s"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        raise SystemExit("Existing run status found; choose a new output directory")
    if ARGS.background:
        arguments = [
            argument for argument in sys.argv[1:] if argument != "--background"
        ]
        with (OUT / "runner.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(f"Runner PID {child.pid}; status/report: {OUT}")
        sys.exit(0)
    import fcntl

    lock = (OUT / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    os.nice(5)
    frozen = OUT / "bin"
    frozen.mkdir()
    shutil.copy2(BIN, frozen / "train_wikitext")
    BIN = frozen / "train_wikitext"
    plan = {
        "models": CASES,
        "steps": ARGS.steps,
        "resume_steps": ARGS.resume_steps,
        "seq_len": ARGS.seq_len,
        "stress_seq_len": ARGS.stress_seq_len,
        "stress_steps": ARGS.stress_steps,
        "learning_rate": ARGS.lr,
        "rank": ARGS.rank,
        "alpha": ARGS.alpha,
        "dropout": ARGS.dropout,
        "threads_cap": ARGS.threads,
        "dataset": str(DATA),
        "calibration_dataset": str(EXCERPT),
        "binary_sha256": sha256_file(BIN),
        "dataset_sha256": sha256_file(DATA / "wiki.train.raw"),
        "rss_limit_mib": ARGS.rss_limit_mib,
        "swap_growth_limit_mib": ARGS.swap_growth_limit_mib,
        "after_status": str(ARGS.after_status) if ARGS.after_status else None,
    }
    atomic(OUT / "plan.json", plan)
    record()
    caffeine = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if ARGS.after_status:
            STATUS["state"] = "waiting_for_prior_run"
            record()
            while True:
                if stop_requested():
                    raise RuntimeError("user_stop_requested")
                prior = json.loads(ARGS.after_status.read_text())
                if prior["state"] == "complete":
                    break
                if prior["state"] not in (
                    "running",
                    "starting",
                    "waiting_for_prior_run",
                ):
                    raise RuntimeError(
                        "Prior queue failed/stopped; inspect it before retrying"
                    )
                time.sleep(5)
        for prefix, model, count in CASES:
            run_job(prefix + "_calibration", model, 3, count, calibration=True)
            run_job(prefix + "_long", model, ARGS.steps, count)
            if ARGS.resume_steps:
                run_job(
                    prefix + "_resume",
                    model,
                    ARGS.resume_steps,
                    count,
                    resume=prefix + "_long",
                )
        if ARGS.stress_steps:
            prefix, model, count = CASES[0]
            run_job(
                prefix + "_stress",
                model,
                ARGS.stress_steps,
                count,
                seq=ARGS.stress_seq_len,
                resume=prefix + "_resume" if ARGS.resume_steps else prefix + "_long",
            )
        STATUS.update(
            state="complete",
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        record()
    except Exception as error:
        STATUS.update(
            state="stopped_or_failed",
            error=str(error),
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        record()
        raise
    finally:
        caffeine.terminate()
        caffeine.wait(timeout=15)
