"""50-update native Android demo, using the packaged (random, tiny) Qwen fixture."""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import struct
import subprocess
import uuid

from compare_archived_runs import build_report, state, tensor_payloads
from run_real_resume import updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined", action="store_true",
                        help="also use native sharding, activation checkpointing and streaming attention")
    a = parser.parse_args()
    if not re.fullmatch(r"emulator-\d+", a.serial):
        parser.error("use an Android emulator serial")
    root = Path(__file__).resolve().parents[1]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    remote = "/data/local/tmp/mft_elastic_" + uuid.uuid4().hex

    def adb(*args):
        return subprocess.check_output([a.adb, "-s", a.serial, *args], text=True,
                                       stderr=subprocess.STDOUT, timeout=360)

    def shell(*args):
        return adb("shell", shlex.join([str(x) for x in args]))

    shell("mkdir", remote)
    adb("push", str(root / "android/app/src/main/assets/tiny_qwen"), remote + "/fixture")
    adb("push", str(root / "android/app/src/main/jniLibs/arm64-v8a/libmft_qwen.so"), remote + "/mft")
    shell("chmod", "755", remote + "/mft")
    completed = False
    try:
        def train(name, argv):
            cmd = ["env", "MFT_NUM_THREADS=1", remote + "/mft", *argv]
            (out / (name + ".command.json")).write_text(json.dumps(cmd, indent=2))
            try:
                log = shell(*cmd)
            except subprocess.CalledProcessError as exc:
                (out / (name + ".log")).write_text(exc.output)
                raise
            (out / (name + ".log")).write_text(log)

        fixture = remote + "/fixture/"
        base = ["--model_dir", fixture + "model", "--jsonl_train", fixture + "train.jsonl",
                "--jsonl_valid", fixture + "valid.jsonl", "--steps", "50", "--batch", "8",
                "--seq_len", "16", "--lora_r", "2", "--lora_alpha", "4", "--lora_dropout", "0",
                "--seed", "42", "--shuffle", "--lr", "0.0002", "--grad_accum", "1",
                "--eval_interval", "5", "--eval_batches", "1", "--save_initial_adapter"]
        if a.combined:
            base += ["--activation_checkpointing", "--memory_efficient_attention", "--shard_enable",
                     "--shard_budget_mb", "8", "--shard_fp16_disk", "false"]
        ref, elastic = remote + "/reference", remote + "/elastic"
        train("reference", [*base, "--run_dir", ref])
        adb("pull", ref, str(out / "reference"))
        initial, = (out / "reference/sessions").glob("*/artifacts/lora_initial.safetensors")
        adb("push", str(initial), remote + "/initial.safetensors")
        train("first20", [*base, "--run_dir", elastic, "--resume_from", remote + "/initial.safetensors",
                          "--suspend_after_updates", "20"])
        train("next15", ["--resume_state", elastic, "--resume_microbatch", "4", "--suspend_after_updates", "35"])
        train("last15", ["--resume_state", elastic, "--resume_microbatch", "1"])
        adb("pull", elastic, str(out / "elastic"))
        s, f = state(out / "reference")
        t, g = state(out / "elastic")
        assert s["status"] == t["status"] == "completed"
        assert s["optimizer"] == t["optimizer"] and s["dataset_state"] == t["dataset_state"]
        assert s["trainer"]["physical_batch"] == 8 and t["trainer"]["physical_batch"] == 1
        ref_rows, split_rows = updates(out / "reference"), updates(out / "elastic")
        assert len(ref_rows) == len(split_rows) == 50
        for left, right in zip(ref_rows, split_rows):
            assert all(left[k] == right[k] for k in ("completed_updates", "lr", "total_tokens"))
            assert math.isfinite(left["loss"]) and math.isfinite(right["loss"])
        delta = max(abs(left["loss"] - right["loss"]) for left, right in zip(ref_rows, split_rows))
        reports = [build_report(out / arm) for arm in ("reference", "elastic")]
        assert all(r["valid"] for r in reports), [r["errors"] for r in reports]
        evals = [r["effective_evaluations"] for r in reports]
        assert all([e["completed_updates"] for e in rows] == list(range(5, 51, 5)) for rows in evals)
        assert all(math.isfinite(e["valid_ppl"]) and e["valid_ppl"] > 0 for rows in evals for e in rows)
        ppl_relative = max(abs(x["valid_ppl"] / y["valid_ppl"] - 1) for x, y in zip(*evals))
        af, bf = tensor_payloads(f), tensor_payloads(g)
        assert af.keys() == bf.keys()
        max_abs = {}
        for name, (shape, raw) in af.items():
            assert shape == bf[name][0]
            diff = max(abs(x[0] - y[0]) for x, y in zip(struct.iter_unpack("<f", raw),
                                                        struct.iter_unpack("<f", bf[name][1])))
            max_abs[name] = diff
        phases = []
        for path in (out / "elastic/sessions").glob("*/events.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            plan, = [e for e in events if e["type"] == "batch_plan"]
            actual = [e for e in events if e["type"] == "update"]
            phases.append([actual[0]["completed_updates"], actual[-1]["completed_updates"],
                           plan["physical_batch"], plan["physical_accumulation"], plan["effective_batch"]])
        assert sorted(phases) == [[1, 20, 8, 1, 8], [21, 35, 4, 2, 8], [36, 50, 1, 8, 8]]
        result = dict(scope="Native fresh processes; tiny Qwen; not task accuracy or App UI", combined=a.combined,
                      matched_validation_checkpoints=len(evals[0]), max_relative_ppl_diff=ppl_relative,
                      phases=sorted(phases), max_abs_loss_diff=delta, state_tensor_max_abs_diff=max_abs,
                      final_reference_loss=ref_rows[-1]["loss"], final_elastic_loss=split_rows[-1]["loss"],
                      dataset_state_exact=True, optimizer_metadata_exact=True, update_lr_token_counts_exact=True)
        (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        assert delta <= 2e-5 and max(max_abs.values()) <= 2e-5 and ppl_relative <= 2e-5, result
        completed = True
    finally:
        if completed:
            shell("rm", "-r", remote)
        else:
            print("Retained emulator scratch for diagnosis:", remote)


if __name__ == "__main__":
    main()
