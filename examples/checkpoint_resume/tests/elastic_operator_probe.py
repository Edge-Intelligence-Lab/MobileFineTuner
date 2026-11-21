"""Localize batch-size dependence before any resume/epoch transition.

Use an existing test_elastic_microbatch output with a successful first update.
Uses its exact model, initial adapter, data and executable; never modifies it.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess

from test_elastic_microbatch import (THREAD_VARIABLES, compare_tensors, lines, load_state,
                                     only, read_json, require, tensor_payloads, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cmd = read_json(args.reference / "result.json")["updates"][0]["arms"]["elastic"]["command"]
    tensors, losses, counts = {}, {}, {}
    report = {"failures": [], "arms": {}}

    def check(ok, message):
        if not ok:
            report["failures"].append(message)

    for batch in (8, 4, 1):
        path = root / str(batch)
        for flag, value in (("--batch_size", batch), ("--grad_accum", 16 // batch), ("--run_dir", path)):
            cmd[cmd.index(flag) + 1] = str(value)
        with (root / f"{batch}.log").open("x") as log:
            child = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=root, timeout=900,
                                   env=dict(os.environ, **{k: "1" for k in THREAD_VARIABLES}))
        info = report["arms"][batch] = {"command": cmd.copy(), "exit": child.returncode}
        write_json(root / "result.json", report)
        require(child.returncode == 0, f"physical {batch}: exit {child.returncode}")
        gen, state = load_state(path)
        require(state["trainer"]["completed_updates"] == 1, "wrong step")
        tensors[batch] = tensor_payloads(gen / "state.safetensors")
        prev = read_json(path / "LATEST")["previous"]
        origin = tensor_payloads(path / "training_state" / prev / "state.safetensors")
        if batch == 8:
            initial = origin
        require(initial == origin, "initial weights differ")
        micros = [e for e in lines(only(path.glob("sessions/*/measurements.jsonl"))) if e["event"] == "micro_step"]
        counts[batch] = [e["supervised_tokens"] for e in micros]
        losses[batch] = info["micro_losses"] = [e["loss"] for e in micros]
        require(len(micros) == 16 // batch, "wrong micro count")
        if batch != 8:
            compare_tensors(tensors[8], tensors[batch], initial, info, check)
    for batch in (8, 4):
        for i, loss in enumerate(losses[batch]):
            s = slice(i * batch, (i + 1) * batch)
            count = sum(counts[1][s])
            mean = sum(x * n for x, n in zip(losses[1][s], counts[1][s])) / count
            check(count == counts[batch][i] and math.isclose(mean, loss, rel_tol=1e-4),
                  f"physical {batch} micro {i}: batched loss {loss} vs weighted per-row {mean}")
    report["status"] = "failed" if report["failures"] else "passed"
    write_json(root / "result.json", report)
    print(f"{report['status']}: {len(report['failures'])} differences; {root}/result.json")
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
