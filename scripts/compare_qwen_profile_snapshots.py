#!/usr/bin/env python3
"""Compare pre-update tensors at exactly the same Qwen parameter/input point."""
import argparse
import array
import json
import math
from pathlib import Path
import sys


def load(path):
    meta = json.loads((path / "snapshot.json").read_text())
    if meta["schema"] != "mobilefinetuner.fixed_point.v1":
        raise ValueError("Unsupported fixed-point snapshot schema")
    expected_name = f"memory{meta['memory_profile']}_stream{int(meta['streaming'])}"
    if path.name != expected_name:
        raise ValueError("Profile metadata does not match its directory")
    raw = (path / "values.bin").read_bytes()
    tensors = {}
    for name, spec in meta["tensors"].items():
        offset, size = spec["offset"], spec["bytes"]
        if (
            offset < 0
            or size != 4 * math.prod(spec["shape"])
            or offset + size > len(raw)
        ):
            raise ValueError("Invalid tensor extent: " + name)
        tensors[name] = (spec, raw[offset : offset + size])
    required = {"input_ids", "attention_mask", "labels", "logits", "loss"}
    if not required.issubset(tensors):
        raise ValueError("Missing inputs, logits or loss")
    parameters = {name.removeprefix("parameter.") for name in tensors if name.startswith("parameter.")}
    gradients = {name.removeprefix("gradient.") for name in tensors if name.startswith("gradient.")}
    if not parameters or parameters != gradients:
        raise ValueError("Every adapter parameter must have a gradient snapshot")
    return tensors


def floats(tensor):
    spec, raw = tensor
    if spec["dtype"] != "F32":
        raise ValueError("Expected FP32 tensor")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite diagnostic tensor")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    directories = sorted(p.parent for p in args.root.glob("*/snapshot.json"))
    expected_profiles = {f"memory{m}_stream{s}" for m in (0, 2, 4, 6) for s in (0, 1)}
    if {p.name for p in directories} != expected_profiles:
        raise ValueError("Expected all 8 distinct memory/stream profiles")
    baseline = load(args.root / "memory0_stream0")
    criteria = dict(
        gradient_relative_l2=1e-3,
        gradient_l2_floor=2e-7,
        gradient_max_relative_scale=2e-3,
        gradient_max_floor=2e-6,
        logits_atol=2e-3,
        logits_rtol=2e-4,
        loss_atol=4e-4,
        loss_rtol=6e-4,
    )
    result = dict(
        state="running", baseline="memory0_stream0", criteria=criteria, profiles=[]
    )
    errors = []
    for directory in directories:
        actual = load(directory)
        if actual.keys() != baseline.keys():
            raise ValueError("Tensor names changed: " + directory.name)
        metrics = dict(
            profile=directory.name,
            gradient_tensors=0,
            gradient_max_abs=0.0,
            gradient_max_relative_l2=0.0,
            logits_max_abs=0.0,
            loss_max_abs=0.0,
        )
        for name, reference in baseline.items():
            if (
                actual[name][0]["shape"] != reference[0]["shape"]
                or actual[name][0]["dtype"] != reference[0]["dtype"]
            ):
                raise ValueError("Tensor shape or dtype changed: " + name)
            if name.startswith("parameter.") or name in [
                "input_ids",
                "attention_mask",
                "labels",
            ]:
                if name.startswith("parameter."):
                    floats(actual[name])
                if actual[name][1] != reference[1]:
                    raise ValueError(
                        "Parameter/input point is not byte-identical: " + name
                    )
                continue
            a, b = floats(reference), floats(actual[name])
            maximum = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
            if name.startswith("gradient."):
                squared_error = sum((x - y) * (x - y) for x, y in zip(a, b))
                norm = math.sqrt(sum(x * x for x in a))
                scale = max((abs(x) for x in a), default=0.0)
                rel = math.sqrt(squared_error) / max(norm, 1e-30)
                metrics["gradient_tensors"] += 1
                metrics["gradient_max_abs"] = max(metrics["gradient_max_abs"], maximum)
                metrics["gradient_max_relative_l2"] = max(
                    metrics["gradient_max_relative_l2"], rel
                )
                if (
                    math.sqrt(squared_error) > 1e-3 * norm + 2e-7
                    or maximum > 2e-3 * scale + 2e-6
                ):
                    errors.append(directory.name + ": gradient " + name)
            else:
                kind = "loss" if name == "loss" else "logits"
                metrics[kind + "_max_abs"] = max(metrics[kind + "_max_abs"], maximum)
                atol, rtol = criteria[kind + "_atol"], criteria[kind + "_rtol"]
                if any(abs(x - y) > atol + rtol * abs(x) for x, y in zip(a, b)):
                    errors.append(directory.name + ": " + name)
        if metrics["gradient_tensors"] == 0:
            raise ValueError("No adapter gradients in snapshot")
        result["profiles"].append(metrics)
    result.update(
        state="passed" if not errors else "not_passed",
        errors=errors,
        interpretation="Pre-update fixed-point comparison; does not replace the recorded post-Adam coordinate diagnostic",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "PASS:",
        len(directories),
        "profiles; byte-identical parameters/inputs; logits/loss and every adapter gradient within declared tolerances",
    )


if __name__ == "__main__":
    main()
