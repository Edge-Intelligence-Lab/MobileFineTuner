#!/usr/bin/env python3
"""Validate shared Qwen execution controls; optional generated tiny Qwen3.5 assets."""
import argparse
import array
import csv
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from test_qwen_training_cli import write_model


def load_adapter(path):
    blob = path.read_bytes()
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + size])
    tensors = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        assert entry["dtype"] == "F32"
        start, end = entry["data_offsets"]
        values = array.array("f")
        values.frombytes(blob[8 + size + start : 8 + size + end])
        assert len(values) == math.prod(entry["shape"]) and all(
            math.isfinite(v) for v in values
        )
        if ".lora_B." in key:
            assert any(v != 0 for v in values)
        tensors[key] = values
    return tensors


def check_close(reference, actual, name):
    assert reference.keys() == actual.keys(), name
    maximum = 0.0
    for key in reference:
        assert len(reference[key]) == len(actual[key]), (name, key)
        for a, b in zip(reference[key], actual[key]):
            error = abs(a - b)
            maximum = max(maximum, error)
            assert error <= 2e-5 + 3e-4 * max(abs(a), abs(b)), (name, key, a, b)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--qwen35-base", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="mft-execution-") as temporary:
        root = Path(temporary)
        model = root / "qwen2"
        write_model(model)
        models = [model]
        if args.qwen35_base:
            config = json.loads((args.qwen35_base / "config.json").read_text())
            assert (
                config["model_type"] == "qwen3_5_text"
                and config["vocab_size"] <= 256
                and config["hidden_size"] <= 64
                and config["num_hidden_layers"] <= 4
            ), "Use generated tiny Qwen3.5 fixtures"
            hybrid = root / "qwen35"
            shutil.copytree(args.qwen35_base, hybrid)
            for name in ["vocab.json", "merges.txt", "tokenizer_config.json"]:
                shutil.copy2(model / name, hybrid / name)
            models.append(hybrid)
        data = root / "data"
        data.mkdir()
        for split in ["train", "valid", "test"]:
            (data / f"wiki.{split}.raw").write_text("a b c d e f g " * 16 + "\n")
        for model in models:
            common = [
                str(binary),
                "--model_dir",
                str(model),
                "--data_dir",
                str(data),
                "--seq_len",
                "8",
                "--batch_size",
                "1",
                "--max_steps",
                "2",
                "--lora_r",
                "2",
                "--lora_alpha",
                "4",
                "--lora_dropout",
                "0",
                "--lora_targets",
                "all-linear",
                "--lr",
                "0.001",
                "--no_shuffle",
                "--stability_checks",
            ]
            references = None
            reference_loss = None

            def run(name, extra):
                out = root / (model.name + "_" + name)
                result = subprocess.run(
                    common + ["--output_dir", str(out), *extra],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
                assert result.returncode == 0, result.stdout
                with (out / "step_metrics.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 2 and all(
                    math.isfinite(float(r["loss"])) for r in rows
                )
                return (
                    out,
                    result.stdout,
                    load_adapter(out / "adapter_model.safetensors"),
                    [float(r["loss"]) for r in rows],
                )

            for mode in range(8):
                for streamed_head in [False, True]:
                    extra = ["--base_weight_storage", "fp32"]
                    if mode & 1:
                        extra += ["--memory_efficient_attention"]
                    if mode & 2:
                        extra += ["--activation_checkpointing"]
                    if mode & 4:
                        extra += [
                            "--offload_dir",
                            str(
                                root / (model.name + f"_offload{mode}_{streamed_head}")
                            ),
                            "--offload_resident_mb",
                            "8",
                        ]
                    if streamed_head:
                        extra += ["--loss_impl", "full_streaming"]
                    name = f"mode{mode}_head{int(streamed_head)}"
                    out, log, tensors, losses = run(name, extra)
                    if references is None:
                        references = tensors
                        reference_loss = losses
                    error = check_close(references, tensors, name)
                    assert (
                        max(abs(a - b) for a, b in zip(reference_loss, losses)) < 3e-5
                    ), (name, reference_loss, losses)
                    if mode & 4:
                        match = re.search(
                            r"\[Offload\] entries=(\d+) loads=(\d+) evictions=(\d+) bytes_read=(\d+) bytes_written=(\d+) writes=(\d+)",
                            log,
                        )
                        assert match, log
                        entries, loads, evictions, read, written, writes = map(
                            int, match.groups()
                        )
                        assert (
                            loads > entries > 0
                            and evictions > 0
                            and read > 0
                            and written > 0
                            and writes == entries
                        )
                    print(
                        model.name, name, "adapter_max_abs", error, "PASS", flush=True
                    )
            for mode in range(4):
                extra = ["--base_weight_storage", "native"]
                if mode & 1:
                    extra += ["--memory_efficient_attention"]
                if mode & 2:
                    extra += ["--activation_checkpointing"]
                _, _, tensors, losses = run(f"native{mode}", extra)
                error = check_close(references, tensors, f"native{mode}")
                assert max(abs(a - b) for a, b in zip(reference_loss, losses)) < 3e-5
                print(
                    model.name,
                    f"native{mode}",
                    "adapter_max_abs",
                    error,
                    "PASS",
                    flush=True,
                )
            _, log, tensors, losses = run(
                "energy", ["--base_weight_storage", "fp32", "--pm_schedule", "0-:5"]
            )
            assert (
                tensors == references and losses == reference_loss
            ), "Throttling changed updates"
            events = re.findall(
                r"\[Energy\] update=(\d+) requested_sleep_ms=(\d+) observed_sleep_s=([\d.e+-]+)",
                log,
            )
            assert len(events) == 2 and all(
                int(ms) == 5 and float(seconds) >= 0.005 - 1e-6
                for _, ms, seconds in events
            ), log
            _, _, resumed, _ = run(
                "resume_all",
                [
                    "--base_weight_storage",
                    "fp32",
                    "--resume_from",
                    str(root / (model.name + "_mode0_head0")),
                    "--memory_efficient_attention",
                    "--activation_checkpointing",
                    "--loss_impl",
                    "full_streaming",
                    "--offload_dir",
                    str(root / (model.name + "_resume_offload")),
                    "--offload_resident_mb",
                    "8",
                ],
            )
            assert all(
                resumed[k] != references[k] for k in references
            ), "Resume left an adapter tensor unchanged"
            for extra, message in [
                (
                    ["--activation_checkpointing", "--lora_dropout", "0.1"],
                    "require --lora_dropout 0",
                ),
                (
                    ["--offload_dir", "unused", "--base_weight_storage", "native"],
                    "offload requires",
                ),
                (
                    ["--synthetic_smoke", "--memory_efficient_attention"],
                    "memory options require real training",
                ),
            ]:
                result = subprocess.run(
                    common + extra,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=15,
                )
                assert (
                    result.returncode != 0 and message in result.stdout
                ), result.stdout
            print(
                model.name,
                "energy exactness, adapter resume and invalid-option guards PASS",
                flush=True,
            )


if __name__ == "__main__":
    main()
