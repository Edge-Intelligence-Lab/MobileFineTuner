#!/usr/bin/env python3
"""Hermetic Qwen CLI checks using tiny deterministic weights; no HF/network needed."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def write_model(root):
    root.mkdir()
    config = dict(
        model_type="qwen2",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=True,
        eos_token_id=0,
        pad_token_id=0,
    )
    (root / "config.json").write_text(json.dumps(config))
    (root / "vocab.json").write_text(
        json.dumps(
            {
                "<|endoftext|>": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "Ġ": 8,
            }
        )
    )
    (root / "merges.txt").write_text("#version: 0.2\n")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>"})
    )
    tensors = {"model.embed_tokens.weight": [32, 8], "model.norm.weight": [8]}
    prefix = "model.layers.0."
    tensors.update(
        {
            prefix + "input_layernorm.weight": [8],
            prefix + "post_attention_layernorm.weight": [8],
        }
    )
    for name, width in [("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)]:
        tensors[prefix + "self_attn." + name + ".weight"] = [width, 8]
        if name != "o_proj":
            tensors[prefix + "self_attn." + name + ".bias"] = [width]
    for name, shape in [
        ("gate_proj", [12, 8]),
        ("up_proj", [12, 8]),
        ("down_proj", [8, 12]),
    ]:
        tensors[prefix + "mlp." + name + ".weight"] = shape
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, shape in sorted(tensors.items()):
        count = math.prod(shape)
        if len(shape) == 1:
            values = [0.0 if name.endswith(".bias") else 1.0] * count
        else:
            values = [0.02 * math.sin(i + 1) for i in range(count)]
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * count, *values))
        header[name] = dict(
            dtype="F32", shape=shape, data_offsets=[start, len(payload)]
        )
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(text)) + text + payload
    )


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_mmlu(binary, root, model):
    # Keep all prompt characters inside the tiny model's vocabulary.
    vocab_path = model / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    for character in "QuestionAnswerBCD.:Ċ":
        if character not in vocab:
            vocab[character] = len(vocab)
    assert max(vocab.values()) < 32
    vocab_path.write_text(json.dumps(vocab))
    data = root / "mmlu_data"
    (data / "dev").mkdir(parents=True)
    (data / "dev/tiny_dev.csv").write_text("a,b,c,d,e,A\n" * 8)
    common = [
        str(binary.resolve()),
        "--model_dir",
        str(model),
        "--data_dir",
        str(data),
        "--seq_len",
        "128",
        "--batch_size",
        "1",
        "--grad_accum_steps",
        "2",
        "--lora_r",
        "2",
        "--lora_alpha",
        "4",
        "--lora_dropout",
        "0",
        "--lora_targets",
        "all-linear",
        "--log_interval",
        "1",
    ]
    adapters = []
    for name, extra in [
        ("mmlu_train", []),
        ("mmlu_resume", ["--resume_from", str(root / "mmlu_train")]),
    ]:
        result = subprocess.run(
            common
            + [
                "--max_steps",
                "2",
                "--output_dir",
                str(root / name),
                "--log_file",
                str(root / (name + ".log")),
            ]
            + extra,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=root,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout
        blob = (root / name / "adapter_model.safetensors").read_bytes()
        size = struct.unpack("<Q", blob[:8])[0]
        tensors = {}
        for key, entry in json.loads(blob[8 : 8 + size]).items():
            if key == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            assert entry["dtype"] == "F32"
            values = struct.unpack(
                "<" + "f" * ((end - start) // 4),
                blob[8 + size + start : 8 + size + end],
            )
            assert all(math.isfinite(value) for value in values)
            if ".lora_B." in key:
                assert any(value != 0 for value in values)
            tensors[key] = values
        assert len(tensors) == 14
        adapters.append(tensors)
    assert all(adapters[0][key] != adapters[1][key] for key in adapters[0])
    result = subprocess.run(
        common
        + [
            "--seq_len",
            "2",
            "--max_steps",
            "1",
            "--output_dir",
            str(root / "truncated"),
            "--log_file",
            str(root / "truncated.log"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=root,
        timeout=30,
    )
    assert result.returncode != 0 and "no answer tokens" in result.stdout, result.stdout
    print(
        "PASS: MMLU native training/resume, finite updated adapters, truncated-answer rejection"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mmlu-binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="mft-qwen-cli-") as temporary:
        root = Path(temporary)
        model, data = root / "model", root / "data"
        write_model(model)
        data.mkdir()
        for split in ["train", "valid", "test"]:
            (data / f"wiki.{split}.raw").write_text(
                "a b c d e f g a b c d e f g a b c d e f g\n"
            )
        common = [
            str(binary),
            "--model_dir",
            str(model),
            "--data_dir",
            str(data),
            "--seq_len",
            "4",
            "--batch_size",
            "1",
            "--lora_r",
            "2",
            "--lora_alpha",
            "4",
            "--lora_dropout",
            "0",
            "--lr",
            "0",
            "--max_steps",
            "-1",
            "--no_shuffle",
            "--stability_checks",
        ]
        # Choose a sequence count whose final accumulation group contains one batch.
        for repeats in range(1, 20):
            (data / "wiki.train.raw").write_text("a b c d " * repeats + "\n")
            result = subprocess.run(
                common
                + [
                    "--output_dir",
                    str(root / "probe"),
                    "--dump_first_batch_json",
                    str(root / "batch.json"),
                    "--dump_only",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout
            match = re.search(r"Total sequences: (\d+)", result.stdout)
            count = int(match[1])
            if count >= 4 and count % 3 == 1:
                break
        else:
            raise AssertionError("Could not construct an incomplete accumulation group")
        for name, accumulation in [("micro", 1), ("accumulated", 3)]:
            result = subprocess.run(
                common
                + [
                    "--grad_accum_steps",
                    str(accumulation),
                    "--save_every",
                    "1",
                    "--output_dir",
                    str(root / name),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            assert result.returncode == 0, result.stdout
        micro = read_csv(root / "micro/step_metrics.csv")
        accumulated = read_csv(root / "accumulated/step_metrics.csv")
        assert len(micro) == count
        assert len(accumulated) == (count + 2) // 3
        for i, row in enumerate(accumulated):
            group = micro[i * 3 : (i + 1) * 3]
            expected = sum(float(r["loss"]) for r in group) / len(group)
            assert abs(float(row["loss"]) - expected) < 3e-6, (row, expected)
        micro_stats = read_csv(root / "micro/stability_metrics.csv")
        accumulated_stats = read_csv(root / "accumulated/stability_metrics.csv")
        expected = float(micro_stats[-1]["grad_l2_before_clip"])
        actual = float(accumulated_stats[-1]["grad_l2_before_clip"])
        assert abs(actual - expected) <= 1e-6 + 1e-5 * abs(expected), (actual, expected)
        assert all(
            math.isfinite(float(v)) for row in accumulated_stats for v in row.values()
        )
        assert (
            root
            / "accumulated"
            / f"checkpoint-{len(accumulated)}"
            / "adapter_model.safetensors"
        ).is_file()
        assert (root / "accumulated/adapter_config.json").is_file()
        # Bad numeric controls must fail before loading a model or dividing by zero.
        for key, value in [
            ("--batch_size", "0"),
            ("--grad_accum_steps", "0"),
            ("--seq_len", "1"),
            ("--save_every", "-1"),
            ("--lr", "nan"),
        ]:
            result = subprocess.run(
                [str(binary), key, value],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
            assert result.returncode != 0 and "[ERROR]" in result.stdout, result.stdout
        if args.mmlu_binary:
            test_mmlu(args.mmlu_binary, root, model)
    print(
        "PASS: accumulation tail, actual gradient/loss normalization, finite metrics, snapshots and invalid arguments"
    )


if __name__ == "__main__":
    main()
