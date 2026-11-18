#!/usr/bin/env python3
"""Generate independent Transformers/PEFT references for the native Qwen graph.

Requires torch, transformers, peft and safetensors in a host-only Python env.
The C++ runtime has no Python dependency. Loss uses the same causal shift in
both frameworks: logits[:, :-1] against labels[:, 1:].
"""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file, load_file
from transformers import (
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)


def generate(
    directory,
    model,
    targets,
    *,
    inference_only=False,
    sequence_length=4,
    optimizer_kind="adam",
    reuse_base=False,
    native_storage=False,
):
    directory.mkdir(parents=True, exist_ok=True)
    model.config._attn_implementation = "eager"
    if not reuse_base:
        base = directory / "base"
        if base.exists() and any(path.is_symlink() for path in base.iterdir()):
            raise RuntimeError(
                "Refusing to overwrite a fixture base that contains symlinks"
            )
        if native_storage:
            model.bfloat16()
        model.save_pretrained(base, safe_serialization=True)
        if native_storage:
            model.float()
    elif not (directory / "base/config.json").exists():
        raise RuntimeError("Requested real-model fixture is missing its base snapshot")
    torch.manual_seed(19)
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=3,
            lora_alpha=6,
            lora_dropout=0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    # Nonzero A and B exercise both gradients from the very first update.
    for name, parameter in wrapped.named_parameters():
        if parameter.requires_grad:
            torch.nn.init.uniform_(parameter, -0.025, 0.025)
    wrapped.save_pretrained(directory / "initial", safe_serialization=True)
    wrapped.eval()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, -100]], dtype=torch.long)
    if sequence_length != 4:
        ids = (
            torch.arange(1, sequence_length + 1).repeat(2, 1) % model.config.vocab_size
        )
        ids[1] = ids[1].flip(0)
        mask = torch.ones_like(ids)
        mask[1, -3:] = 0
        labels = (ids + 1) % model.config.vocab_size
        labels[1, -3:] = -100
    save_file(
        {"ids": ids.int(), "mask": mask.float(), "labels": labels.int()},
        str(directory / "inputs.safetensors"),
    )
    parameters = [p for p in wrapped.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.SGD(parameters, lr=1e-3)
        if optimizer_kind == "sgd"
        else torch.optim.Adam(
            parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
    )
    logits = wrapped(input_ids=ids, attention_mask=mask, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    reference = {
        "logits": logits.detach().contiguous(),
        "loss": loss.detach().reshape(1),
    }
    if not inference_only:
        loss.backward()
        for name, p in wrapped.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                key = name.replace(".default.weight", ".weight")
                reference["grad." + key] = p.grad.detach().clone().contiguous()
        optimizer.step()
        for name, p in wrapped.named_parameters():
            if p.requires_grad:
                key = name.replace(".default.weight", ".weight")
                reference["after." + key] = p.detach().clone().contiguous()
        wrapped.save_pretrained(directory / "after", safe_serialization=True)
        with torch.no_grad():
            reference["after_logits"] = wrapped(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits.contiguous()
    save_file(reference, str(directory / "reference.safetensors"))
    (directory / "protocol.json").write_text(
        json.dumps(
            {
                "inference_only": inference_only,
                "native_storage": native_storage,
                "optimizer": optimizer_kind,
                "full_size": model.config.hidden_size >= 512,
                "sequence_length": sequence_length,
                "targets": targets,
                "loss": loss.item(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
            },
            indent=2,
        )
    )
    print(directory.name, "loss", loss.item(), "references", len(reference), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen_hf_fixtures"))
    parser.add_argument("--real-model", type=Path)
    parser.add_argument("--real-gradients", action="store_true")
    parser.add_argument(
        "--native-storage",
        action="store_true",
        help="Retain frozen F16/BF16 matrices in real-model native comparisons",
    )
    parser.add_argument(
        "--reference-optimizer", choices=["adam", "sgd"], default="adam"
    )
    parser.add_argument("--name", default="real_qwen")
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument(
        "--qwen35", action="store_true", help="Requires Transformers 5.3.0"
    )
    args = parser.parse_args()
    if args.reference_optimizer != "adam" and not args.real_model:
        parser.error("--reference-optimizer applies to --real-model checks")
    if args.real_model and (
        Path(args.name).name != args.name or args.name in (".", "..")
    ):
        parser.error("--name must be a single fixture directory name")
    if args.native_storage and not args.real_model:
        parser.error("--native-storage requires --real-model")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    if args.verify_native:
        verified = 0
        for directory in sorted(args.output.iterdir()):
            if not (directory / "native_standard/adapter_model.safetensors").exists():
                continue
            for mode in ["native_standard", "native_streamed"]:
                config_json = json.loads((directory / "base/config.json").read_text())
                if config_json.get("model_type") in ("qwen3_5", "qwen3_5_text"):
                    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

                    base = Qwen3_5ForCausalLM.from_pretrained(
                        directory / "base",
                        config=Qwen3_5TextConfig.from_pretrained(directory / "base"),
                        dtype=torch.float32,
                        attn_implementation="eager",
                    )
                else:
                    base = AutoModelForCausalLM.from_pretrained(
                        directory / "base",
                        attn_implementation="eager",
                        torch_dtype=torch.float32,
                    )
                model = PeftModel.from_pretrained(base, directory / mode).eval()
                inputs = load_file(str(directory / "inputs.safetensors"))
                reference = load_file(str(directory / "reference.safetensors"))
                with torch.no_grad():
                    logits = model(
                        input_ids=inputs["ids"].long(),
                        attention_mask=inputs["mask"].long(),
                        use_cache=False,
                    ).logits
                full = json.loads((directory / "protocol.json").read_text()).get(
                    "full_size", False
                )
                torch.testing.assert_close(
                    logits,
                    reference["after_logits"],
                    atol=1e-3 if full else 3e-5,
                    rtol=1e-4 if full else 3e-4,
                )
                print(
                    directory.name,
                    mode,
                    "C++ adapter loaded and evaluated by PEFT: PASS",
                    flush=True,
                )
                verified += 1
        if not verified:
            raise RuntimeError(
                "No native trained adapters found; run native parity after reference generation"
            )
        return
    if args.qwen35 and not args.real_model:
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

        for tied, heads, layers, targets in [
            (True, 2, ["linear_attention", "full_attention"], "all-linear"),
            (
                False,
                1,
                ["full_attention", "linear_attention", "linear_attention"],
                "all-linear",
            ),
            (
                True,
                2,
                ["linear_attention", "full_attention"],
                ["in_proj_qkv", "in_proj_a", "q_proj"],
            ),
        ]:
            config = Qwen3_5TextConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=12,
                num_hidden_layers=len(layers),
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                linear_num_key_heads=1,
                linear_num_value_heads=heads,
                linear_key_head_dim=3,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=3,
                layer_types=layers,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 0],
                },
                tie_word_embeddings=tied,
                attention_dropout=0,
                max_position_embeddings=128,
            )
            name = f"qwen35_{'tied' if tied else 'untied'}_{'all' if targets=='all-linear' else 'selected'}"
            generate(args.output / name, Qwen3_5ForCausalLM(config).float(), targets)
            if not tied:
                generate(
                    args.output / (name + "_long"),
                    Qwen3_5ForCausalLM(config).float(),
                    targets,
                    sequence_length=67,
                )
        generate(
            args.output / "qwen35_native",
            Qwen3_5ForCausalLM(config).float(),
            "all-linear",
            native_storage=True,
        )
        plain = args.output / "qwen35_plain"
        generate(plain, Qwen3_5ForCausalLM(config).float(), "all-linear")
        state = load_file(str(plain / "base/model.safetensors"))
        state = {
            key.replace("model.language_model.", "model.", 1): value
            for key, value in state.items()
        }
        save_file(
            state, str(plain / "base/model.safetensors"), metadata={"format": "pt"}
        )

        return
    if args.real_model:
        directory = args.output / args.name
        base = directory / "base"
        base.mkdir(parents=True, exist_ok=True)
        for source in args.real_model.resolve().iterdir():
            if source.is_file() and (
                source.suffix == ".safetensors"
                or source.name == "config.json"
                or source.name.endswith(".index.json")
            ):
                destination = base / source.name
                if destination.exists() and destination.resolve() != source.resolve():
                    raise RuntimeError(
                        f"Fixture base points to a different model: {destination}"
                    )
                if not destination.exists():
                    destination.symlink_to(source)
        if args.qwen35:
            from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

            config = Qwen3_5TextConfig.from_pretrained(args.real_model)
            model = Qwen3_5ForCausalLM.from_pretrained(
                args.real_model,
                config=config,
                dtype=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(args.real_model),
                torch_dtype=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
            )
        generate(
            directory,
            model,
            "all-linear" if args.qwen35 else ["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_only=not args.real_gradients,
            optimizer_kind=args.reference_optimizer,
            reuse_base=True,
            native_storage=args.native_storage,
        )
        return
    for model_type, tied, targets in [
        ("qwen2", True, ["q_proj", "v_proj"]),
        ("qwen2", False, ["k_proj"]),
        ("qwen3", True, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("qwen3", False, "all-linear"),
    ]:
        cls, config_cls = (
            (Qwen2ForCausalLM, Qwen2Config)
            if model_type == "qwen2"
            else (Qwen3ForCausalLM, Qwen3Config)
        )
        options = dict(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=tied,
            attention_dropout=0,
            rope_theta=10000,
            rms_norm_eps=1e-6,
        )
        if model_type == "qwen3":
            options.update(head_dim=6, attention_bias=False)
        model = cls(config_cls(**options)).float()
        model.config._attn_implementation = "eager"
        generate(
            args.output / f"{model_type}_{'tied' if tied else 'untied'}", model, targets
        )
    options.update(attention_bias=True)
    generate(
        args.output / "qwen3_bias",
        Qwen3ForCausalLM(Qwen3Config(**options)).float(),
        ["o_proj"],
    )

    native_options = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        attention_dropout=0,
        rope_theta=10000,
        rms_norm_eps=1e-6,
    )
    generate(
        args.output / "qwen2_native",
        Qwen2ForCausalLM(Qwen2Config(**native_options)).float(),
        "all-linear",
        native_storage=True,
    )


if __name__ == "__main__":
    main()
