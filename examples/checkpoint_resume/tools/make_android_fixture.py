#!/usr/bin/env python3
"""Generate a small deterministic random Qwen test model; not benchmark data."""
import json
import math
from pathlib import Path
import random
import struct


def main():
    root = Path(__file__).resolve().parents[1] / 'android/app/src/main/assets/tiny_qwen'
    (root / 'model').mkdir(parents=True, exist_ok=True)
    config = dict(model_type='qwen2', vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=10000,
                  eos_token_id=31, pad_token_id=0)
    (root / 'model/config.json').write_text(json.dumps(config, indent=2))
    weights = {'model.embed_tokens.weight': [32, 16], 'model.norm.weight': [16]}
    for i in range(2):
        prefix = f'model.layers.{i}.'
        for n in ('input_layernorm', 'post_attention_layernorm'):
            weights[prefix + n + '.weight'] = [16]
        for n, out in [('q', 16), ('k', 8), ('v', 8), ('o', 16)]:
            weights[prefix + f'self_attn.{n}_proj.weight'] = [out, 16]
            if n != 'o':
                weights[prefix + f'self_attn.{n}_proj.bias'] = [out]
        for n, shape in [('gate', [32, 16]), ('up', [32, 16]), ('down', [16, 32])]:
            weights[prefix + f'mlp.{n}_proj.weight'] = shape
    rng = random.Random(45)
    header, payload = {}, bytearray()
    for name, shape in sorted(weights.items()):
        values = [1.0 if 'norm' in name else (0.0 if name.endswith('bias') else rng.uniform(-.1, .1))
                  for _ in range(math.prod(shape))]
        data = struct.pack('<' + 'f' * len(values), *values)
        header[name] = dict(dtype='F32', shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
        payload.extend(data)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (root / 'model/model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    for split, count in [('train', 17), ('valid', 5)]:
        rows = [dict(ids=[1 + (i + j * 3) % 30 for j in range(16)], mask=[1] * 16) for i in range(count)]
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'spec.json').write_text(json.dumps(dict(model='qwen', steps=200, batch_size=2,
        seq_len=16, lora_r=2, grad_accum=2, lr=.0002, threads=1, slice_seconds=30,
        min_available_mb=128, fixture='random tiny Qwen, NOT pretrained/PIQA'), indent=2))
    print(root)


if __name__ == '__main__':
    main()
