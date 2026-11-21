"""Prepare an offline SAF import folder; never downloads weights or tokenizes data."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile


def validate_data(path, vocab):
    count = 0
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            ids, mask = row.get('ids'), row.get('mask')
            if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask) or len(ids) < 2:
                raise ValueError(f'{path}:{number}: ids/mask must have equal length >= 2')
            if any(type(x) is not int or not 0 <= x < vocab for x in ids):
                raise ValueError(f'{path}:{number}: token outside model vocabulary')
            if any(type(x) is not int or x not in (0, 1) for x in mask) or not any(mask[1:]):
                raise ValueError(f'{path}:{number}: binary mask with supervised targets required')
            count += 1
    if not count:
        raise ValueError(f'{path}: empty dataset')
    return count


def prepare(model, train, valid, output, family, steps=200, seq=64, rank=8, pause_grace=30):
    if output.exists():
        raise ValueError('Output already exists; use a new folder')
    if family not in ('qwen', 'gemma') or min(steps, rank) <= 0 or seq < 2 or not 30 <= pause_grace <= 420:
        raise ValueError('Invalid model/steps/sequence/rank')
    config = json.loads((model / 'config.json').read_text())
    config = config.get('text_config', config)
    vocab = config['vocab_size']
    if type(vocab) is not int or vocab <= 0:
        raise ValueError('Invalid model vocabulary')
    weights = model / 'model.safetensors'
    if not weights.is_file() or weights.stat().st_size <= 8:
        raise ValueError('Expected nonempty single-file model.safetensors')
    for data in (train, valid):
        validate_data(data, vocab)
        with data.open() as stream:
            if any(not any(json.loads(line)['mask'][1:seq]) for line in stream):
                raise ValueError(f'{data}: truncation leaves no supervised targets')
    spec = dict(model=family, steps=steps, batch_size=1, seq_len=seq, lora_r=rank,
                grad_accum=1, lr=0.0002, threads=4, slice_seconds=120,
                pause_grace_seconds=pause_grace, min_available_mb=1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.mft-import-', dir=output.parent))
    try:
        (temp / 'model').mkdir()
        for name in ('config.json', 'model.safetensors'):
            shutil.copyfile(model / name, temp / 'model' / name)
        shutil.copyfile(train, temp / 'train.jsonl')
        shutil.copyfile(valid, temp / 'valid.jsonl')
        (temp / 'experiment.json').write_text(json.dumps(spec, indent=2) + '\n')
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'train', 'valid', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--family', choices=['qwen', 'gemma'], required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--pause-grace', type=int, default=30,
                   help='30..420 seconds; choose from full-profile timing, not only update time')
    a = p.parse_args()
    prepare(a.model, a.train, a.valid, a.output, a.family, a.steps, a.seq, a.rank, a.pause_grace)
    print(f'Prepared {a.output}; copy folder to emulator shared storage and select it in Import.')
    print('Validation covers layout/token ranges, not model-weight correctness or tokenizer identity.')
    print('The 120-second slice is not a guarantee that this workload fits; run a bounded pilot first.')


if __name__ == '__main__':
    main()
