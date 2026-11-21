#!/usr/bin/env python3
"""Recheck retained pretrained pairs, including validation PPL and immutable best weights."""
import argparse
import json
import math
from pathlib import Path
from run_real_resume import load_state, updates
from test_resume_process import equal_tensor_files


def audit(root):
    records = []
    results = json.loads((root / 'results.json').read_text())
    for row in results:
        mode = row['mode']
        a, b = root / (mode + '_continuous'), root / (mode + '_resumed')
        ad, sa = load_state(a); bd, sb = load_state(b)
        equal_tensor_files(ad / 'state.safetensors', bd / 'state.safetensors', str(root))
        assert sa['optimizer'] == sb['optimizer']
        assert sa['dataset_state'] == sb['dataset_state']
        ua, ub = updates(a), updates(b)
        assert len(ua) == len(ub) == row['updates']
        fields = ['loss', 'lr', 'grad_norm', 'total_tokens', 'ema_loss']
        assert all(isinstance(e[k], (int, float)) and math.isfinite(e[k]) for e in ua + ub for k in fields)
        assert [[e[k] for k in fields] for e in ua] == [[e[k] for k in fields] for e in ub]
        def evaluations(run):
            values = [json.loads(line) for p in (run / 'sessions').glob('*/eval.jsonl')
                      for line in p.read_text().splitlines() if line.strip()]
            values.sort(key=lambda e: e['completed_updates'])
            assert len(set(e['completed_updates'] for e in values)) == len(values)
            return values
        ea, eb = evaluations(a), evaluations(b)
        assert ea and ea == eb, f'Validation differs: {root}/{mode}'
        assert all(isinstance(e['valid_ppl'], (int, float)) and math.isfinite(e['valid_ppl']) and e['valid_ppl'] > 0 for e in ea)
        ta, tb = dict(sa['trainer']), dict(sb['trainer'])
        best_a, best_b = Path(ta.pop('best_path')), Path(tb.pop('best_path'))
        equal_tensor_files(best_a, best_b, str(root) + ' best')
        for t in [ta, tb]:
            assert t['timing']['updates'] == row['updates']
            assert t['timing']['full_step_sec'] > 0
            t.pop('timing')
        assert ta == tb
        records.append(dict(fixture=str(root), mode=mode, updates=row['updates'],
            split=row['split'], batch=row['batch'], seq=row['seq'], rank=row['rank'],
            exact_trainable_and_adam_tensors=True, exact_optimizer_metadata=True,
            exact_sampler_and_rng=True, exact_update_trace=True, exact_validation_trace=True,
            exact_best_adapter=True, finite_tensors_and_metrics=True, validation_points=len(ea),
            final_loss=ua[-1]['loss'], last_validation_ppl=ea[-1]['valid_ppl']))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fixtures', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = [row for root in args.fixtures for row in audit(root)]
    args.output.write_text(json.dumps(dict(passed=True, pairs=len(rows), rows=rows), indent=2))
    print(f'PASS: {len(rows)} pairs; tensors, optimizer, sampler/RNG, training and validation trajectories, best adapters')


if __name__ == '__main__':
    main()
