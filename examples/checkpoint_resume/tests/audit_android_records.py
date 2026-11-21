"""Read-only parity/finite check of exported full-Qwen Android test records."""
import argparse
import json
import math
from pathlib import Path
from run_real_resume import load_state, updates
from test_resume_process import equal_tensor_files


def audit(files):
    result = json.loads((files / 'pretrained_parity_result.json').read_text())
    runs = [files / 'training' / result[k] / 'run' for k in ('baseline', 'resumed')]
    states = [load_state(run) for run in runs]
    (ad, a), (bd, b) = states
    equal_tensor_files(ad / 'state.safetensors', bd / 'state.safetensors', 'Android state')
    assert a['optimizer'] == b['optimizer']
    assert a['dataset_state'] == b['dataset_state']
    traces = [updates(run) for run in runs]
    fields = ('loss', 'lr', 'grad_norm', 'total_tokens', 'ema_loss')
    assert all(len(trace) == result['steps'] for trace in traces)
    assert all(math.isfinite(row[k]) for trace in traces for row in trace for k in fields)
    assert [[row[k] for k in fields] for row in traces[0]] == [[row[k] for k in fields] for row in traces[1]]
    evaluations = []
    for run in runs:
        rows = [json.loads(line) for path in (run / 'sessions').glob('*/eval.jsonl')
                for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: row['completed_updates'])
        assert rows and len({row['completed_updates'] for row in rows}) == len(rows)
        assert all(math.isfinite(row['valid_ppl']) and row['valid_ppl'] > 0 for row in rows)
        evaluations.append(rows)
    assert evaluations[0] == evaluations[1]
    trainers = [dict(state['trainer']) for _, state in states]
    best = []
    for trainer in trainers:
        # Export preserves Android absolute paths; resolve only within extracted app files.
        relative = trainer.pop('best_path').split('/files/', 1)[1]
        path = (files / relative).resolve()
        path.relative_to(files.resolve())
        best.append(path)
        assert trainer.pop('timing')['updates'] == result['steps']
    assert trainers[0] == trainers[1]
    equal_tensor_files(*best, 'Android best adapter')
    return dict(passed=True, source=str(files), steps=result['steps'], split=result['split'],
                finite_tensors_and_metrics=True, exact_state_and_best_tensors=True,
                exact_optimizer_sampler_rng=True, exact_train_and_validation_trace=True,
                validation_points=len(evaluations[0]), environment='Android 15 arm64 emulator')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.files)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
