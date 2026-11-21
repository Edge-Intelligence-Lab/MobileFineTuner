"""Compare committed numeric state in pulled native runs without rewriting paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from effective_metrics import build_report
from test_resume_process import equal_tensor_files, tensor_payloads


def state(run):
    generation = json.loads((run / 'LATEST').read_text())['current']
    directory = run / 'training_state' / generation
    return json.loads((directory / 'state.json').read_text()), directory / 'state.safetensors'


def archived_best(run, original):
    # Absolute Android paths remain in the raw record; resolve only in this reader.
    suffix = original.split('/sessions/', 1)[1]
    return run / 'sessions' / suffix


def compare(reference, candidate):
    a, af = state(reference)
    b, bf = state(candidate)
    assert a['status'] == b['status'] == 'completed'
    equal_tensor_files(af, bf, 'Final LoRA and Adam')
    assert a['optimizer'] == b['optimizer'], 'Adam metadata differs'
    assert a['dataset_state'] == b['dataset_state'], 'Dataset order/cursor/RNG differs'
    ta, tb = a['trainer'].copy(), b['trainer'].copy()
    best_a, best_b = ta.pop('best_path'), tb.pop('best_path')
    assert bool(best_a) == bool(best_b), 'Only one trajectory has a best adapter'
    if best_a:
        equal_tensor_files(archived_best(reference, best_a), archived_best(candidate, best_b), 'Best adapter')
    ta.pop('timing'); tb.pop('timing')
    assert ta == tb, 'Trainer numeric state differs'
    reports = [build_report(run) for run in (reference, candidate)]
    assert all(r['valid'] for r in reports), [r['errors'] for r in reports]
    fields = ('completed_updates', 'loss', 'lr', 'grad_norm', 'total_tokens', 'ema_loss')
    updates = [[{key: e[key] for key in fields} for e in r['effective_updates']] for r in reports]
    assert updates[0] == updates[1], 'Effective update trajectory differs'
    def numeric_evals(report):
        return [{k: v for k, v in e.items() if k not in ('event_seq', 'session_id', 'session_elapsed_sec', 'best_path')}
                for e in report['effective_evaluations']]
    assert numeric_evals(reports[0]) == numeric_evals(reports[1]), 'Effective evaluation differs'
    assert bool(best_a) == any(e.get('best_path') for e in reports[0]['effective_evaluations']), 'Best reference disagrees with evaluations'
    for x, y in zip(reports[0]['effective_evaluations'], reports[1]['effective_evaluations']):
        if x.get('best_path') or y.get('best_path'):
            equal_tensor_files(archived_best(reference, x['best_path']),
                               archived_best(candidate, y['best_path']), 'Evaluation best adapter')
    starts = [[x['event'] for x in r['physical_attempts']
               if x['event'] and x['event']['type'] == 'session_start'] for r in reports]
    assert len(starts[0]) == 1, 'Reference was not a single process'
    pids = [x['pid'] for x in starts[1]]
    assert len(pids) > 1 and len(set(pids)) == len(pids), 'Candidate did not restart native processes'
    return dict(passed=True, updates=len(updates[0]), evaluations=len(numeric_evals(reports[0])),
                exact_final_tensors=len(tensor_payloads(af)), reference_pid=starts[0][0]['pid'],
                resumed_pids=pids, final_loss=updates[0][-1]['loss'],
                scope='Same-build numeric continuity, not full task accuracy or UI validation',
                reference_summary=reports[0]['summary'], candidate_summary=reports[1]['summary'])


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'candidate', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Do not overwrite a prior comparison')
    result = compare(a.reference, a.candidate)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
