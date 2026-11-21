#!/usr/bin/env python3
"""Sequential three-repeat phase measurements on pretrained PIQA fixtures.

No concurrent model runs, no cache flushing, no cherry-picking. Each repeat uses
a continuous 5-update reference and a 2+3 fresh-process resume comparison.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from phase_metrics import records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('run_real_resume.py')
    raw, summary = [], []
    for model, folder, binary, csvpath in [
        ('qwen05b', 'qwen_pretrained', 'mft_qwen_resume', 'piqa_qwen/piqa_train_mmlu.csv'),
        ('gemma270m', 'legacy_gemma270m', 'mft_gemma_resume', 'piqa/piqa_train_mmlu.csv'),
        ('gemma1b', 'legacy_gemma1b', 'mft_gemma_resume', 'piqa/piqa_train_mmlu.csv'),
    ]:
        for repeat in range(1, 4):
            run = args.output / f'{model}_{repeat}'
            command = [sys.executable, str(script), '--binary', str(args.build / binary),
                '--model', str(args.assets / folder), '--piqa-csv', str(args.assets / 'research_data' / csvpath),
                '--output', str(run), '--modes', 'baseline', '--steps', '5', '--stop', '2']
            with (args.output / f'{model}_{repeat}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            measurements = list(records(run / 'baseline_resumed'))
            for row in measurements:
                raw.append(dict(model=model, repeat=repeat, **row))
            print(f'PASS {model} repeat {repeat}', flush=True)
        for phase in ['durable_save', 'state_restore', 'resume_ready']:
            group = [r for r in raw if r['model'] == model and r['phase'] == phase]
            # Equal weighting per repeat; report mean of per-repeat means for saves.
            row = dict(model=model, phase=phase, repeats=3, records=len(group))
            for field in ['wall_sec', 'logical_read_bytes', 'logical_write_bytes',
                          'peak_sampled_rss_bytes', 'os_disk_read_bytes', 'os_disk_write_bytes']:
                per_repeat = [statistics.mean(r[field] for r in group if r['repeat'] == i)
                              for i in range(1, 4) if all(r[field] is not None for r in group if r['repeat'] == i)]
                row[field + '_mean'] = statistics.mean(per_repeat) if len(per_repeat) == 3 else None
                row[field + '_min'] = min(r[field] for r in group if r[field] is not None) if per_repeat else None
                row[field + '_max'] = max(r[field] for r in group if r[field] is not None) if per_repeat else None
            summary.append(row)
    for name, rows in [('raw.csv', raw), ('summary.csv', summary)]:
        with (args.output / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output / 'protocol.json').write_text(json.dumps(dict(repeats=3, batch=1, seq=32, rank=2,
        steps=5, pause=2, cpu='Apple M4 Max', precision='FP32', blas=True, thread_env=4,
        cache='uncontrolled warm filesystem cache; no forced cold-disk claim',
        scope='process sampling, not device power or isolated instantaneous RSS peak'), indent=2))


if __name__ == '__main__':
    main()
