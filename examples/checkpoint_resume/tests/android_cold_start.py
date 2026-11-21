#!/usr/bin/env python3
"""Explicit full-model load-budget regression on a test emulator, not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', default='emulator-5554')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.serial.startswith('emulator-'), 'Use a test emulator, not a physical phone'
    args.output.mkdir(parents=True, exist_ok=False)
    adb = [args.adb, '-s', args.serial]
    app = 'com.mobilefinetuner.continuity'
    root = 'files/training/cold-start-qwen'
    def run(*cmd, **kw):
        return subprocess.run(adb + list(cmd), check=True, **kw)
    run('shell', 'run-as', app, 'sh', '-c', f'"test ! -e {root}/spec.json"')
    run('shell', 'run-as', app, 'mkdir', '-p', root + '/model')
    paths = [(args.model / name, 'model/' + name) for name in ['config.json', 'model.safetensors']]
    paths += [(args.data / (name + '.jsonl'), name + '.jsonl') for name in ['train', 'valid']]
    for src, dest in paths:
        with src.open('rb') as f:
            run('exec-in', 'run-as', app, 'sh', '-c', f'cat > {root}/{dest}', stdin=f, timeout=120)
    result = subprocess.run(adb + ['shell', 'am', 'instrument', '-w', '-e', 'class',
        'com.mobilefinetuner.visualizer.continuity.ColdStartCheck',
        app + '.test/androidx.test.runner.AndroidJUnitRunner'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    (args.output / 'instrumentation.log').write_text(result.stdout)
    assert result.returncode == 0 and 'OK (1 test)' in result.stdout, result.stdout
    value = subprocess.check_output(adb + ['shell', 'run-as', app, 'cat', root + '/cold_start_result.json'], text=True)
    assert json.loads(value)['passed']
    (args.output / 'result.json').write_text(value)
    with (args.output / 'records.tar').open('wb') as out:
        run('exec-out', 'run-as', app, 'tar', '-cf', '-', '-C', root,
            'run', 'attempts', 'spec.json', 'control.json', 'ui_state.json', 'cold_start_result.json', stdout=out)
    print('PASS real Qwen cold loading: short slice stops once; longer slice completes')


if __name__ == '__main__':
    main()
