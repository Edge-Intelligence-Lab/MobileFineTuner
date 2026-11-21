"""One controlled native-only Android training pair; keep all attempts on failure."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

from compare_archived_runs import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', required=True)
    p.add_argument('--family', required=True, choices=['qwen', 'gemma'])
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path, help='Directory with train.jsonl and valid.jsonl')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--stop', type=int, default=4)
    p.add_argument('--timeout', type=int, default=3600, help='Per-process seconds; set from workload pilot')
    p.add_argument('--optimization', choices=['baseline', 'checkpointing', 'streaming', 'accumulation',
                                              'sharding', 'combined'], default='baseline')
    a = p.parse_args()
    if not re.fullmatch(r'emulator-\d+', a.serial) or not 0 < a.stop < a.steps or a.seq < 2 or a.timeout <= 0:
        p.error('Requires an emulator, 0 < stop < steps, seq >= 2, and positive timeout')
    here = Path(__file__).resolve().parent
    binary = here.parent / f'android/app/src/main/jniLibs/arm64-v8a/libmft_{a.family}.so'
    files = [(binary, 'trainer'), (a.model / 'config.json', 'config.json'),
             (a.model / 'model.safetensors', 'model.safetensors'),
             (a.data / 'train.jsonl', 'train.jsonl'), (a.data / 'valid.jsonl', 'valid.jsonl')]
    if any(not path.is_file() for path, _ in files):
        p.error('Missing model, data or native executable')
    a.output.mkdir(parents=True, exist_ok=False)
    remote = '/data/local/tmp/mft_vm17_' + uuid.uuid4().hex
    adb = [a.adb, '-s', a.serial]
    def call(*args):
        subprocess.run(adb + list(args), check=True, timeout=300)
    def probe(phase, name, *extra):
        subprocess.run([sys.executable, str(here / 'android_native_probe.py'), '--adb', a.adb,
                        '--serial', a.serial, '--remote', remote, '--name', name, '--steps', str(a.steps),
                        '--seq', str(a.seq), '--output', str(a.output / phase), '--timeout', str(a.timeout),
                        '--optimization', a.optimization, *extra], check=True)
    result = dict(passed=False, scope='native only, not App/WorkManager or accuracy', remote=remote)
    try:
        call('shell', 'mkdir', remote)
        for path, name in files:
            call('push', str(path), remote + '/' + name)
        call('shell', 'chmod', '755', remote + '/trainer')
        probe('reference', 'reference')
        initial, = (a.output / 'reference/run/sessions').glob('*/artifacts/lora_initial.safetensors')
        call('push', str(initial), remote + '/initial.safetensors')
        probe('pause', 'candidate', '--initial', remote + '/initial.safetensors', '--stop', str(a.stop))
        probe('resume', 'candidate', '--resume')
        result.update(compare(a.output / 'reference/run', a.output / 'resume/run'))
        result.update(family=a.family, optimization=a.optimization, stop=a.stop, seq=a.seq)
    finally:
        (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    # Only a successful pair's unique scratch directory is disposable; failures retain diagnostics.
    call('shell', 'rm', '-r', remote)


if __name__ == '__main__':
    main()
