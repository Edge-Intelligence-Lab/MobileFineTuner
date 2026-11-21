"""Stage and execute the opt-in full Qwen Runner test on a disposable emulator."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import time


def validate_records(path):
    required = {'spec.json', 'control.json', 'ui_state.json',
                'profile_result.json', 'profile_attempts.json', 'run/LATEST'}
    found = set()
    attempt_log = False
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            name = member.name.removeprefix('./')
            if not member.isfile():
                continue
            # Read payloads as well as headers so truncated checkpoint data fails validation.
            with archive.extractfile(member) as stream:
                if name in required:
                    if member.size == 0 or not isinstance(json.load(stream), dict):
                        raise ValueError(f'Invalid record: {name}')
                    found.add(name)
                else:
                    while stream.read(1024 * 1024):
                        pass
            if name.startswith('attempts/') and name.endswith('.log') and member.size > 0:
                attempt_log = True
    if required - found:
        raise ValueError('Missing archive records: ' + ', '.join(sorted(required - found)))
    if not attempt_log:
        raise ValueError('Missing nonempty attempt log in archive')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'initial', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    adb = [a.adb, '-s', a.serial]
    app = 'com.mobilefinetuner.continuity'
    root = 'files/training/profile-qwen'
    spec = dict(model='qwen', steps=5, batch_size=1, seq_len=64, lora_r=8,
                grad_accum=1, lr=0.0002, threads=4, slice_seconds=120,
                pause_grace_seconds=300, min_available_mb=1024)
    assets = [(a.model / name, 'model/' + name) for name in ('config.json', 'model.safetensors')]
    assets += [(a.data / (name + '.jsonl'), name + '.jsonl') for name in ('train', 'valid')]
    assets += [(a.initial, 'initial.safetensors'), (a.output / 'spec.json', 'spec.json')]
    command = adb + ['shell', 'am', 'instrument', '-w', '-e', 'class',
                    'com.mobilefinetuner.visualizer.continuity.FullProfileCheck',
                    app + '.test/androidx.test.runner.AndroidJUnitRunner']
    started = time.monotonic()
    result = dict(scope='Full Qwen seq64 via TrainingRunner; not SAF or background Worker',
                  passed=False, instrumentation_passed=False, evidence_collected=False, stage='preflight')
    owns_task = False
    with (a.output / 'harness.log').open('w') as harness_log:
        try:
            if not a.serial.startswith('emulator-'):
                raise ValueError('Disposable emulator only')
            (a.output / 'spec.json').write_text(json.dumps(spec, indent=2))
            (a.output / 'command.json').write_text(json.dumps(command, indent=2))
            for path, _ in assets:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing staging asset: {path}')
            subprocess.run(adb + ['shell', 'run-as', app, 'test', '!', '-e', root], check=True,
                           stdout=harness_log, stderr=subprocess.STDOUT, timeout=15)
            # Never collect from a pre-existing task after the no-overwrite check fails.
            owns_task = True
            result['stage'] = 'staging'
            subprocess.run(adb + ['shell', 'run-as', app, 'mkdir', '-p', root + '/model'], check=True,
                           stdout=harness_log, stderr=subprocess.STDOUT, timeout=15)
            for path, name in assets:
                print(f'Staging {name}', file=harness_log, flush=True)
                with path.open('rb') as stream:
                    subprocess.run(adb + ['exec-in', 'run-as', app, 'sh', '-c', 'cat > ' + root + '/' + name],
                                   stdin=stream, check=True, stdout=harness_log,
                                   stderr=subprocess.STDOUT, timeout=120)
            result['stage'] = 'instrumentation'
            # Preserve the existing completed-run timing: instrumentation, excluding staging/extraction.
            started = time.monotonic()
            with (a.output / 'instrumentation.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1300)
            result['returncode'] = completed.returncode
            result['instrumentation_passed'] = completed.returncode == 0 and 'OK (1 test)' in (
                a.output / 'instrumentation.log').read_text(errors='replace')
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            result['error'] = str(error)
            result['timeout'] = isinstance(error, subprocess.TimeoutExpired)
            print(f"{result['stage']}: {error}", file=harness_log, flush=True)
            if result['stage'] == 'instrumentation' and result['timeout']:
                try:
                    subprocess.run(adb + ['shell', 'am', 'force-stop', app], check=True,
                                   stdout=harness_log, stderr=subprocess.STDOUT, timeout=15)
                except (subprocess.SubprocessError, OSError) as cleanup_error:
                    result['cleanup_error'] = str(cleanup_error)
                    print(f'Cleanup: {cleanup_error}', file=harness_log, flush=True)
        finally:
            result['elapsed_sec'] = time.monotonic() - started
            if owns_task:
                try:
                    with (a.output / 'records.tar').open('wb') as stream, (a.output / 'records.log').open('w') as log:
                        extracted = subprocess.run(adb + ['exec-out', 'run-as', app, 'tar', '-cf', '-', '-C', root,
                                                        'run', 'attempts', 'spec.json', 'control.json', 'ui_state.json',
                                                        'profile_result.json', 'profile_attempts.json'],
                                                   stdout=stream, stderr=log, check=False, timeout=120)
                    result['extraction_returncode'] = extracted.returncode
                    if extracted.returncode != 0:
                        raise ValueError(f'tar exited with status {extracted.returncode}')
                    validate_records(a.output / 'records.tar')
                    result['evidence_collected'] = True
                except (subprocess.SubprocessError, OSError, ValueError, tarfile.TarError) as extraction_error:
                    result['extraction_error'] = str(extraction_error)
                    print(f'Extraction: {extraction_error}', file=harness_log, flush=True)
            result['passed'] = result['instrumentation_passed'] and result['evidence_collected']
            (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
