"""Bounded native-only emulator probe; not an Android UI/Worker acceptance test."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def probe_metadata(args):
    result = dict(scope='Android emulator native-only probe, shell UID, not UI or Worker',
                  mode='resume' if args.resume else 'fresh', requested_threads=4)
    if not args.resume:
        result['requested_training'] = dict(steps=args.steps, seq=args.seq, microbatch=1, rank=8,
                                            optimization=args.optimization)
    return result


def collect_evidence(adb, run, output):
    with (output / 'pull.log').open('w') as log:
        pulled = subprocess.run(adb + ['pull', run, str(output / 'run')], check=False,
                                stdout=log, stderr=subprocess.STDOUT, timeout=120)
    return dict(pull_returncode=pulled.returncode,
                evidence_collected=pulled.returncode == 0 and (output / 'run' / 'LATEST').is_file())


def observed_state(run):
    pointer = json.loads((run / 'LATEST').read_text())
    generation = pointer['current']
    if not isinstance(generation, str) or Path(generation).name != generation or generation in ('.', '..'):
        raise ValueError('Invalid generation')
    state = json.loads((run / 'training_state' / generation / 'state.json').read_text())
    return dict(completed_updates=state['trainer']['completed_updates'],
                total_steps=state['trainer']['total_steps'], status=state['status'])


def outcome_matches(observed, args):
    total, done = observed['total_steps'], observed['completed_updates']
    if type(total) is not int or type(done) is not int or not 0 <= done <= total:
        return False
    if not args.resume and total != args.steps:
        return False
    target = min(args.stop, total) if args.stop > 0 else total
    return done == target and observed['status'] == ('completed' if target == total else 'suspended')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', required=True)
    p.add_argument('--remote', required=True)
    p.add_argument('--name', default='pilot')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--optimization', choices=['baseline', 'accumulation', 'checkpointing', 'streaming',
                                              'sharding', 'combined'], default='baseline')
    p.add_argument('--stop', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--initial')
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    if args.seq < 2 or args.steps < 1:
        p.error("Expected --steps >= 1 and --seq >= 2")
    assert args.serial.startswith('emulator-')
    assert args.remote.startswith('/data/local/tmp/mft_vm17_')
    assert args.name.isidentifier()
    args.output.mkdir(parents=True, exist_ok=False)
    adb = [args.adb, '-s', args.serial]
    root = args.remote
    run = root + '/' + args.name
    command = [root + '/trainer']
    if args.resume:
        command += ['--resume_state', run]
    else:
        command += ['--model_dir', root, '--jsonl_train', root + '/train.jsonl',
                    '--jsonl_valid', root + '/valid.jsonl', '--run_dir', run,
                    '--steps', str(args.steps), '--batch_size', '1', '--seq_len', str(args.seq),
                    '--lora_r', '8', '--lora_alpha', '16', '--lora_dropout', '0',
                    '--lr', '0.0002', '--lr_scheduler', 'cosine', '--warmup_steps', '2',
                    '--log_interval', '1', '--eval_interval', '3', '--eval_batches', '1',
                    '--save_every', '3', '--seed', '42', '--shuffle', '--save_initial_adapter']
        if args.initial:
            command += ['--resume_from', args.initial]
        modes = {'accumulation': ['--grad_accum', '2'], 'checkpointing': ['--activation_checkpointing'],
                 'streaming': ['--memory_efficient_attention'],
                 'sharding': ['--shard_enable', '--shard_budget_mb', '768', '--shard_fp16_disk', 'false']}
        for mode, flags in modes.items():
            if args.optimization in (mode, 'combined'):
                command += flags
    command += ['--slice_seconds', '0', '--checkpoint_every_updates', '1',
                '--suspend_after_updates', str(args.stop)]
    environment = ['MFT_NUM_THREADS=4', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                   'VECLIB_MAXIMUM_THREADS=4', 'MKL_NUM_THREADS=4', 'BLIS_NUM_THREADS=4']
    pidfile = run + '.pid'
    shell = 'echo $$ > ' + shlex.quote(pidfile) + '; exec env ' + shlex.join(environment + command)
    (args.output / 'command.json').write_text(json.dumps(command, indent=2))
    start = time.monotonic()
    result = probe_metadata(args)
    launched = False
    try:
        result['stage'] = 'preflight'
        if not args.resume:
            subprocess.run(adb + ['shell', 'test', '!', '-e', run], check=True, timeout=15)
        result['stage'] = 'training'
        launched = True
        with (args.output / 'train.log').open('w') as log:
            proc = subprocess.run(adb + ['shell', shell], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.timeout)
        result['returncode'] = proc.returncode
    except (subprocess.SubprocessError, OSError) as error:
        result['error'] = str(error)
        result['timeout'] = isinstance(error, subprocess.TimeoutExpired)
        if launched and result['timeout']:
            try:
                pid = subprocess.check_output(adb + ['shell', 'cat', pidfile], text=True, timeout=5).strip()
                if pid.isdigit():
                    cmdline = subprocess.check_output(adb + ['shell', 'cat', f'/proc/{pid}/cmdline'], timeout=5)
                    if run.encode() in cmdline:
                        subprocess.run(adb + ['shell', 'kill', '-9', pid], check=True, timeout=5)
            except (subprocess.SubprocessError, OSError) as cleanup_error:
                result['cleanup_error'] = str(cleanup_error)
    finally:
        result['elapsed_sec'] = time.monotonic() - start
        result['passed'] = False
        if launched:
            try:
                result.update(collect_evidence(adb, run, args.output))
                if result['evidence_collected']:
                    result['observed_committed_state'] = observed_state(args.output / 'run')
                    result['passed'] = result.get('returncode') == 0 and outcome_matches(result['observed_committed_state'], args)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as pull_error:
                result['evidence_error'] = str(pull_error)
        if args.resume:
            result['scope_note'] = 'Observed cumulative state only; resume may be a no-op of an already completed run.'
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
