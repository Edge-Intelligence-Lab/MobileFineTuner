"""Check Qwen scheduler integration with synthetic weights, no external assets."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mft-energy-') as tmp:
        root = Path(tmp)
        logs = {}
        for case, extra in [('baseline', []), ('schedule', ['--pm_schedule', '0-3:10,4-7:0,8-:20'])]:
            command = [str(args.binary.resolve()), '--synthetic_smoke', '--smoke_steps', '10',
                       '--lora_dropout', '0', '--output_dir', str(root / case), *extra]
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            if result.returncode != 0 or '[PASS]' not in result.stdout:
                raise RuntimeError(result.stdout + result.stderr)
            logs[case] = result.stdout
        if '[Energy]' in logs['baseline']:
            raise RuntimeError('Default training unexpectedly enabled throttling')
        events = re.findall(r'\[Energy\] update=(\d+) requested_sleep_ms=(\d+) observed_sleep_s=([\d.e+-]+)', logs['schedule'])
        expected = [10] * 4 + [0] * 4 + [20] * 2
        if len(events) != 10:
            raise RuntimeError('Expected one scheduler event per optimizer update')
        for i, (step, requested, observed) in enumerate(events):
            if int(step) != i + 1 or int(requested) != expected[i]:
                raise RuntimeError(f'Unexpected schedule event: {events[i]}')
            if float(observed) + 1e-6 < int(requested) / 1000:
                raise RuntimeError('Observed delay is shorter than requested')
        baseline = sorted((root / 'baseline').glob('*.bin'))
        scheduled = sorted((root / 'schedule').glob('*.bin'))
        if not baseline or [p.name for p in baseline] != [p.name for p in scheduled]:
            raise RuntimeError('Missing/mismatched adapter files')
        for first, second in zip(baseline, scheduled):
            if not first.stat().st_size or first.read_bytes() != second.read_bytes():
                raise RuntimeError(f'Throttling changed adapter: {first.name}')
        losses = lambda log: re.findall(r'\[smoke step \d+/10\] loss=([^ ]+)', log)
        if len(losses(logs['baseline'])) != 10 or losses(logs['baseline']) != losses(logs['schedule']):
            raise RuntimeError('Throttling changed the loss sequence')
        invalid_out = root / 'invalid'
        invalid = subprocess.run([str(args.binary.resolve()), '--synthetic_smoke',
            '--output_dir', str(invalid_out), '--pm_schedule', '0-bad:20'],
            text=True, capture_output=True, timeout=30)
        if invalid.returncode != 1 or '[ERROR]' not in invalid.stderr or invalid_out.exists():
            raise RuntimeError('Invalid schedule must fail before model/output creation')
        print('PASS: 10 updates each, exact loss/adapter match, all delays applied, invalid CLI rejected')


if __name__ == '__main__':
    main()
