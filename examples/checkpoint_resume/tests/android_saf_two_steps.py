"""Verify a 1+1 manual pause with a prepared two-step folder on a disposable API35 emulator."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', required=True)
    p.add_argument('--folder', required=True, help='Folder already staged under shared Documents')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if not re.fullmatch(r'emulator-\d+', args.serial) or not re.fullmatch(r'[\w-]+', args.folder):
        p.error('Requires an emulator and a simple folder name')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    pkg = 'com.mobilefinetuner.continuity'
    adb = [args.adb, '-s', args.serial]

    def command(*argv):
        return subprocess.check_output(adb + list(argv), text=True, timeout=90).strip()

    def open_app():
        command('shell', 'am', 'start', '-W', '-n', pkg + '/com.mobilefinetuner.visualizer.continuity.TrainingActivity')

    def ui(*options):
        command_line = [sys.executable, str(Path(__file__).with_name('android_ui_step.py')),
                        '--adb', args.adb, '--serial', args.serial, '--output', str(root / 'ui'), *options]
        out = subprocess.check_output(command_line, text=True, timeout=50)
        return [json.loads(line) for line in out.splitlines()]

    def labels(nodes):
        return {n['text'] for n in nodes} | {n['content-desc'] for n in nodes}

    def button(text, direction='down'):
        for _ in range(8):
            if text in labels(ui()):
                return ui('--click', text)
            ui('--scroll', direction)
        raise RuntimeError('Button not visible: ' + text)

    def read(path):
        return json.loads(command('shell', 'run-as', pkg, 'cat', path))

    def wait(check, seconds=540):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = check()
            if value:
                return value
            time.sleep(.5)
        raise TimeoutError('Task did not reach expected state')

    def tasks():
        names = command('shell', 'run-as', pkg, 'ls', 'files/training').split()
        return {n for n in names if re.fullmatch(r'[0-9a-f-]{36}', n)}

    open_app()
    before = tasks()
    picker = button('Import experiment folder', 'up')
    if 'Files in ' + args.folder not in labels(picker):
        if 'Files in Documents' not in labels(picker):
            ui('--click', 'Documents')
        ui('--click', args.folder)
    ui('--click', 'USE THIS FOLDER')
    ui('--click', 'ALLOW')
    task, = wait(lambda: tasks() - before, 60)
    path = 'files/training/' + task
    spec = read(path + '/spec.json')
    assert spec['steps'] == 2
    command('shell', 'pm', 'grant', pkg, 'android.permission.POST_NOTIFICATIONS')
    button('Start / Resume')
    def current():
        pointer = read(path + '/run/LATEST')
        return read(path + '/run/training_state/' + pointer['current'] + '/state.json')
    def ui_state():
        state = read(path + '/ui_state.json')
        if state['state'] == 'failed':
            raise RuntimeError(state['detail'])
        return state['state']
    wait(lambda: ui_state() == 'running')
    wait(lambda: command('shell', 'run-as', pkg, 'ls', path + '/run').split().count('LATEST'))
    button('Pause and disable background continuation')
    wait(lambda: ui_state() == 'suspended')
    paused = current()
    assert paused['trainer']['completed_updates'] == 1
    (root / 'paused.json').write_text(json.dumps(paused, indent=2))
    command('shell', 'am', 'force-stop', pkg)
    open_app()
    assert 'Task: ' + task in labels(ui()), 'Cold launch did not select the paused task'
    button('Start / Resume')
    wait(lambda: ui_state() == 'completed')
    button('Export paused task records')
    ui('--click', 'SAVE')
    name = 'mft-' + task + '.zip'
    def exported():
        files = command('shell', 'find', '/sdcard/Documents', '/sdcard/Download', '-name', name).splitlines()
        if len(files) != 1:
            return False
        command('pull', files[0], str(root / 'export.zip'))
        try:
            with zipfile.ZipFile(root / 'export.zip') as z:
                return z.testzip() is None
        except zipfile.BadZipFile:
            return False
    wait(exported, 90)
    with zipfile.ZipFile(root / 'export.zip') as z:
        assert not any(n.startswith('model/') for n in z.namelist())
        z.extractall(root / 'records')
    run = root / 'records/run'
    latest = json.loads((run / 'LATEST').read_text())
    state = json.loads((run / 'training_state' / latest['current'] / 'state.json').read_text())
    assert state['trainer']['completed_updates'] == 2 and state['status'] == 'completed'
    events = [json.loads(line) for f in (run / 'sessions').glob('*/events.jsonl') for line in f.read_text().splitlines()]
    pids = [e['pid'] for e in events if e['type'] == 'session_start']
    assert len(set(pids)) == len(pids) == 2
    assert [e['completed_updates'] for e in events if e['type'] == 'resume'] == [1]
    (root / 'result.json').write_text(json.dumps(dict(passed=True, task=task, spec=spec, pids=pids,
        paused_updates=1, final_updates=2, scope='system picker and UI, not task accuracy'), indent=2))


if __name__ == '__main__':
    main()
