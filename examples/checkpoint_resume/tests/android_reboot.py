#!/usr/bin/env python3
"""Reboot an explicitly selected test emulator, retaining and verifying its task.

Requires the test APK and main APK already installed. Refuses a physical serial.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import re
import xml.etree.ElementTree as ET


def visible_node(node):
    bounds = list(map(int, re.findall(r'\d+', node.attrib.get('bounds', ''))))
    return len(bounds) == 4 and bounds[2] > bounds[0] and bounds[3] > bounds[1]


def checkbox_for_label(current, label):
    labels = [n for n in current if n.attrib.get('text') == label]
    if len(labels) != 1:
        return None
    bounds = list(map(int, re.findall(r'\d+', labels[0].attrib['bounds'])))
    if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    matches = []
    for node in current:
        if node.attrib.get('class') != 'android.widget.CheckBox':
            continue
        x1, y1, x2, y2 = map(int, re.findall(r'\d+', node.attrib['bounds']))
        # Compose semantics can round touching edges to a one-pixel overlap.
        if x2 > x1 and y2 > y1 and x1 < bounds[0] and x2 <= bounds[0] + 2 and max(y1, bounds[1]) < min(y2, bounds[3]):
            matches.append(node)
    return matches[0] if len(matches) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', default='emulator-5554')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['reboot', 'user-stop', 'paused-doze'], default='reboot')
    args = p.parse_args()
    assert args.serial.startswith('emulator-'), 'Emulator only; never reboot a user phone'
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = [args.adb, '-s', args.serial]
    package = 'com.mobilefinetuner.continuity'
    def adb(*cmd, timeout=60):
        return subprocess.check_output(prefix + list(cmd), text=True, timeout=timeout)
    def read(path):
        return json.loads(adb('shell', 'run-as', package, 'cat', path))
    def until(fn, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                value = fn()
                if value:
                    return value
            except (subprocess.SubprocessError, json.JSONDecodeError, KeyError):
                pass
            time.sleep(.3)
        raise TimeoutError('Android lifecycle wait expired')
    setup = adb('shell', 'am', 'instrument', '-w', '-e', 'class',
                'com.mobilefinetuner.visualizer.continuity.RebootSetup',
                package + '.test/androidx.test.runner.AndroidJUnitRunner')
    (args.output / 'setup.log').write_text(setup)
    assert 'OK (1 test)' in setup, setup
    task = read('files/reboot_task.json')['task']
    # Instrumentation completion force-stops its app. Exercise the normal UI
    # Resume instead of mistaking that test-runner stop for an Android reboot.
    def nodes():
        adb('shell', 'uiautomator', 'dump', '/sdcard/mft-reboot-window.xml')
        return list(ET.fromstring(adb('shell', 'cat', '/sdcard/mft-reboot-window.xml')).iter('node'))
    def tap(node):
        x1, y1, x2, y2 = map(int, re.findall(r'\d+', node.attrib['bounds']))
        adb('shell', 'input', 'tap', str((x1 + x2) // 2), str((y1 + y2) // 2))
    def find_text(text):
        for _ in range(16):
            match = next((n for n in nodes() if n.attrib.get('text') == text and visible_node(n)), None)
            if match is not None: return match
            adb('shell', 'input', 'swipe', '540', '1800', '540', '900', '300')
        raise AssertionError('UI text not found: ' + text)
    def open_selected():
        # Use the same public task route as a notification, not a hidden task mutation.
        adb('shell', 'am', 'start', '-W', '-n',
            package + '/com.mobilefinetuner.visualizer.continuity.TrainingActivity', '--es', 'task', task)
        def selected_visible():
            if any(n.attrib.get('text') == 'Task: ' + task for n in nodes()):
                return True
            # Reopening the existing activity preserves its scroll offset.
            adb('shell', 'input', 'swipe', '540', '800', '540', '1900', '300')
            return False
        until(selected_visible, 30)
    def start_selected(automatic):
        for label, desired in [('Continue with scheduled background slices', automatic),
                               ('Train only while charging', False)]:
            for _ in range(16):
                check = checkbox_for_label(nodes(), label)
                if check is not None:
                    if (check.attrib['checked'] == 'true') != desired: tap(check)
                    break
                adb('shell', 'input', 'swipe', '540', '1800', '540', '900', '300')
            else:
                raise AssertionError('Visible checkbox not found: ' + label)
        adb('shell', 'pm', 'grant', package, 'android.permission.POST_NOTIFICATIONS')
        tap(find_text('Start / Resume'))
    open_selected()
    start_selected(True)
    def state():
        root = f'files/training/{task}/run'
        pointer = read(root + '/LATEST')
        return read(root + '/training_state/' + pointer['current'] + '/state.json')
    before = until(lambda: (s if 1 <= (s := state())['trainer']['completed_updates'] < 300 else None), 60)
    (args.output / 'before.json').write_text(json.dumps(before, indent=2))
    if args.mode == 'paused-doze':
        tap(find_text('Pause and disable background continuation'))
        paused = until(lambda: (s if (s := state())['status'] == 'suspended' else None), 60)
        at = paused['trainer']['completed_updates']
        assert 0 < at < 300
        ctl = read(f'files/training/{task}/control.json')
        assert not ctl['automatic'] and not ctl['enabled']
        try:
            adb('shell', 'dumpsys', 'battery', 'unplug')
            adb('shell', 'input', 'keyevent', '223')
            entry = adb('shell', 'dumpsys', 'deviceidle', 'force-idle', 'deep')
            (args.output / 'force-idle.log').write_text(entry)
            def idle():
                return adb('shell', 'cmd', 'deviceidle', 'get', 'deep').strip()
            assert idle() == 'IDLE', 'Did not enter deep idle'
            time.sleep(20)
            assert idle() == 'IDLE' and state() == paused, 'Paused state changed in deep idle'
        finally:
            adb('shell', 'dumpsys', 'deviceidle', 'unforce')
            adb('shell', 'dumpsys', 'battery', 'reset')
            adb('shell', 'input', 'keyevent', '224')
            adb('shell', 'wm', 'dismiss-keyguard')
        open_selected()
        assert not read(f'files/training/{task}/control.json')['enabled']
        start_selected(False)
        after = until(lambda: (s if (s := state())['trainer']['completed_updates'] > at else None), 60)
        tap(find_text('Pause and disable background continuation'))
        until(lambda: state()['status'] in ('completed', 'suspended'), 60)
        (args.output / 'paused.json').write_text(json.dumps(paused, indent=2))
        (args.output / 'after.json').write_text(json.dumps(after, indent=2))
        (args.output / 'result.json').write_text(json.dumps(dict(task=task, passed=True,
            mode=args.mode, deep_state='IDLE', observation_seconds=20, paused_updates=at,
            resumed_updates=after['trainer']['completed_updates'], model='tiny Qwen',
            scope='User-paused task remains paused in forced deep idle; explicit resume after wake.'), indent=2))
        return
    if args.mode == 'user-stop':
        adb('shell', 'cmd', 'activity', 'stop-app', package)
        stopped = state()['trainer']['completed_updates']
        time.sleep(20) # Android can suppress jobs entirely after Stop; no callback is guaranteed.
        assert state()['trainer']['completed_updates'] == stopped
        open_selected()
        until(lambda: not read(f'files/training/{task}/control.json')['enabled'], 30)
        assert not read(f'files/training/{task}/control.json')['automatic']
        assert state()['trainer']['completed_updates'] == stopped
        start_selected(False)
        after = until(lambda: (s if (s := state())['trainer']['completed_updates'] > stopped else None), 60)
        tap(find_text('Pause and disable background continuation'))
        until(lambda: state()['status'] in ['suspended', 'completed'], 60)
        (args.output / 'after.json').write_text(json.dumps(after, indent=2))
        (args.output / 'result.json').write_text(json.dumps(dict(task=task, passed=True,
            mode='Android Task Manager Stop via cmd activity stop-app',
            stopped_updates=stopped, explicit_resume_updates=after['trainer']['completed_updates'],
            environment='Android 15 arm64 emulator'), indent=2))
        print('PASS user Stop disables automatic work; explicit UI Resume continues', task)
        return
    adb('reboot')
    adb('wait-for-device', timeout=120)
    until(lambda: adb('shell', 'getprop', 'sys.boot_completed').strip() == '1', 120)
    adb('shell', 'input', 'keyevent', '82')
    # WorkManager owns the persisted job: do not explicitly start/resume it here.
    after = until(lambda: (s if (s := state())['status'] == 'completed' else None), 240)
    (args.output / 'after.json').write_text(json.dumps(after, indent=2))
    assert after['trainer']['completed_updates'] == 300
    control = read(f'files/training/{task}/control.json')
    assert control['automatic'] and control['enabled']
    (args.output / 'result.json').write_text(json.dumps(dict(task=task, passed=True,
        before_updates=before['trainer']['completed_updates'], after_updates=300,
        environment='Android 15 arm64 emulator; not physical power loss'), indent=2))
    print('PASS reboot continuation', task)


if __name__ == '__main__':
    main()
