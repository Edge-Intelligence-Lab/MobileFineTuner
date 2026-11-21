"""Record observed UI actions on a disposable emulator; never edit private task data."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adb', required=True)
    p.add_argument('--serial', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--click')
    p.add_argument('--scroll', choices=['down', 'up'])
    a = p.parse_args()
    if not a.serial.startswith('emulator-'):
        raise ValueError('Disposable emulator only')
    a.output.mkdir(parents=True, exist_ok=True)
    adb = [a.adb, '-s', a.serial]
    stamp = str(time.time_ns())
    def command(*args):
        return subprocess.check_output(adb + list(args), timeout=30)
    def snapshot(suffix):
        command('shell', 'uiautomator', 'dump', '/sdcard/mft-ui.xml')
        raw = command('shell', 'cat', '/sdcard/mft-ui.xml')
        (a.output / (stamp + '-' + suffix + '.xml')).write_bytes(raw)
        return list(ET.fromstring(raw).iter('node'))
    nodes = snapshot('before')
    action = dict(time_ns=stamp, click=a.click, scroll=a.scroll)
    if a.click:
        found = [n for n in nodes if a.click in (n.get('text'), n.get('content-desc')) and n.get('enabled') == 'true']
        if len(found) != 1:
            raise ValueError(f'Expected one visible enabled match, got {len(found)} for {a.click}')
        bounds = list(map(int, re.findall(r'\d+', found[0].get('bounds'))))
        action['bounds'] = bounds
        x1, y1, x2, y2 = bounds
        command('shell', 'input', 'tap', str((x1+x2)//2), str((y1+y2)//2))
    if a.scroll:
        height = 1800 if a.scroll == 'down' else 800
        command('shell', 'input', 'swipe', '540', str(height), '540', str(2600-height), '300')
    with (a.output / 'actions.jsonl').open('a') as stream:
        stream.write(json.dumps(action) + '\n')
    nodes = snapshot('after')
    for n in nodes:
        if n.get('text') or n.get('content-desc'):
            print(json.dumps({k: n.get(k) for k in ('text', 'content-desc', 'bounds', 'checked', 'enabled')}))
    (a.output / (stamp + '.png')).write_bytes(command('exec-out', 'screencap', '-p'))


if __name__ == '__main__':
    main()
