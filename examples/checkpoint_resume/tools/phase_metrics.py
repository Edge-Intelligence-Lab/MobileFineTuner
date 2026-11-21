#!/usr/bin/env python3
"""Export measured checkpoint phases, without confusing logical I/O with disk I/O."""
import argparse
import csv
import json
from pathlib import Path


def records(root, strict=True):
    for file in sorted(root.rglob('events.jsonl')):
        for line in file.read_text().splitlines():
            e = json.loads(line)
            for key, phase in [('save_metrics', 'durable_save'), ('state_restore_metrics', 'state_restore'),
                               ('metrics', 'resume_ready')]:
                if key not in e or (key == 'metrics' and e['type'] != 'resume_ready'):
                    continue
                m = e[key]
                assert m['wall_sec'] >= 0 and m['rss_sample_count'] >= 2
                assert m['logical_read_bytes'] >= 0 and m['logical_write_bytes'] >= 0
                rss = m['peak_sampled_rss_bytes']
                consistent = rss is None or rss >= max(m['rss_start_bytes'] or 0, m['rss_end_bytes'] or 0)
                if strict and not consistent:
                    raise ValueError(f'RSS endpoint exceeds sampled peak: {file}:{e["event_seq"]}; '
                                     'use --include-invalid-rss to retain explicitly flagged historical values')
                disk = m['os_io_bytes'] or {}
                row = dict(source=str(file), session=e['session_id'], event_seq=e['event_seq'], phase=phase)
                row.update({k: v for k, v in m.items() if k != 'os_io_bytes'})
                row.update(os_disk_read_bytes=disk.get('read_bytes'), os_disk_write_bytes=disk.get('write_bytes'))
                row.update(rss_endpoints_consistent=consistent,
                           measurement_warning='' if consistent else 'INVALID_RSS_ENDPOINTS; original values retained')
                yield row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-invalid-rss', action='store_true', help='Export flagged old inconsistent RSS without altering raw values')
    args = p.parse_args()
    rows = list(records(args.root, strict=not args.include_invalid_rss))
    if not rows:
        raise SystemExit('No phase measurements found')
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} physical phase records -> {args.output}')
    invalid = sum(not r['rss_endpoints_consistent'] for r in rows)
    if invalid:
        print(f'WARNING: {invalid} rows have invalid RSS endpoints; do not treat their RSS as validated')


if __name__ == '__main__':
    main()
