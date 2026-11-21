import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from phase_metrics import records


class PhaseMetricsTest(unittest.TestCase):
    def fixture(self, root, end):
        path = root / 'events.jsonl'
        value = dict(wall_sec=.01, rss_sample_count=2, logical_read_bytes=0,
            logical_write_bytes=42, peak_sampled_rss_bytes=1384824832,
            rss_start_bytes=1384562688, rss_end_bytes=end, os_io_bytes=None)
        path.write_text(json.dumps(dict(type='checkpoint_committed', session_id='test', event_seq=2, save_metrics=value)) + '\n')
        return path.read_bytes()

    def test_consistent_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root, 1384824832)
            row = list(records(root))[0]
            self.assertTrue(row['rss_endpoints_consistent'])
            self.assertIsNone(row['os_disk_read_bytes'])

    def test_real_historical_inconsistency_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); before = self.fixture(root, 1384841216)
            with self.assertRaisesRegex(ValueError, 'RSS endpoint exceeds'):
                list(records(root))
            self.assertEqual(before, (root / 'events.jsonl').read_bytes())

    def test_explicit_historical_export_flags_without_rewriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); before = self.fixture(root, 1384841216)
            row = list(records(root, strict=False))[0]
            self.assertFalse(row['rss_endpoints_consistent'])
            self.assertIn('INVALID_RSS_ENDPOINTS', row['measurement_warning'])
            self.assertEqual(1384824832, row['peak_sampled_rss_bytes'])
            self.assertEqual(1384841216, row['rss_end_bytes'])
            self.assertEqual(before, (root / 'events.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
