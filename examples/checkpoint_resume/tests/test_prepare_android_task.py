import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_android_task import prepare


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"vocab_size": 10}')
        (self.model / 'model.safetensors').write_bytes(b'fixture-only-not-real-weights')
        self.data = self.root / 'data.jsonl'
        self.data.write_text('{"ids": [1, 2], "mask": [1, 1]}\n')
        self.output = self.root / 'output'

    def test_layout_and_no_overwrite(self):
        prepare(self.model, self.data, self.data, self.output, 'qwen')
        spec = json.loads((self.output / 'experiment.json').read_text())
        self.assertEqual(spec['batch_size'], 1)
        self.assertEqual(spec['pause_grace_seconds'], 30)
        self.assertEqual((self.output / 'train.jsonl').read_bytes(), self.data.read_bytes())
        with self.assertRaises(ValueError):
            prepare(self.model, self.data, self.data, self.output, 'qwen')

    def test_invalid_data_leaves_no_output(self):
        for row in ({'ids': [1, 10], 'mask': [1, 1]},
                    {'ids': [1, 2], 'mask': [1]},
                    {'ids': [1, 2], 'mask': [1, 0]}):
            self.data.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                prepare(self.model, self.data, self.data, self.output, 'qwen')
            self.assertFalse(self.output.exists())

    def test_explicit_bounded_pause_grace(self):
        with self.assertRaises(ValueError):
            prepare(self.model, self.data, self.data, self.output, 'qwen', pause_grace=421)
        spec = prepare(self.model, self.data, self.data, self.output, 'qwen', pause_grace=300)
        self.assertEqual(spec['pause_grace_seconds'], 300)


if __name__ == '__main__':
    unittest.main()
