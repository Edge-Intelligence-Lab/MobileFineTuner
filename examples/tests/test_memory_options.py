"""Validate memory options without loading any model or dataset assets.

Usage: python3 examples/tests/test_memory_options.py QWEN_CLI GEMMA270_CLI GEMMA1B_CLI
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MemoryOptions(unittest.TestCase):
    def rejected(self, executable, args, message):
        result = subprocess.run([str(executable), *args], cwd=self.workdir,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stdout)

    def test_gemma(self):
        for executable in self.executables[1:]:
            for args, message in [
                (['--offload_dir'], '--offload_dir requires a path'),
                (['--offload_dir='], '--offload_dir requires a path'),
                (['--offload_resident_mb=0'], '--offload_resident_mb must be positive'),
                (['--synthetic_smoke', '--offload_dir=unused'], 'memory options require'),
                (['--synthetic_smoke', '--offload_dir', 'unused'], 'memory options require'),
            ]:
                with self.subTest(executable=executable, args=args):
                    self.rejected(executable, args, message)

    def test_qwen(self):
        for args, message in [
            (['--offload_dir', 'unused', '--base_weight_storage', 'native'], 'offload requires'),
            (['--activation_checkpointing', '--lora_dropout', '0.1'], 'require --lora_dropout 0'),
        ]:
            with self.subTest(args=args):
                self.rejected(self.executables[0], args, message)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    MemoryOptions.executables = [Path(arg).resolve() for arg in sys.argv[1:]]
    if not all(path.is_file() for path in MemoryOptions.executables):
        raise SystemExit('Build all three executables first.')
    with tempfile.TemporaryDirectory(prefix='mft_memory_options_') as directory:
        MemoryOptions.workdir = directory
        unittest.main(argv=[sys.argv[0]], verbosity=2)
