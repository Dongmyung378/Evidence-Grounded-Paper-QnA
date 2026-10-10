"""A reproduction must execute only the measured source snapshot."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_corpus_benchmark import prepare_runtime


class CorpusReplayTests(unittest.TestCase):
    def make_snapshot(self, root):
        snapshot = root / 'snapshot'
        path = snapshot / 'scripts' / 'measured.py'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'MEASURED = True\n')
        hashes = {'scripts/measured.py': hashlib.sha256(path.read_bytes()).hexdigest()}
        (snapshot / 'snapshot.json').write_text(json.dumps({'sha256': hashes}), encoding='utf-8')
        return snapshot, hashes

    def test_current_files_cannot_contaminate_measured_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot, hashes = self.make_snapshot(root)
            current = root / 'current'
            (current / 'scripts').mkdir(parents=True)
            (current / 'scripts' / 'measured.py').write_bytes(b'MEASURED = False\n')
            (current / 'scripts' / 'newer.py').write_bytes(b'NEWER = True\n')
            clone = root / 'clone'
            prepare_runtime(snapshot, clone, source_root=current, data_directories=())
            self.assertFalse((clone / 'scripts' / 'newer.py').exists())
            self.assertEqual(hashlib.sha256((clone / 'scripts' / 'measured.py').read_bytes()).hexdigest(),
                             hashes['scripts/measured.py'])

    def test_changed_or_outside_snapshot_sources_are_rejected(self):
        for mode in ('changed', 'outside'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                snapshot, hashes = self.make_snapshot(root)
                if mode == 'changed':
                    (snapshot / 'scripts' / 'measured.py').write_bytes(b'changed\n')
                else:
                    outside = root / 'outside.py'
                    outside.write_bytes(b'outside\n')
                    hashes = {'../outside.py': hashlib.sha256(outside.read_bytes()).hexdigest()}
                    (snapshot / 'snapshot.json').write_text(json.dumps({'sha256': hashes}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    prepare_runtime(snapshot, root / 'clone', data_directories=())


if __name__ == '__main__':
    unittest.main()
