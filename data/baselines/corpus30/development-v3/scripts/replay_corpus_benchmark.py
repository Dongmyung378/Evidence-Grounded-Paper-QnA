"""Replay a measured runtime snapshot in an isolated temporary source directory."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from corpus_benchmark import safe_output, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'final'], required=True)
    parser.add_argument('--split', choices=['dev', 'holdout'], required=True)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = safe_output(args.output)
    if target.exists():
        raise ValueError('Refusing to overwrite a completed replay')
    snapshot = ROOT / 'data/baselines/corpus30' / args.variant
    manifest = json.loads((snapshot / 'snapshot.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or sha(path) != expected:
            raise ValueError(f'Changed or invalid snapshot file: {name}')
    with tempfile.TemporaryDirectory(prefix='corpus-replay-', dir=ROOT / 'outputs') as folder:
        clone = Path(folder)
        for directory in ('app', 'scripts', 'config', 'data/processed', 'data/metadata', 'data/benchmarks/corpus30'):
            shutil.copytree(ROOT / directory, clone / directory,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.npz', '*.npy'))
        for name in manifest['sha256']:
            destination = clone / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, destination)
        result_path = clone / 'outputs' / 'replay.json'
        command = [sys.executable, '-B', str(clone / 'scripts/corpus_benchmark.py'), 'run',
            '--split', args.split, '--variant', args.variant + '-replay',
            '--output', str(result_path), '--threads', '8']
        if args.offline:
            command.append('--offline')
        subprocess.run(command, cwd=clone, check=True)
        shutil.copyfile(result_path, target)
    print(f'Replay saved: {target.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
