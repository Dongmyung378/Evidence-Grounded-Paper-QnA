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


def prepare_runtime(snapshot, clone, source_root=ROOT, data_directories=(
        'data/processed', 'data/metadata', 'data/benchmarks/corpus30')):
    snapshot, clone, source_root = map(Path, (snapshot, clone, source_root))
    manifest = json.loads((snapshot / 'snapshot.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        path = (snapshot / name).resolve()
        if (not path.is_relative_to(snapshot.resolve()) or
                sha(path) != expected):
            raise ValueError(f'Changed or invalid snapshot file: {name}')
    clone.mkdir(parents=True, exist_ok=True)
    for directory in data_directories:
        shutil.copytree(source_root / directory, clone / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.npz', '*.npy'))
    for name in manifest['sha256']:
        destination = (clone / name).resolve()
        if not destination.is_relative_to(clone.resolve()):
            raise ValueError(f'Invalid snapshot destination: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, destination)
    observed = {p.relative_to(clone).as_posix(): sha(p)
        for directory in ('app', 'scripts', 'config')
        for p in (clone / directory).rglob('*')
        if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts}
    if observed != manifest['sha256']:
        raise ValueError('Prepared runtime differs from the measured snapshot')
    return manifest


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
    with tempfile.TemporaryDirectory(prefix='corpus-replay-', dir=ROOT / 'outputs') as folder:
        clone = Path(folder)
        prepare_runtime(snapshot, clone)
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
