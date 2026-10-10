# Separate Linux reproduction

[English](README.md) | [한국어](README_KO.md)

Twelve preselected language outputs across four papers reproduced the Windows answers, aligned source units and citation sources exactly. Both negative pairs from paper 013 were included. Dense embeddings were regenerated from all thirty papers. The model process used Python 3.12.15, Torch 2.5.1 CPU and two Torch threads in a network-disabled Linux container with a read-only pinned model cache. It ran on the same physical host as Windows.

[Predictions](linux-predictions.json), [comparison and selection](comparison.json) and the [measured entrypoint](measured_runner.py) preserve this sample. The entrypoint expects its restored measured snapshot at `/repo`, pinned offline models and a writable `/repo/outputs` directory. Its runtime hashes match all 80 files of the measured final snapshot. The outer replay and artifact validator were subsequently repaired without changing inference sources.

This was an agent-operated environment reproduction. It is not independent human factual review, full 360-output model reproduction or a latency benchmark. Timings and floating-point reranker scores are excluded from the exact output comparison. Repeating the entire published evaluation split is supported by `scripts/replay_corpus_benchmark.py --variant final --split holdout --offline --output outputs/final-replay.json` from the repository root.
