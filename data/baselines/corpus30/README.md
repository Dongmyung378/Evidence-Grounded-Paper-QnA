# Measured corpus30 runtimes

[English](README.md) | [한국어](README_KO.md)

The labels "Pre-study implementation" and "Published implementation" describe the project's own system before the expansion study and after publication decisions. The snapshot directory names remain `baseline/` and `final/` to preserve reproducibility.

`baseline/` preserves 75 source/configuration files from the original commit plus the frozen benchmark runner. `final/` preserves 80 measured files after the three strategy trials were rejected. `development-v3/` records the last rejected trial. Each `snapshot.json` lists the exact bytes; altering these files invalidates recorded outputs.

The outer replay command restores only files in the selected snapshot. Its post-measurement preparation fix prevents new repository files from contaminating the older snapshot; it does not alter measured inference sources. The older replay helper inside a snapshot is preserved for provenance and is not called by this outer preparation step.

[Benchmark and results](../../benchmarks/corpus30/README.md) explain scope and exclusions. Private notes, raw PDFs, model weights and embedding caches are not included.
