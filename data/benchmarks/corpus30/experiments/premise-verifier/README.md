# Premise verifier feasibility trial

[English](README.md) | [한국어](README_KO.md)

The classifier was rejected. It refused false premises in all 10 paired development groups, but accepted correct premises in only 2/10 paired groups. The predeclared acceptance rule required at least 9/10 correct premises and 8/10 false-premise refusals. Excessive rejection of valid requests outweighed the negative-case score.

The trial uses the English [nli-MiniLM2-L6-H768 model](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768), revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, with an entailment threshold of 0.8. It classifies 12 BM25 candidate passages per claim from the selected paper. Korean claims are reverse-translated with the existing pinned NLLB weights. The classifier itself is English-only; the reverse translation also distorted technical terms.

[Protocol](protocol.json) freezes 40 language inputs, positive controls and thresholds before inference. [Predictions](predictions.json) preserve probabilities, selected passages and translations. [Hashes](freeze.json) protect those records. The trial covers the ten disclosed development papers, with AI-assisted source curation and no independent human review. It is separate from complete-answer scoring and is not enabled in production.

With inference dependencies installed, replay using `python -B scripts/experiment_premise_verifier.py --offline --output outputs/premise-verifier-replay.json`. Offline replay additionally requires this classifier revision in the Hugging Face cache. Omit `--offline` for its first download. Model weights and host paths are excluded from publication.
