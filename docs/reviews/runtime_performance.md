# Local Runtime Performance

[English](runtime_performance.md) | [한국어](runtime_performance_KO.md)

## Why this change was needed

A local Korean question took 84.52 seconds and ended in a safe refusal. Inspection showed that the retrieval models had occupied most of the 6 GB GPU before the answer generator was loaded. The generator therefore moved to CPU and repeated an invalid response up to three times. The question language was not the direct cause.

## Runtime policy

The browser runtime uses a separate profile without changing the frozen baseline generation configuration.

- The multilingual embedding model runs on CPU.
- The Cross-Encoder reranker runs on CPU.
- The Qwen answer generator uses GPU when enough memory is available.
- Models and the uploaded paper's dense index are prepared during analysis.
- GPU generation keeps the reviewed 384-token budget and up to three validation attempts.
- CPU fallback uses at most two attempts, eight PyTorch threads, and a soft 18-second limit per attempt.
- The API reports retrieval time, generation time, attempt count, generation device, fallback state, and abstention source.
- The UI shows the actual abstention reason and the runtime breakdown.

The original `config/generation.json` remains the frozen baseline experiment input. The browser service reads `config/generation_runtime.json` through `config/runtime_qna.json`.

## Acceptance measurement

The complete HTTP flow was rerun on an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM and seed 378. The run used `paper-003.pdf` and the established English HTTP acceptance question.

| Measurement | Result |
|---|---:|
| Model preparation during analysis | 16.692 s |
| Answer request | 9.700 s |
| Retrieval | 0.970 s |
| Generation and validation | 8.729 s |
| Generation attempts | 1 |
| Generator device | CUDA |
| Answer result | Sufficient, cited evidence verified |

The earlier 84.52-second observation and this acceptance run used different papers and questions, so they are not a controlled benchmark pair. The new run demonstrates that the intended GPU allocation works and that the known answerable integration question still passes. It does not claim a universal latency bound.

The verified Korean counterpart `q-007-ko` was also run through the complete HTTP flow. It returned a sufficient Korean answer in 9.405 seconds: 0.751 seconds for retrieval and 8.653 seconds for one CUDA generation attempt. A differently worded Korean paraphrase completed in 11.893 seconds but was conservatively refused by the model. This confirms that Korean input does not cause the 84-second delay, while the known 0.5B answer-quality limitation still remains.

## Operational note

Restart FastAPI after changing runtime settings because model device selection happens when models are loaded. If other applications occupy most GPU memory, the generator can still fall back to CPU and respond more slowly. Set `PAPER_QNA_PREPARE_MODELS=0` only when analysis-time preparation must be disabled.
