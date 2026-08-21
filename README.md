# Evidence-Grounded Paper Q&A

An evidence-grounded multilingual Q&A service for one English academic paper
at a time.

## Problem

Reading an academic paper and finding the exact evidence needed to answer a
question can take significant time.

Many paper summarization tools provide a general summary but do not clearly
show which parts of the paper support each answer.

## Solution

This project accepts one English academic paper PDF and provides:

- Text-based paper overview
- Korean and English questions
- Korean and English answers
- Multilingual evidence retrieval
- BM25 and dense retrieval
- Hybrid retrieval
- Reranked evidence
- Page and section references
- Abstention when the paper does not contain enough evidence

## Core Pipeline

```text
PDF Upload
    ↓
Text Parsing
    ↓
Page and Section Metadata
    ↓
Chunking
    ↓
BM25 + Multilingual Dense Retrieval
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Evidence Selection
    ↓
Grounded Answer
```

## Example

### User Question

```text
이 논문의 주요 연구 목적은 무엇인가?
```

### Answer

```text
이 논문의 주요 연구 목적은 제안된 방법의 성능을 평가하는 것입니다.
```

### Evidence

```text
Page: 3
Section: Introduction
Chunk ID: paper-001-page-003-chunk-002

Original text:
The main objective of this study is to evaluate the proposed method.
```

## Supported Scope

- Input: one English academic paper PDF
- Processing: one paper per session
- Questions: Korean or English
- Answers: Korean or English
- Evidence: original English text with page and section metadata
- File size limit: 20MB
- PDF type: text-extractable PDFs

## Limitations

The MVP does not fully support:

- Scanned PDFs
- OCR
- Image or graph understanding
- Complete mathematical formula understanding
- Multiple papers at the same time
- Cross-paper comparison
- Korean, Chinese, or Japanese paper input
- User login and account management
- Large-scale paper search

## Evaluation Plan

The system will compare:

- BM25 retrieval
- Dense retrieval
- Hybrid retrieval
- Hybrid retrieval with reranking

Planned metrics:

- Recall@1
- Recall@5
- Recall@10
- MRR or nDCG
- Evidence correctness
- Abstention accuracy

## Project Status

This project is currently in the planning and implementation stage.

## Planned Technologies

- Python
- FastAPI
- Streamlit
- BM25
- Multilingual embedding model
- Cross-Encoder reranker
- FAISS
- SQLite
- Docker