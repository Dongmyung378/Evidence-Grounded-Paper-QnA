# Data Sources

## Purpose

This document records the sources, usage conditions, license information,
and selection decisions for the English academic papers and evaluation data
used in this project.

The project supports one English academic paper at a time and provides
Korean or English answers based on the paper's extractable text.

---

## Selected Sources

### 1. arXiv

- Official URL: https://arxiv.org/
- Data type: English academic paper source
- Primary use: Test papers for PDF upload, parsing, chunking, retrieval, and Q&A
- English papers: Yes
- PDF access: Available for individual papers
- Paper identifier: arXiv ID
- Evidence annotations: Not provided by arXiv
- License: Must be checked for each paper
- Decision: Selected as the primary source for test paper PDFs

Important notes:

- Each paper has a unique arXiv identifier.
- Each paper has a paper page and a PDF URL.
- Paper copyright and license conditions may differ by paper.
- The availability of a PDF does not automatically grant permission to
  redistribute it.
- Raw PDFs should not be committed to GitHub unless redistribution is
  explicitly permitted.
- The repository should store source URLs and metadata whenever possible.

### 2. QASPER

- Official dataset URL: https://huggingface.co/datasets/allenai/qasper
- Paper URL: https://arxiv.org/abs/2105.03011
- Data type: Scientific paper question-answering dataset
- Primary use: Retrieval and evidence evaluation
- Language: English
- Questions: Available
- Answers: Available
- Supporting evidence: Available
- Full paper text: Available in the dataset records
- Original PDF: Not automatically assumed; verify separately
- License: CC BY 4.0
- Decision: Selected as an evaluation data candidate

QASPER contains questions and answers about scientific research papers.
Its records include paper metadata, abstracts, full text, questions, answers,
and supporting evidence.

The evidence may refer to:

- Text paragraphs
- Tables
- Figures

Because the MVP is text-based, table and figure evidence will be excluded
from the initial evaluation pipeline.

---

## Current Dataset Status

### QASPER

The dataset has been downloaded locally and verified.

Available splits:

| Split | Number of Papers |
|---|---:|
| Train | 888 |
| Validation | 281 |
| Test | 416 |

Verified fields:

- `id`
- `title`
- `abstract`
- `full_text`
- `qas`
- `figures_and_tables`

Verified question and answer fields:

- `question`
- `question_id`
- `free_form_answer`
- `extractive_spans`
- `evidence`
- `highlighted_evidence`
- `unanswerable`

Local storage path:

```text
data/raw/qasper/
```

### Current Paper Corpus

The local robustness corpus contains 30 English text-extractable papers. The
complete paper-level title, domain, source, license, filename, and page-count
record is maintained in `data/metadata/paper_manifest.csv`.

| Corpus role | Paper IDs | Papers | Evaluation use |
|---|---|---:|---|
| Frozen benchmark | paper-001 to paper-010 | 10 | 40 verified English/Korean questions |
| Expansion robustness set | paper-011 to paper-030 | 20 | Parsing and model-backed retrieval smoke tests |
| Total | paper-001 to paper-030 | 30 | 563 pages and 2,463 processed chunks |

Expansion source caveats:

- `paper-020` was verified as arXiv:2608.26855v1. Its arXiv non-exclusive
  distribution license does not independently authorize this repository to
  redistribute the raw PDF, so the PDF remains local-only.
- `paper-026` was verified as arXiv:2607.03214v1 under CC BY-NC-ND 4.0;
  attribution, non-commercial, and no-derivatives restrictions apply.
- All former `check` values were resolved against official arXiv or journal
  license pages. Papers under the arXiv non-exclusive distribution license
  remain local-only because that license grants distribution rights to arXiv,
  not to this repository.
- Equation-heavy papers remain useful for parser robustness, but formula
  interpretation is still outside the MVP.

### Benchmark Example Paper

| Paper ID | Title | Domain | Source | Language | Text Extractable | License Status |
|---|---|---|---|---|---|---|
| paper-001 | Operational Non-identifiability of Single-epoch Low-rank RFI Mitigation: Controlled Failure-mode Analysis and HERA Evidence | Radio Astronomy / Data Analysis | https://arxiv.org/abs/2601.00046v2 | English | Yes | Check before redistribution |

Local file:

```text
data/raw/papers/paper-001.pdf
```

Paper metadata:

- Page count: 13
- Abstract: Available
- Main body text: Available
- Section headings: Available
- Text extraction: Verified
- Image understanding: Not included in the MVP

---

## Data Selection Criteria

A paper or dataset should satisfy the following conditions:

- Contains English academic papers
- Has a stable source URL
- Provides a stable paper or record identifier
- Allows access to the paper text or PDF
- Allows text extraction from the PDF or provides full text
- Has a clear usage policy or license
- Can be used for local development and evaluation
- Can be linked to metadata such as title, source, and identifier
- Contains enough text for retrieval and evidence selection

---

## Evaluation Data Requirements

Each evaluation sample should eventually contain:

- `paper_id`
- `question`
- `question_language`
- `gold_answer`
- `gold_evidence_ids`
- `answerable`
- `source`
- `license`
- `evidence_type`

Supported initial evidence types:

```text
text
```

Excluded evidence types for the MVP:

```text
figure
table
formula
image
```

Recommended evaluation record format:

```json
{
  "paper_id": "paper-001",
  "question": "What is the main limitation of the proposed method?",
  "question_language": "en",
  "gold_answer": "The method cannot fully separate science signals from structured interference when their subspaces overlap.",
  "gold_evidence_ids": [
    "paper-001-page-2-chunk-004"
  ],
  "answerable": true,
  "evidence_type": "text",
  "source": "arxiv",
  "license": "to-be-verified"
}
```

---

## Data Handling Policy

### Raw PDF Files

Raw PDFs should be stored locally during development.

Do not commit raw PDFs to the public repository unless their licenses
explicitly allow redistribution.

Recommended local directory:

```text
data/raw/papers/
```

### QASPER Dataset

The downloaded QASPER dataset should be stored locally and excluded from
Git version control unless redistribution is confirmed to be permitted.

Recommended local directory:

```text
data/raw/qasper/
```

### Public Repository

The public repository should contain:

- Source URLs
- Paper identifiers
- Dataset names
- License information
- Metadata files
- Download instructions
- Preprocessing scripts
- Evaluation scripts

The public repository should not contain:

- Unverified copyrighted PDFs
- Private files
- API keys
- Unnecessary raw datasets
- User-uploaded papers

---

## Git Ignore Rules

The following paths should be added to `.gitignore`:

```gitignore
data/raw/qasper/
data/raw/papers/*.pdf
```

---

## License and Copyright Notes

- The license of each arXiv paper must be checked individually.
- QASPER is marked as CC BY 4.0 on its dataset card.
- License requirements must be followed when redistributing or modifying data.
- Source attribution must be included in the README.
- Raw data should remain local when redistribution rights are unclear.
- This project should not imply endorsement by arXiv, Hugging Face, or the
  original paper authors.

---

## Decision Summary

| Source | Decision | Reason |
|---|---|---|
| arXiv | Selected | Suitable source for English academic paper PDFs |
| QASPER | Selected | Provides questions, answers, full text, and supporting evidence |
| Figure and table evidence | Excluded from MVP | The initial system is text-based |
| Multiple-paper processing | Excluded from MVP | The service processes one paper per session |

---

## Next Data Tasks

1. Add more English test papers from different research domains.
2. Verify the license status of each selected paper.
3. Confirm that each PDF has extractable text.
4. Inspect QASPER evidence types.
5. Filter out figure and table evidence for the MVP.
6. Create the project-specific evaluation records.
