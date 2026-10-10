# Licensed PDF for deployment verification

[English](README.md) | [한국어](README_KO.md)

`chen20j.pdf` is the original PMLR PDF of **A Simple Framework for Contrastive
Learning of Visual Representations**, by Ting Chen, Simon Kornblith, Mohammad
Norouzi and Geoffrey Hinton, Proceedings of Machine Learning Research 119,
1597–1607, 2020.

[Publisher record](https://proceedings.mlr.press/v119/chen20j.html) and
[original PDF](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf).
The publisher licenses this work under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
as specified in [PMLR's publication agreement](https://proceedings.mlr.press/pmlr-license-agreement.html).
The PDF is redistributed unchanged; its SHA-256 is declared in `real_pdf_cases.json`.
Full source attribution and extraction metadata also appear in
`../real_paper_sources/real-dev-a.json`.

These two DEV questions verify original PDF upload, parsing, paper-scoped
citations and lexical output coverage through a running HTTP API. They are
disclosed integration cases, not unseen answer accuracy or independent human review.

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment-real-pdf-check.json
```
