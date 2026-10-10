# 배포 검증용 이용 허가 PDF

[English](README.md) | [한국어](README_KO.md)

`chen20j.pdf`는 Ting Chen, Simon Kornblith, Mohammad Norouzi와 Geoffrey Hinton의 **A Simple Framework for Contrastive Learning of Visual Representations** 원본 PMLR PDF입니다. Proceedings of Machine Learning Research 119권, 1597–1607쪽에 2020년 게재됐습니다.

[출판사 페이지](https://proceedings.mlr.press/v119/chen20j.html)와 [원본 PDF](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)를 제공합니다. [PMLR 출판 계약](https://proceedings.mlr.press/pmlr-license-agreement.html)에 따라 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) 라이선스를 적용합니다. PDF는 변경 없이 재배포하며 SHA-256을 `real_pdf_cases.json`에 기록했습니다. 전체 출처와 추출 메타데이터는 `../real_paper_sources/real-dev-a.json`에서도 확인할 수 있습니다.

개발 질문 2개는 실행 중인 HTTP API에서 원본 PDF 업로드와 분석, 해당 논문으로 제한한 인용 및 출력 어휘를 검사합니다. 공개된 통합 검증 사례이며 새로운 논문의 답변 정확도나 독립적인 사람의 검토를 뜻하지 않습니다.

저장소 루트의 호스트 환경에서 실행합니다.

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment-real-pdf-check.json
```
