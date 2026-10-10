# 품질 평가 산출물

[English](README.md) | [한국어](README_KO.md)

답변 동작을 재현하고 검토하는 데 필요한 공개 입력과 측정 출력을 보존합니다.

- `papers/`와 루트의 검증 자료 목록 및 질문은 작성한 통합 검증 자료입니다.
- `real_paper_sources/`, `real_paper_manifest.json`과 `real_paper_questions.jsonl`에는 출판사 원문 및 이중 언어 개발과 평가 질문이 있습니다.
- `deployment/`에는 이용 허가된 원본 PDF와 HTTP 사례가 있습니다.
- `runs/2026-10-10-final/`에는 측정 당시 고정한 구현과 최종 답변, 보고서 및 실행시간 근거를 보존합니다.
- `runs/answer-builder-pilot/`에는 사전에 기준을 정한 후보 모델 비교와 중단 판단을 보존합니다.
- `runs/2026-10-10/`와 `runs/2026-10-10-v2/`에는 공개 검증과 실행 정책에 필요한 보정 출처만 남깁니다.
- `reproduction/`에는 에이전트가 수행한 Linux 환경 재현과 답변 비교가 있습니다.

나머지 중간 실행과 장비별 로그는 Git에서 제외합니다. 날짜는 측정 자료를 식별하기 위한 것이며 작업 일지가 아닙니다. 고정한 파일은 바이트 단위로 보존하고 `.gitattributes`로 줄바꿈 변환을 방지합니다. 범위와 한계는 [현재 결과](../../docs/results_KO.md), [비교 실험](../../docs/experiments_KO.md)과 [평가 설계](../../docs/quality_benchmark_KO.md)에 정리했습니다.
