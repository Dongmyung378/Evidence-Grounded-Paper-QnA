# 스크립트

[English](README.md) | [한국어](README_KO.md)

이 디렉터리에는 현재 사용하는 실행, 평가와 검증 진입점만 남겼습니다. 과거 단계별 스크립트는 포트폴리오 브랜치에서 제거했습니다.

## 주요 명령

```bash
python -B scripts/run_ingestion_pipeline.py
python -B scripts/ask_paper.py --help
python -B scripts/verify_project.py
```

모델을 사용하는 긴 평가는 가벼운 전체 검증과 분리했습니다.

```bash
python -B scripts/evaluate_frozen_retrieval.py
python -B scripts/evaluate_abstention.py
python -B scripts/evaluate_grounded_generation.py
python -B scripts/run_container_qna_benchmark.py
```

빌더는 저장한 실행 결과와 검토 라벨을 최종 산출물로 만들고, 검증기는 모델을 내려받지 않고 산출물 일관성을 확인합니다.
